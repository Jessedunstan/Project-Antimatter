"""Autonomous ReAct Agent Loop for Project Antimatter.

Synthesizes Local LLM Inference, Deterministic Tools, and Code Intelligence
into a sovereign offline software engineering runtime.
"""

from __future__ import annotations
import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Callable, List, Optional

from antimatter.agent.types import AgentStatus, AgentStep, AgentTrajectory
from antimatter.agent.compactor import ContextCompactor
from antimatter.agent.skills.loader import SkillLoader
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.tool_extractor import ToolCallExtractor
from antimatter.brain.types import (
    ChatMessage,
    ChatRole,
    CompletionChunk,
    ToolCall,
)
from antimatter.context.service import CodeIntelligenceService
from antimatter.core.config import get_settings
from antimatter.tools import build_default_tool_registry
from antimatter.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are Project Antimatter, a sovereign, 100% offline-first AI software engineering agent.
You possess deterministic workspace tools, Tree-sitter code intelligence, and a persistent sqlite-vec semantic index.

When solving tasks:
1. Think step-by-step.
2. If context is needed, query symbols or search code semantically before modifying files.
3. Apply precise multi-line patches or edits using the provided tools.
4. Verify changes using terminal commands (e.g. running pytest or python).
5. When done, summarize your solution clearly.
"""


class AgentReActLoop:
    """Orchestrates the ReAct (Reason + Act) loop across local inference and tools."""

    def __init__(
        self,
        llm_client: Optional[LocalLLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        code_intel: Optional[CodeIntelligenceService] = None,
        workspace_dir: Optional[Path] = None,
        compactor: Optional[ContextCompactor] = None,
        skill_loader: Optional[SkillLoader] = None,
        max_steps: int = 15,
    ):
        settings = get_settings()
        self.workspace_dir = (workspace_dir or settings.workspace_dir).resolve()
        self.llm_client = llm_client or LocalLLMClient(settings)
        self.tool_registry = tool_registry or build_default_tool_registry(self.workspace_dir)
        self.code_intel = code_intel or CodeIntelligenceService(workspace_dir=self.workspace_dir)
        self.compactor = compactor or ContextCompactor(max_token_budget=8192)
        self.skill_loader = skill_loader
        self.max_steps = max_steps

    async def run(
        self,
        task: str,
        on_step: Optional[Callable[[AgentStep], None]] = None,
    ) -> AgentTrajectory:
        """Execute an autonomous ReAct task to completion."""
        trajectory = AgentTrajectory(task=task, status=AgentStatus.THINKING)
        messages: List[ChatMessage] = []

        # 1. Retrieve initial code intelligence context
        context_snippets = []
        try:
            results = self.code_intel.semantic_search(task, limit=3)
            for r in results:
                context_snippets.append(
                    f"### Context: {r.chunk.file_path} ({r.chunk.name}, L{r.chunk.start_line}-L{r.chunk.end_line})\n"
                    f"```\n{r.chunk.content}\n```"
                )
        except Exception as e:
            logger.debug("Code intelligence pre-retrieval skipped: %s", e)

        system_content = SYSTEM_PROMPT_TEMPLATE
        if self.skill_loader:
            skills_prompt = self.skill_loader.format_all_for_system_prompt()
            if skills_prompt:
                system_content += f"\n{skills_prompt}"

        if context_snippets:
            system_content += "\n\n## Retrieved Codebase Context:\n" + "\n\n".join(context_snippets)

        messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system_content))
        messages.append(ChatMessage(role=ChatRole.USER, content=task))

        tool_defs = self.tool_registry.get_definitions()

        # 2. Main ReAct execution loop
        for step_idx in range(1, self.max_steps + 1):
            step = AgentStep(step_number=step_idx, timestamp=time.time())
            trajectory.steps.append(step)

            # Compact messages to remain strictly within local token budget
            messages = self.compactor.compact(messages)

            # Request LLM completion
            try:
                response_msg = await self.llm_client.chat(messages, tools=tool_defs)
            except Exception as e:
                trajectory.status = AgentStatus.ERROR
                trajectory.error_message = f"Inference failed at step {step_idx}: {e}"
                logger.error(trajectory.error_message)
                return trajectory

            # Extract tool calls
            tool_calls = response_msg.tool_calls or []
            if not tool_calls and response_msg.content:
                cleaned_text, extracted_tcs = ToolCallExtractor.extract_from_text(
                    response_msg.content, available_tools=tool_defs
                )
                tool_calls = extracted_tcs
                if tool_calls:
                    response_msg.content = cleaned_text

            step.thought = response_msg.content
            step.tool_calls = tool_calls
            messages.append(response_msg)

            # If no tool calls, task is complete
            if not tool_calls:
                trajectory.status = AgentStatus.COMPLETED
                trajectory.final_response = response_msg.content
                if on_step:
                    on_step(step)
                break

            # Execute tool calls
            trajectory.status = AgentStatus.EXECUTING_TOOL
            for tc in tool_calls:
                try:
                    tool_res = await self.tool_registry.dispatch(tc)
                except Exception as e:
                    tool_res = f"Error executing tool '{tc.name}': {e}"

                step.tool_results[tc.id] = tool_res
                messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        content=str(tool_res),
                        name=tc.name,
                        tool_call_id=tc.id,
                    )
                )

            if on_step:
                on_step(step)

            trajectory.status = AgentStatus.THINKING

        if trajectory.status != AgentStatus.COMPLETED and not trajectory.error_message:
            trajectory.status = AgentStatus.COMPLETED
            trajectory.final_response = messages[-1].content

        return trajectory

    def close(self) -> None:
        """Release any open file or database handles held by this loop."""
        if hasattr(self, "code_intel") and hasattr(self.code_intel, "vector_store"):
            try:
                self.code_intel.vector_store.close()
            except Exception:
                pass
