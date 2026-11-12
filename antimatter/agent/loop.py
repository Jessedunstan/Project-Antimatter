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

CRITICAL OPERATIONAL RULES:
1. Think step-by-step before acting.
2. ALWAYS inspect files first using `read_file(path="...")` to read the exact lines before editing. Never guess or hallucinate target lines.
3. When modifying code, use `edit_file(path="...", target="...", replacement="...")` where `target` is an exact character-for-character substring of the file, OR use `write_file(path="...", content="...")` to write the complete corrected file.
4. Verify your modifications by executing tests with `run_command(command="...")`.
5. Only stop calling tools once the task is verified complete.
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
        allowed_tools: Optional[List[str]] = None,
        model: Optional[str] = None,
    ):
        settings = get_settings()
        self.workspace_dir = (workspace_dir or settings.workspace_dir).resolve()
        self.llm_client = llm_client or LocalLLMClient(settings)
        self.tool_registry = tool_registry or build_default_tool_registry(self.workspace_dir)
        self.code_intel = code_intel or CodeIntelligenceService(workspace_dir=self.workspace_dir)
        self.compactor = compactor or ContextCompactor(max_token_budget=8192)
        self.skill_loader = skill_loader
        self.max_steps = max_steps
        self.allowed_tools = allowed_tools
        self.model = model

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

        tool_defs = self.tool_registry.get_definitions(allowed_tools=self.allowed_tools)
        required_params_by_tool = {
            td.function.name: set(td.function.parameters.get("required", []))
            for td in tool_defs
        }
        retried_malformed = False
        retried_unapplied_code = False
        applied_file_edit = False

        # 2. Main ReAct execution loop
        for step_idx in range(1, self.max_steps + 1):
            step = AgentStep(step_number=step_idx, timestamp=time.time())
            trajectory.steps.append(step)

            # Compact messages to remain strictly within local token budget
            messages = self.compactor.compact(messages)

            # Request LLM completion
            try:
                if self.model is not None:
                    response_msg = await self.llm_client.chat(
                        messages, tools=tool_defs, model=self.model
                    )
                else:
                    response_msg = await self.llm_client.chat(messages, tools=tool_defs)
            except Exception as e:
                trajectory.status = AgentStatus.ERROR
                trajectory.error_message = f"Inference failed at step {step_idx}: {e}"
                logger.error(trajectory.error_message)
                return trajectory

            # Extract tool calls
            raw_tool_calls = response_msg.tool_calls or []
            raw_text = response_msg.content or ""
            if not raw_tool_calls and raw_text:
                cleaned_text, extracted_tcs = ToolCallExtractor.extract_from_text(
                    raw_text, available_tools=tool_defs
                )
                raw_tool_calls = extracted_tcs
                if raw_tool_calls:
                    response_msg.content = cleaned_text

            # Filter out empty/nameless tool calls when the tool requires parameters
            tool_calls: List[ToolCall] = []
            for tc in raw_tool_calls:
                if not tc.name or tc.name == "unknown_tool":
                    continue
                reqs = required_params_by_tool.get(tc.name, set())
                if reqs and not tc.arguments:
                    continue
                tool_calls.append(tc)

            step.thought = response_msg.content
            step.tool_calls = tool_calls
            messages.append(response_msg)

            # If no valid tool calls were produced, check if the model attempted a malformed tool call
            if not tool_calls:
                has_malformed_markup = any(
                    marker in raw_text for marker in ("<tool_call>", "Action:", '"arguments"', '"parameters"')
                )
                if has_malformed_markup and not retried_malformed and step_idx < self.max_steps:
                    retried_malformed = True
                    messages.append(
                        ChatMessage(
                            role=ChatRole.USER,
                            content=(
                                "Your previous tool call was malformed or missing required arguments. "
                                "Please invoke the tool again with valid JSON arguments matching its schema."
                            ),
                        )
                    )
                    if on_step:
                        on_step(step)
                    continue

                if (
                    "```python" in raw_text
                    and not applied_file_edit
                    and not retried_unapplied_code
                    and step_idx < self.max_steps
                    and any(t in required_params_by_tool for t in ("edit_file", "write_file"))
                ):
                    retried_unapplied_code = True
                    messages.append(
                        ChatMessage(
                            role=ChatRole.USER,
                            content=(
                                "You output a Python code block in text but have not saved any changes to disk yet. "
                                "Please call `write_file(path=..., content=...)` or `edit_file(path=..., target=..., replacement=...)` "
                                "to apply the fix to the file."
                            ),
                        )
                    )
                    if on_step:
                        on_step(step)
                    continue

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
                    if tc.name in ("edit_file", "write_file") and not str(tool_res).startswith("Error"):
                        applied_file_edit = True
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
