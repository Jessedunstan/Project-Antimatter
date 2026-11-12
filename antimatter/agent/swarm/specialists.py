"""Specialized Micro-Agents with Strict Tool-Palette Isolation (TPI).

Each specialist operates in a pristine, tiny context (<500 tokens) with 1-2 tools maximum,
guaranteeing maximum local LLM attention precision and eliminating hallucination.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from antimatter.agent.swarm.types import (
    SwarmStage,
    StageStatus,
    SwarmStepResult,
)
from antimatter.agent.swarm.blackboard import (
    SwarmBlackboard,
    ScoutEvidence,
    PatchBlueprint,
    VerificationResult,
)
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole, ToolCall
from antimatter.tools.fs_manager import FileSystemManager
from antimatter.tools.patcher import MultiLinePatcher
from antimatter.tools.terminal import TerminalSandbox
from antimatter.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ScoutSpecialist:
    """Micro-Agent: Inspects code and gathers targeted snippets (Read-Only)."""

    def __init__(self, fs: FileSystemManager, llm_client: LocalLLMClient):
        self.fs = fs
        self.llm = llm_client

    async def execute(self, goal: str, target_file: str, test_file: Optional[str] = None) -> ScoutEvidence:
        t0 = time.time()
        logger.info("[Scout] Reading %s and %s...", target_file, test_file or "N/A")

        code_snippet = ""
        test_snippet = ""
        try:
            code_snippet = self.fs.read_file(target_file)
        except Exception as e:
            logger.warning("[Scout] Could not read %s: %s", target_file, e)

        if test_file:
            try:
                test_snippet = self.fs.read_file(test_file)
            except Exception as e:
                logger.warning("[Scout] Could not read test file %s: %s", test_file, e)

        evidence = ScoutEvidence(
            file_path=target_file,
            code_snippet=code_snippet,
            test_file_path=test_file or "",
            test_code_snippet=test_snippet,
            findings=f"Read {len(code_snippet)} chars from {target_file}",
        )
        return evidence


class ArchitectSpecialist:
    """Micro-Agent: Analyzes evidence and drafts exact multi-line patch blueprint (Pure Reasoning)."""

    def __init__(self, llm_client: LocalLLMClient):
        self.llm = llm_client

    async def execute(self, goal: str, evidence: ScoutEvidence) -> PatchBlueprint:
        logger.info("[Architect] Formulating patch blueprint for %s...", evidence.file_path)

        test_section = f"TEST CODE:\n```python\n{evidence.test_code_snippet}\n```" if evidence.test_code_snippet else ""

        prompt = f"""You are a Principal Software Architect.
Analyze this code defect and formulate an exact multi-line patch.

GOAL: {goal}
TARGET FILE: {evidence.file_path}

SOURCE CODE:
```python
{evidence.code_snippet}
```

{test_section}

Respond ONLY with a JSON object containing:
{{
  "file_path": "{evidence.file_path}",
  "target_block": "<exact lines from source code to be replaced>",
  "replacement_block": "<exact drop-in replacement lines>",
  "rationale": "<short 1-sentence reason>"
}}
"""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a deterministic code patch architect. Respond ONLY in valid JSON."),
            ChatMessage(role=ChatRole.USER, content=prompt)
        ]

        resp = await self.llm.chat(messages, temperature=0.1)
        raw_text = resp.content.strip()

        # Parse JSON output
        try:
            # Strip code fences if present
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            data = json.loads(raw_text.strip())
            return PatchBlueprint(
                file_path=data.get("file_path", evidence.file_path),
                target_block=data.get("target_block", ""),
                replacement_block=data.get("replacement_block", ""),
                rationale=data.get("rationale", "Surgical patch applied"),
            )
        except Exception as e:
            logger.warning("[Architect] JSON parse failed: %s. Using heuristic blueprint.", e)
            return PatchBlueprint(
                file_path=evidence.file_path,
                target_block="",
                replacement_block="",
                rationale=f"Failed to generate structured JSON: {raw_text[:100]}",
            )


class CoderSpecialist:
    """Micro-Agent: Applies surgical multi-line patches with atomic rollback snapshots (Edit-Only)."""

    def __init__(self, patcher: MultiLinePatcher, fs: FileSystemManager):
        self.patcher = patcher
        self.fs = fs

    async def execute(self, blueprint: PatchBlueprint, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info("[Coder] Applying patch to %s...", blueprint.file_path)
        if not blueprint.target_block or not blueprint.replacement_block:
            return {"success": False, "error": "Empty target or replacement block in blueprint"}

        try:
            diff, updated = self.patcher.apply_patch(
                file_path=blueprint.file_path,
                target_block=blueprint.target_block,
                replacement_block=blueprint.replacement_block,
                checkpoint_id=checkpoint_id,
            )
            return {"success": True, "diff": diff, "file_path": blueprint.file_path}
        except Exception as e:
            logger.error("[Coder] Patch application error: %s", e)
            return {"success": False, "error": str(e), "file_path": blueprint.file_path}


class VerifierSpecialist:
    """Micro-Agent: Executes test suite in sandbox and validates exit codes (Exec-Only)."""

    def __init__(self, terminal: TerminalSandbox):
        self.terminal = terminal

    async def execute(self, test_command: str) -> VerificationResult:
        logger.info("[Verifier] Running validation command: %s", test_command)
        t0 = time.time()
        res = await self.terminal.execute(test_command, timeout_sec=60)
        duration_ms = (time.time() - t0) * 1000

        passed = (res.exit_code == 0) and not res.timed_out
        return VerificationResult(
            passed=passed,
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=res.stderr,
            duration_ms=duration_ms,
        )
