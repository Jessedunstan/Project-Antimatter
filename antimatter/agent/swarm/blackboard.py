"""Shared Blackboard State for Project Antimatter Autonomous Swarm.

Provides clean, structured inter-agent data passing without conversational transcript bloat.
"""

from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ScoutEvidence(BaseModel):
    file_path: str = ""
    target_lines: List[int] = Field(default_factory=list)
    code_snippet: str = ""
    test_file_path: str = ""
    test_code_snippet: str = ""
    findings: str = ""


class PatchBlueprint(BaseModel):
    file_path: str = ""
    target_block: str = ""
    replacement_block: str = ""
    rationale: str = ""


class VerificationResult(BaseModel):
    passed: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0


class SwarmBlackboard:
    """Centralized shared state store accessed by all swarm micro-specialists."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.scout_evidence = ScoutEvidence()
        self.patch_blueprint = PatchBlueprint()
        self.verification_result = VerificationResult()
        self.checkpoint_id: Optional[str] = None
        self.custom_state: Dict[str, Any] = {}
        self.history_log: List[Dict[str, Any]] = []

    async def update_scout(self, evidence: ScoutEvidence) -> None:
        async with self._lock:
            self.scout_evidence = evidence
            self.history_log.append({
                "stage": "scout",
                "timestamp": time.time(),
                "data": evidence.model_dump(),
            })

    async def update_blueprint(self, blueprint: PatchBlueprint) -> None:
        async with self._lock:
            self.patch_blueprint = blueprint
            self.history_log.append({
                "stage": "architect",
                "timestamp": time.time(),
                "data": blueprint.model_dump(),
            })

    async def update_verification(self, result: VerificationResult) -> None:
        async with self._lock:
            self.verification_result = result
            self.history_log.append({
                "stage": "verifier",
                "timestamp": time.time(),
                "data": result.model_dump(),
            })

    async def set_checkpoint(self, checkpoint_id: str) -> None:
        async with self._lock:
            self.checkpoint_id = checkpoint_id

    async def get_state(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "scout": self.scout_evidence.model_dump(),
                "blueprint": self.patch_blueprint.model_dump(),
                "verification": self.verification_result.model_dump(),
                "checkpoint_id": self.checkpoint_id,
                "custom": dict(self.custom_state),
            }
