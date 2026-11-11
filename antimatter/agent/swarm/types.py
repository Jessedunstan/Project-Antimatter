"""Data contracts and schemas for the Antimatter Autonomous Swarm Engine."""

from __future__ import annotations
from enum import Enum
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from antimatter.brain.types import ToolDefinition


class SwarmStage(str, Enum):
    """The 4-stage standard decomposition lifecycle for local LLM micro-agents."""
    SCOUT = "scout"          # Codebase inspection & evidence gathering (read-only)
    ARCHITECT = "architect"  # Root-cause diagnosis & surgical patch blueprinting (pure reasoning)
    CODER = "coder"          # Atomic patch application & snapshot checkpointing (edit-only)
    VERIFIER = "verifier"    # Sandboxed test execution & diagnostic validation (exec-only)
    SYNTHESIS = "synthesis"  # Final report assembly and state consolidation


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MicroAgentSpec(BaseModel):
    """Configuration for an isolated specialist micro-agent."""
    stage: SwarmStage
    role_name: str
    system_prompt: str
    allowed_tools: List[str] = Field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.1


class SwarmStepResult(BaseModel):
    """Result of an individual specialist execution."""
    stage: SwarmStage
    status: StageStatus
    evidence: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    duration_sec: float = 0.0
    error_message: Optional[str] = None


class SwarmTrajectory(BaseModel):
    """Complete execution record of the autonomous multi-agent swarm."""
    task_id: str = Field(default_factory=lambda: f"swarm_{uuid.uuid4().hex[:8]}")
    user_goal: str
    stages: List[SwarmStepResult] = Field(default_factory=list)
    overall_status: StageStatus = StageStatus.PENDING
    total_duration_sec: float = 0.0
    checkpoint_id: Optional[str] = None
    final_report: str = ""
    retries_count: int = 0
