"""Data contracts for the Autonomous Agent ReAct Loop."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from antimatter.brain.types import ChatMessage, ToolCall


class AgentStatus(str, Enum):
    """Lifecycle status of agent execution."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    ERROR = "error"


class AgentStep(BaseModel):
    """An individual step in the agent's ReAct trajectory."""
    step_number: int
    thought: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    response_delta: str = ""
    timestamp: float = Field(default=0.0)


class AgentTrajectory(BaseModel):
    """Full execution trajectory of an agent run."""
    task: str
    status: AgentStatus = AgentStatus.IDLE
    steps: List[AgentStep] = Field(default_factory=list)
    final_response: Optional[str] = None
    error_message: Optional[str] = None
    checkpoint_id: Optional[str] = None
    total_tokens: int = 0
