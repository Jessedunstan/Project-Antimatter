"""Isolated Subagent Worker Runtime for Project Antimatter.

Supports workspace branching, dedicated event queues, and independent ReAct loops.
"""

from __future__ import annotations
import asyncio
from enum import Enum
import logging
from pathlib import Path
import shutil
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from antimatter.agent.types import AgentStatus, AgentTrajectory
from antimatter.agent.loop import AgentReActLoop
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole
from antimatter.core.config import get_settings
from antimatter.tools import build_default_tool_registry
from antimatter.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from antimatter.agent.supervisor import SubagentSupervisor

logger = logging.getLogger(__name__)


class WorkspaceMode(str, Enum):
    """Subagent workspace isolation mode."""
    INHERIT = "inherit"  # Inherits parent workspace directly
    BRANCH = "branch"    # Isolated workspace directory branched from parent
    SHARE = "share"      # Shares repo with independent state


class SubagentSpec(BaseModel):
    """Specification to spawn a subagent."""
    id: str = Field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    name: str = Field(description="Unique type or display name of the subagent")
    role: str = Field(description="Role description (e.g. Code Reviewer, Database Specialist)")
    prompt: str = Field(description="Actionable task instruction")
    workspace_mode: WorkspaceMode = WorkspaceMode.INHERIT
    model: Optional[str] = None


class SubagentWorker:
    """An autonomous subagent with its own workspace and inbox."""

    def __init__(
        self,
        spec: SubagentSpec,
        supervisor: SubagentSupervisor,
        parent_workspace: Path,
    ):
        self.spec = spec
        self.supervisor = supervisor
        self.parent_workspace = parent_workspace.resolve()
        self.id = spec.id
        self.inbox: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.status: AgentStatus = AgentStatus.IDLE
        self.workspace_dir: Path = self._setup_workspace()
        self.task_handle: Optional[asyncio.Task] = None
        self.trajectory: Optional[AgentTrajectory] = None

        # Build subagent tools and ReAct loop
        self.tool_registry = build_default_tool_registry(self.workspace_dir)
        self._wire_inter_agent_tools()
        self.loop = AgentReActLoop(
            tool_registry=self.tool_registry,
            workspace_dir=self.workspace_dir,
            max_steps=12,
        )

    def _setup_workspace(self) -> Path:
        """Initialize workspace according to isolation mode."""
        if self.spec.workspace_mode == WorkspaceMode.INHERIT:
            return self.parent_workspace

        if self.spec.workspace_mode == WorkspaceMode.BRANCH:
            # Create isolated branch directory
            branch_dir = self.parent_workspace / ".antimatter" / "workspaces" / self.id
            branch_dir.mkdir(parents=True, exist_ok=True)
            # Copy parent files (excluding internal directories)
            for item in self.parent_workspace.iterdir():
                if item.name in (".antimatter", ".venv", ".git", "__pycache__"):
                    continue
                target = branch_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
            return branch_dir

        return self.parent_workspace

    def _wire_inter_agent_tools(self) -> None:
        """Add inter-agent messaging tools to subagent registry."""
        @self.tool_registry.register(
            name="send_message",
            description="Send a message to another agent or parent by ID."
        )
        async def send_message(recipient_id: str, message: str) -> str:
            ok, msg = await self.supervisor.send_message(
                recipient_id=recipient_id,
                message=message,
                sender_id=self.id,
            )
            return msg

    async def deliver_message(self, message: str, sender_id: str) -> None:
        """Deliver a message to this subagent's inbox."""
        await self.inbox.put({
            "sender_id": sender_id,
            "message": message,
            "timestamp": time.time(),
        })

    def start(self) -> None:
        """Start subagent background execution task."""
        self.status = AgentStatus.THINKING
        self.task_handle = asyncio.create_task(self._run_lifecycle())

    async def _run_lifecycle(self) -> None:
        """Main lifecycle of the subagent."""
        try:
            logger.info("Subagent %s (%s) starting task...", self.spec.name, self.id)
            self.trajectory = await self.loop.run(self.spec.prompt)
            self.status = self.trajectory.status
            # Notify supervisor of completion
            await self.supervisor.notify_completion(self.id, self.trajectory)
        except asyncio.CancelledError:
            self.status = AgentStatus.COMPLETED
            logger.info("Subagent %s (%s) was cancelled.", self.spec.name, self.id)
        except Exception as err:
            self.status = AgentStatus.ERROR
            logger.exception("Subagent %s failed: %s", self.id, err)

    def cleanup(self) -> None:
        """Clean up branched workspace if applicable."""
        if hasattr(self, "loop") and hasattr(self.loop, "close"):
            self.loop.close()

        if self.spec.workspace_mode == WorkspaceMode.BRANCH and self.workspace_dir.exists():
            import gc
            gc.collect()
            try:
                shutil.rmtree(self.workspace_dir, ignore_errors=False)
            except Exception as e:
                # Fallback to retry or ignore
                time.sleep(0.05)
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
