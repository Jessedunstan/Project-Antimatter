"""Subagent Supervisor and Inter-Agent Mailbox Manager for Project Antimatter.

Coordinates spawning, communication, and lifecycle of subagents following
the Google Antigravity / Devin multi-agent delegation pattern.
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from antimatter.agent.subagent import SubagentSpec, SubagentWorker, WorkspaceMode
from antimatter.agent.types import AgentTrajectory, AgentStatus
from antimatter.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SubagentInfo(BaseModel):
    """Telemetry information for an active or completed subagent."""
    id: str
    name: str
    role: str
    status: str
    workspace_mode: str
    has_error: bool = False


class SubagentSupervisor:
    """Supervises subagents, routes inter-agent messages, and manages workspaces."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir.resolve()
        self._subagents: Dict[str, SubagentWorker] = {}
        self.parent_inbox: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._completion_events: Dict[str, asyncio.Event] = {}

    def spawn(self, spec: SubagentSpec) -> SubagentWorker:
        """Spawn a new subagent and launch its background lifecycle."""
        worker = SubagentWorker(
            spec=spec,
            supervisor=self,
            parent_workspace=self.workspace_dir,
        )
        self._subagents[worker.id] = worker
        self._completion_events[worker.id] = asyncio.Event()
        worker.start()
        return worker

    async def notify_completion(self, subagent_id: str, trajectory: AgentTrajectory) -> None:
        """Called by subagents when their trajectory completes."""
        event = self._completion_events.get(subagent_id)
        if event:
            event.set()

        # Place completion message into parent inbox
        await self.parent_inbox.put({
            "sender_id": subagent_id,
            "event": "completed",
            "status": trajectory.status.value,
            "final_response": trajectory.final_response,
            "error_message": trajectory.error_message,
        })

    async def send_message(
        self,
        recipient_id: str,
        message: str,
        sender_id: str = "parent",
    ) -> Tuple[bool, str]:
        """Route a message to a subagent or the parent supervisor."""
        if recipient_id == "parent":
            await self.parent_inbox.put({
                "sender_id": sender_id,
                "event": "message",
                "message": message,
            })
            return True, "Message delivered to parent supervisor."

        target = self._subagents.get(recipient_id)
        if not target:
            return False, f"Subagent '{recipient_id}' not found."

        await target.deliver_message(message=message, sender_id=sender_id)
        return True, f"Message delivered to subagent '{target.spec.name}' ({recipient_id})."

    def list_subagents(self) -> List[SubagentInfo]:
        """List status and telemetry of all subagents."""
        return [
            SubagentInfo(
                id=w.id,
                name=w.spec.name,
                role=w.spec.role,
                status=w.status.value,
                workspace_mode=w.spec.workspace_mode.value,
                has_error=(w.status == AgentStatus.ERROR),
            )
            for w in self._subagents.values()
        ]

    def kill_subagent(self, subagent_id: str) -> Tuple[bool, str]:
        """Cancel a running subagent and clean up its resources."""
        worker = self._subagents.get(subagent_id)
        if not worker:
            return False, f"Subagent '{subagent_id}' not found."

        if worker.task_handle and not worker.task_handle.done():
            worker.task_handle.cancel()

        worker.cleanup()
        del self._subagents[subagent_id]
        if subagent_id in self._completion_events:
            del self._completion_events[subagent_id]

        return True, f"Subagent '{worker.spec.name}' ({subagent_id}) killed and cleaned up."

    def kill_all(self) -> int:
        """Kill all subagents and clear registry."""
        count = len(self._subagents)
        ids = list(self._subagents.keys())
        for sid in ids:
            self.kill_subagent(sid)
        return count

    def register_supervisor_tools(self, registry: ToolRegistry) -> None:
        """Expose invoke_subagent, send_message, and manage_subagents to an agent registry."""

        @registry.register(
            name="invoke_subagent",
            description="Launch an autonomous subagent with a specific role and prompt.",
        )
        def invoke_subagent(
            name: str,
            role: str,
            prompt: str,
            workspace_mode: str = "inherit",
        ) -> str:
            mode = WorkspaceMode.BRANCH if workspace_mode == "branch" else WorkspaceMode.INHERIT
            spec = SubagentSpec(
                name=name,
                role=role,
                prompt=prompt,
                workspace_mode=mode,
            )
            worker = self.spawn(spec)
            return f"Subagent '{name}' spawned successfully. Conversation ID: {worker.id}"

        @registry.register(
            name="send_message",
            description="Send a message to another agent or subagent.",
        )
        async def send_message(recipient: str, message: str) -> str:
            ok, msg = await self.send_message(recipient_id=recipient, message=message, sender_id="parent")
            return msg

        @registry.register(
            name="manage_subagents",
            description="Manage running subagents: list, kill, or kill_all.",
        )
        def manage_subagents(action: str, conversation_ids: Optional[List[str]] = None) -> str:
            if action == "list":
                subs = self.list_subagents()
                if not subs:
                    return "No active subagents."
                return "\n".join(
                    f"- [{s.id}] {s.name} ({s.role}) - Status: {s.status} (Mode: {s.workspace_mode})"
                    for s in subs
                )
            if action == "kill":
                if not conversation_ids:
                    return "Error: conversation_ids required for kill action."
                results = [self.kill_subagent(cid)[1] for cid in conversation_ids]
                return "\n".join(results)
            if action == "kill_all":
                killed = self.kill_all()
                return f"Successfully terminated {killed} subagents."

            return f"Unknown action '{action}'."
