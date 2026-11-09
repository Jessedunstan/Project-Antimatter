"""Multi-Agent Event Bus & Shared Blackboard for Project Antimatter.

Enables asynchronous collaboration between specialized local agents (Steering,
Vision QA, Media Creation, and Reasoning Planners) without token overhead.
"""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentMessage(BaseModel):
    """Standard envelope for inter-agent communication."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    recipient: str = "*"  # '*' denotes broadcast to all subscribers of the topic
    topic: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class SharedBlackboard:
    """In-memory key-value blackboard for collaborative agent workspace state."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._state[key] = value

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._state.get(key, default)

    async def dump(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._state)


# Type alias for async message handlers
MessageHandler = Callable[[AgentMessage], Coroutine[Any, Any, None]]


class AgentEventBus:
    """High-throughput async pub/sub bus connecting specialized agents."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Set[MessageHandler]] = {}
        self._direct_inboxes: Dict[str, asyncio.Queue[AgentMessage]] = {}
        self.blackboard = SharedBlackboard()

    def register_agent(self, agent_id: str) -> asyncio.Queue[AgentMessage]:
        """Register an agent and return its dedicated direct message inbox."""
        if agent_id not in self._direct_inboxes:
            self._direct_inboxes[agent_id] = asyncio.Queue()
        return self._direct_inboxes[agent_id]

    def unregister_agent(self, agent_id: str) -> None:
        """Remove agent inbox."""
        self._direct_inboxes.pop(agent_id, None)

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe a handler callback to a specific topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = set()
        self._subscribers[topic].add(handler)

    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """Remove a subscriber from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic].discard(handler)

    async def publish(self, message: AgentMessage) -> int:
        """Publish a message to topic subscribers and route direct messages."""
        delivery_count = 0

        # Direct point-to-point delivery
        if message.recipient != "*":
            inbox = self._direct_inboxes.get(message.recipient)
            if inbox:
                await inbox.put(message)
                delivery_count += 1
            else:
                logger.warning("Agent recipient '%s' not found for direct message.", message.recipient)
            return delivery_count

        # Topic broadcast delivery
        handlers = self._subscribers.get(message.topic, set()).copy()
        for handler in handlers:
            try:
                # Dispatch handler asynchronously
                asyncio.create_task(handler(message))
                delivery_count += 1
            except Exception as e:
                logger.error("Error executing handler for topic '%s': %s", message.topic, e)

        return delivery_count
