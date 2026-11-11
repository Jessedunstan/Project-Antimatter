"""Orchestrated Specialist Relay Engine for Project Antimatter.

Facilitates time-multiplexed 'baton passing' between specialized offline AI models
(Steering, Vision VLM, Media Diffusion, Deep Reasoning) on consumer VRAM (RTX 3060 Ti).
"""

from __future__ import annotations
import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from antimatter.brain.arbiter import VRAMArbiter, ModelTier
from antimatter.agent.event_bus import AgentEventBus, AgentMessage
from antimatter.brain.client import LocalLLMClient
from antimatter.brain.types import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


class SpecialistDomain(str, Enum):
    """Specialized intelligence roles."""
    STEERING = "steering"              # Real-time game actor / robot kinematics
    VISION = "vision"                  # Frame inspection, visual QA, spatial UI bugs
    MEDIA = "media"                    # Asset texture, sprite, audio, and video generation
    DEEP_REASONING = "deep_reasoning"  # Algorithmic synthesis, math & complex refactoring
    CRITIC_TESTER = "critic_tester"    # Policy verification, test coverage & safety checks


class RelayBaton(BaseModel):
    """Payload handed over from the generalist agent to a specialist in a pinch."""
    baton_id: str
    caller_id: str
    domain: SpecialistDomain
    prompt: str
    context_data: Dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = 30.0
    created_at: float = Field(default_factory=time.time)


class RelayConsultationResult(BaseModel):
    """Telemetry and advisory response returned by a specialist."""
    baton_id: str
    domain: SpecialistDomain
    status: str
    advice: str
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    duration_sec: float
    error_message: Optional[str] = None


class SpecialistRelayEngine:
    """Manages coordinated VRAM handoffs and advisory relays between local models."""

    def __init__(
        self,
        arbiter: VRAMArbiter,
        event_bus: AgentEventBus,
        llm_client: Optional[LocalLLMClient] = None,
    ):
        self.arbiter = arbiter
        self.event_bus = event_bus
        self.llm_client = llm_client or LocalLLMClient()
        self._handlers: Dict[SpecialistDomain, Callable[[RelayBaton], Any]] = {}

    def register_specialist(
        self,
        domain: SpecialistDomain,
        handler: Callable[[RelayBaton], Any],
    ) -> None:
        """Register a custom execution handler for a specific specialist domain."""
        self._handlers[domain] = handler

    async def consult(self, baton: RelayBaton) -> RelayConsultationResult:
        """Execute a time-multiplexed relay consultation with a specialist."""
        start_time = time.time()
        logger.info(
            "Relay baton '%s' initiated by '%s' for domain '%s'",
            baton.baton_id, baton.caller_id, baton.domain
        )

        # 1. Map domain to VRAM Arbiter Tier and estimated footprint
        tier_map = {
            SpecialistDomain.STEERING: (ModelTier.STEERING, 2048, "smollm2:1.7b"),
            SpecialistDomain.VISION: (ModelTier.VISION, 4608, "qwen2.5-vl:7b"),
            SpecialistDomain.MEDIA: (ModelTier.MEDIA, 5120, "sdxl-lightning"),
            SpecialistDomain.DEEP_REASONING: (ModelTier.REASONING, 4608, "deepseek-r1:7b"),
            SpecialistDomain.CRITIC_TESTER: (ModelTier.REASONING, 2048, "qwen2.5:3b"),
        }
        tier, vram_mb, model_name = tier_map.get(
            baton.domain, (ModelTier.REASONING, 2048, "qwen2.5:3b")
        )

        lease_id = f"relay-{baton.domain.value}-{baton.baton_id}"

        # 2. Acquire VRAM lease (preempts or unloads conflicting models)
        await self.arbiter.acquire_lease(
            lease_id=lease_id,
            model_name=model_name,
            tier=tier,
            required_vram_mb=vram_mb,
        )

        try:
            # 3. Check for domain-specific handler or fallback to LLM consultation
            if baton.domain in self._handlers:
                handler = self._handlers[baton.domain]
                if asyncio.iscoroutinefunction(handler):
                    output = await handler(baton)
                else:
                    output = handler(baton)
                advice_text = output if isinstance(output, str) else str(output.get("advice", ""))
                artifacts = output.get("artifacts", {}) if isinstance(output, dict) else {}
            else:
                # Default LLM consultation run through the allocated model
                messages = [
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        content=f"You are the Antimatter {baton.domain.value.upper()} Specialist. Provide concise, highly technical actionable advice.",
                    ),
                    ChatMessage(
                        role=ChatRole.USER,
                        content=f"{baton.prompt}\n\nContext:\n{baton.context_data}",
                    ),
                ]
                resp = await self.llm_client.chat(messages, model=model_name)
                advice_text = resp.content or ""
                artifacts = {}

            elapsed = time.time() - start_time
            result = RelayConsultationResult(
                baton_id=baton.baton_id,
                domain=baton.domain,
                status="success",
                advice=advice_text,
                artifacts=artifacts,
                duration_sec=elapsed,
            )

            # 4. Publish advice to the Event Bus and Blackboard
            await self.event_bus.publish(
                AgentMessage(
                    sender=f"specialist_{baton.domain.value}",
                    recipient=baton.caller_id,
                    topic=f"specialist.{baton.domain.value}.advisory",
                    payload=result.model_dump(),
                    correlation_id=baton.baton_id,
                )
            )
            await self.event_bus.blackboard.set(f"advisory:{baton.baton_id}", result.model_dump())
            return result

        except Exception as e:
            logger.error("Error during specialist relay: %s", e)
            return RelayConsultationResult(
                baton_id=baton.baton_id,
                domain=baton.domain,
                status="error",
                advice="",
                duration_sec=time.time() - start_time,
                error_message=str(e),
            )
        finally:
            # 5. Release lease and yield VRAM back to the system
            await self.arbiter.release_lease(lease_id, unload_from_gpu=True)
