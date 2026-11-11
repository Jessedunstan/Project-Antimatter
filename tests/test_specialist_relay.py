"""Tests for the Specialist Relay Engine and Baton Protocol."""

import pytest
import asyncio
from antimatter.agent.relay import (
    SpecialistRelayEngine,
    SpecialistDomain,
    RelayBaton,
)
from antimatter.brain.arbiter import VRAMArbiter
from antimatter.agent.event_bus import AgentEventBus


@pytest.mark.asyncio
async def test_specialist_relay_baton_pass():
    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    event_bus = AgentEventBus()
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=event_bus)

    # Register custom Vision Specialist handler
    async def mock_vision_handler(baton: RelayBaton):
        return {
            "advice": "Detected collision mesh clipping at coordinates (14.2, -3.1, 0.0). Recommended restitution damping.",
            "artifacts": {"bounding_box": [120, 240, 310, 480], "confidence": 0.94},
        }

    relay.register_specialist(SpecialistDomain.VISION, mock_vision_handler)

    # Generalist agent prepares a relay baton
    baton = RelayBaton(
        baton_id="baton-frame-qa-101",
        caller_id="generalist_driver",
        domain=SpecialistDomain.VISION,
        prompt="Inspect frame 42 for physics glitch or terrain clipping.",
        context_data={"frame_id": 42, "game_tick": 1820},
    )

    # Register recipient inbox on event bus
    inbox = event_bus.register_agent("generalist_driver")

    # Execute relay consultation
    result = await relay.consult(baton)

    assert result.status == "success"
    assert "collision mesh clipping" in result.advice
    assert result.artifacts["confidence"] == 0.94

    # Verify message delivered to caller inbox
    assert not inbox.empty()
    delivered_msg = await inbox.get()
    assert delivered_msg.sender == "specialist_vision"
    assert delivered_msg.payload["status"] == "success"

    # Verify blackboard entry
    stored_advisory = await event_bus.blackboard.get("advisory:baton-frame-qa-101")
    assert stored_advisory is not None
    assert stored_advisory["baton_id"] == "baton-frame-qa-101"

    # Verify VRAM lease was cleanly released
    assert arbiter.allocated_vram_mb == 0
    assert arbiter.free_vram_mb == 7168
