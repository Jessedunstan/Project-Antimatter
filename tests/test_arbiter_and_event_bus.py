"""Tests for VRAM Arbiter and Multi-Agent Event Bus."""

import pytest
import asyncio
from antimatter.brain.arbiter import VRAMArbiter, ModelTier
from antimatter.agent.event_bus import AgentEventBus, AgentMessage


@pytest.mark.asyncio
async def test_vram_arbiter_allocations_and_evictions():
    # Setup arbiter with 8GB total, 1GB headroom -> 7GB usable
    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    assert arbiter.usable_vram_mb == 7168
    assert arbiter.free_vram_mb == 7168

    # Lease 1: Steering Agent (SmolLM2 / 2GB)
    lease1 = await arbiter.acquire_lease("lease-steer", "smollm2:1.7b", ModelTier.STEERING, 2048)
    assert arbiter.allocated_vram_mb == 2048
    assert arbiter.free_vram_mb == 5120

    # Lease 2: Vision Agent (Qwen2.5-VL / 4GB)
    lease2 = await arbiter.acquire_lease("lease-vision", "qwen2.5-vl:7b", ModelTier.VISION, 4096)
    assert arbiter.allocated_vram_mb == 6144
    assert arbiter.free_vram_mb == 1024

    # Lease 3: Heavy Media Request (SDXL / 4GB) -> Exceeds 1024MB free, must evict Vision (tier lower than steering)
    lease3 = await arbiter.acquire_lease("lease-media", "sdxl-lightning", ModelTier.MEDIA, 4096)
    assert "lease-vision" not in arbiter._active_leases
    assert "lease-steer" in arbiter._active_leases
    assert arbiter.allocated_vram_mb == 2048 + 4096


@pytest.mark.asyncio
async def test_agent_event_bus_pub_sub_and_blackboard():
    bus = AgentEventBus()
    received_messages = []

    async def on_steer_action(msg: AgentMessage):
        received_messages.append(msg)

    bus.subscribe("game.steer.action", on_steer_action)

    # Publish topic message
    msg = AgentMessage(
        sender="steer_agent",
        topic="game.steer.action",
        payload={"action": "MOVE_FORWARD", "speed": 1.5},
    )
    count = await bus.publish(msg)
    assert count == 1

    # Give event loop a tick to process background handler
    await asyncio.sleep(0.05)
    assert len(received_messages) == 1
    assert received_messages[0].payload["action"] == "MOVE_FORWARD"

    # Test Blackboard
    await bus.blackboard.set("game_score", 1420)
    score = await bus.blackboard.get("game_score")
    assert score == 1420
