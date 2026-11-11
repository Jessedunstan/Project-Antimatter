"""End-to-end multi-specialist relay scenario.

Director agent runs a relay chain on a single simulated 8GB GPU:
  1. Steering specialist reacts to an obstacle (game tick).
  2. Media specialist generates a texture for the level.
  3. Vision specialist inspects the generated asset.
Verifies strict sequential VRAM handoff, event-bus delivery, and blackboard
aggregation. Runs fully offline (no Ollama / ComfyUI required).
"""

import pytest
from pathlib import Path

from antimatter.agent.event_bus import AgentEventBus, AgentMessage
from antimatter.agent.relay import RelayBaton, SpecialistDomain, SpecialistRelayEngine
from antimatter.agent.specialists import (
    ComfyUIMediaEngine,
    SteeringSpecialist,
    VisionFrameInspector,
)
from antimatter.brain.arbiter import VRAMArbiter


class PeakTrackingArbiter(VRAMArbiter):
    """Arbiter that records peak allocation and lease order for assertions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_mb = 0
        self.lease_log = []

    async def acquire_lease(self, lease_id, model_name, tier, required_vram_mb):
        lease = await super().acquire_lease(lease_id, model_name, tier, required_vram_mb)
        self.peak_mb = max(self.peak_mb, self.allocated_vram_mb)
        self.lease_log.append((tier.value, model_name, len(self._active_leases)))
        return lease


@pytest.mark.asyncio
async def test_full_relay_chain_steer_media_vision(tmp_path: Path):
    arbiter = PeakTrackingArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    bus = AgentEventBus()
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=bus)

    # Vision points at a dead port so the test is deterministic offline.
    steering = SteeringSpecialist(event_bus=bus)
    media = ComfyUIMediaEngine(comfyui_base_url="http://127.0.0.1:1", output_dir=tmp_path)
    vision = VisionFrameInspector(ollama_base_url="http://127.0.0.1:1")

    relay.register_specialist(SpecialistDomain.STEERING, steering.handle_relay_baton)
    relay.register_specialist(SpecialistDomain.MEDIA, media.handle_relay_baton)
    relay.register_specialist(SpecialistDomain.VISION, vision.handle_relay_baton)

    director_inbox = bus.register_agent("director")
    broadcast_actions = []

    async def on_steer(msg: AgentMessage):
        broadcast_actions.append(msg.payload)

    bus.subscribe("game.steer.action", on_steer)

    # --- Leg 1: Steering ---
    steer_res = await relay.consult(RelayBaton(
        baton_id="e2e-1-steer",
        caller_id="director",
        domain=SpecialistDomain.STEERING,
        prompt="Avoid wall ahead",
        context_data={"observation": {
            "tick": 7,
            "raycast_distances": [10.0, 9.0, 1.0, 2.0, 10.0],
            "goal_direction": [0.0, 1.0],
        }},
    ))
    assert steer_res.status == "success"
    assert steer_res.artifacts["action"]["action"] == "TURN_LEFT"

    # --- Leg 2: Media (heaviest tier) ---
    media_res = await relay.consult(RelayBaton(
        baton_id="e2e-2-media",
        caller_id="director",
        domain=SpecialistDomain.MEDIA,
        prompt="Mossy stone wall texture, seamless, 512px",
        context_data={"media_type": "image", "width": 512, "height": 512},
    ))
    assert media_res.status == "success"
    asset_path = media_res.artifacts["file_path"]
    assert Path(asset_path).exists()
    assert media_res.artifacts["is_placeholder"] is True  # ComfyUI offline -> honest flag

    # --- Leg 3: Vision inspects the asset from leg 2 (baton chaining) ---
    vision_res = await relay.consult(RelayBaton(
        baton_id="e2e-3-vision",
        caller_id="director",
        domain=SpecialistDomain.VISION,
        prompt="Check texture for seams or artifacts",
        context_data={"image_path": asset_path},
    ))
    assert vision_res.status == "success"
    # VLM offline -> must NOT claim a pass
    assert vision_res.artifacts["inspected"] is False
    assert vision_res.artifacts["pass_qa"] is False
    assert "UNINSPECTED" in vision_res.advice

    # --- VRAM: strictly sequential relay, one model resident at a time ---
    assert [entry[0] for entry in arbiter.lease_log] == ["steering", "media", "vision"]
    assert all(active == 1 for _, _, active in arbiter.lease_log)
    assert arbiter.peak_mb <= arbiter.usable_vram_mb
    assert arbiter.allocated_vram_mb == 0

    # --- Event bus: director received all three advisories, in order ---
    received = [director_inbox.get_nowait() for _ in range(director_inbox.qsize())]
    assert [m.correlation_id for m in received] == ["e2e-1-steer", "e2e-2-media", "e2e-3-vision"]
    assert len(broadcast_actions) == 1

    # --- Blackboard: aggregated session state ---
    for bid in ("e2e-1-steer", "e2e-2-media", "e2e-3-vision"):
        assert (await bus.blackboard.get(f"advisory:{bid}"))["status"] == "success"


@pytest.mark.asyncio
async def test_relay_isolates_specialist_failure():
    """A crashing specialist returns an error result and still frees VRAM."""
    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=AgentEventBus())

    async def broken(_baton):
        raise RuntimeError("GPU kernel fault")

    relay.register_specialist(SpecialistDomain.DEEP_REASONING, broken)
    res = await relay.consult(RelayBaton(
        baton_id="e2e-fail", caller_id="director",
        domain=SpecialistDomain.DEEP_REASONING, prompt="x",
    ))
    assert res.status == "error"
    assert "GPU kernel fault" in res.error_message
    assert arbiter.allocated_vram_mb == 0
