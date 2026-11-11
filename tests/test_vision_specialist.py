"""Tests for Vision Frame Inspector Specialist."""

import pytest
import base64
from pathlib import Path
from antimatter.agent.specialists.vision import (
    VisionFrameInspector,
    VisualAnomaly,
    VisionInspectionReport,
)
from antimatter.agent.relay import (
    SpecialistRelayEngine,
    SpecialistDomain,
    RelayBaton,
)
from antimatter.brain.arbiter import VRAMArbiter
from antimatter.agent.event_bus import AgentEventBus


@pytest.mark.asyncio
async def test_vision_inspector_parsing():
    inspector = VisionFrameInspector()

    # Test parsing when glitch is detected
    vlm_sample_output = (
        "The scene shows a character model with mesh clipping through the dungeon wall. "
        "Also noticed missing texture on the sword hilt resulting in a magenta box."
    )
    report = inspector._parse_vlm_output("frame_01.png", vlm_sample_output)

    assert report.pass_qa is False
    assert len(report.anomalies) >= 2
    labels = [a.label for a in report.anomalies]
    assert "mesh_clipping" in labels
    assert "texture_missing" in labels


@pytest.mark.asyncio
async def test_vision_inspector_relay_integration(tmp_path: Path):
    # Create a dummy image file
    dummy_img = tmp_path / "test_frame.png"
    # 1x1 transparent PNG bytes
    dummy_img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    event_bus = AgentEventBus()
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=event_bus)

    inspector = VisionFrameInspector()
    relay.register_specialist(SpecialistDomain.VISION, inspector.handle_relay_baton)

    baton = RelayBaton(
        baton_id="baton-visual-qa-test",
        caller_id="game_engine_agent",
        domain=SpecialistDomain.VISION,
        prompt="Check terrain rendering and lighting stability.",
        context_data={"image_path": str(dummy_img)},
    )

    result = await relay.consult(baton)

    assert result.status == "success"
    assert "QA Status:" in result.advice
    assert "report" in result.artifacts
    assert result.artifacts["anomalies_count"] == 0
    # Confirm VRAM was yielded
    assert arbiter.allocated_vram_mb == 0
