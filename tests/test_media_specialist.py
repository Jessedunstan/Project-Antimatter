"""Tests for Headless ComfyUI Media Engine."""

import pytest
from pathlib import Path
from antimatter.agent.specialists.media import (
    ComfyUIMediaEngine,
    MediaGenerationRequest,
    MediaArtifact,
)
from antimatter.agent.relay import (
    SpecialistRelayEngine,
    SpecialistDomain,
    RelayBaton,
)
from antimatter.brain.arbiter import VRAMArbiter
from antimatter.agent.event_bus import AgentEventBus


@pytest.mark.asyncio
async def test_media_graph_construction(tmp_path: Path):
    engine = ComfyUIMediaEngine(output_dir=tmp_path)
    req = MediaGenerationRequest(
        prompt="retro cyberpunk mech sprite, pixel art",
        width=512,
        height=512,
        steps=4,
    )
    graph = engine.build_sdxl_lightning_graph(req)

    assert "3" in graph
    assert graph["3"]["class_type"] == "KSampler"
    assert graph["3"]["inputs"]["steps"] == 4
    assert graph["6"]["inputs"]["text"] == "retro cyberpunk mech sprite, pixel art"


@pytest.mark.asyncio
async def test_media_relay_consultation(tmp_path: Path):
    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    event_bus = AgentEventBus()
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=event_bus)

    media_engine = ComfyUIMediaEngine(output_dir=tmp_path)
    relay.register_specialist(SpecialistDomain.MEDIA, media_engine.handle_relay_baton)

    baton = RelayBaton(
        baton_id="baton-media-gen-01",
        caller_id="game_director",
        domain=SpecialistDomain.MEDIA,
        prompt="Generate glowing neon portal texture for level exit",
        context_data={"media_type": "image", "width": 512, "height": 512, "steps": 4},
    )

    result = await relay.consult(baton)

    assert result.status == "success"
    assert "Generated IMAGE asset" in result.advice
    assert "file_path" in result.artifacts
    assert Path(result.artifacts["file_path"]).exists()
    assert arbiter.allocated_vram_mb == 0
