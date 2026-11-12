"""Tests for Autonomous Multi-Agent Game QA & Fuzzing Mesh."""

import pytest
from antimatter.agent.specialists.fuzzer import GameQAFuzzer, FuzzingSessionReport
from antimatter.agent.specialists.steering import SteeringSpecialist
from antimatter.agent.specialists.vision import VisionFrameInspector
from antimatter.agent.specialists.critic import CodeCriticSpecialist
from antimatter.agent.event_bus import AgentEventBus, AgentMessage


@pytest.mark.asyncio
async def test_game_qa_fuzzer_end_to_end():
    event_bus = AgentEventBus()
    glitch_events = []

    async def on_glitch(msg: AgentMessage):
        glitch_events.append(msg.payload)

    event_bus.subscribe("qa.glitch.detected", on_glitch)

    fuzzer = GameQAFuzzer(event_bus=event_bus)
    report = await fuzzer.run_fuzzing_session(session_id="test-fuzz-01", num_ticks=10)

    assert isinstance(report, FuzzingSessionReport)
    assert report.total_ticks == 10
    assert report.collisions_detected == 1
    assert len(report.glitches_identified) >= 1
    assert len(report.synthesized_regression_tests) >= 3
    assert "test_boundary_penetration_at_high_velocity" in report.synthesized_regression_tests
    assert report.critic_verdict in ["APPROVED", "WARNING"]
    # Allow event loop a tick to process background subscriber tasks
    import asyncio
    await asyncio.sleep(0.05)
    assert len(glitch_events) == 1
    assert glitch_events[0]["tick"] == 5
