"""Tests for Deep Reasoning & Code Critic Specialist."""

import pytest
from antimatter.agent.specialists.critic import (
    CodeCriticSpecialist,
    CriticVerdict,
    CodeCritiqueReport,
)
from antimatter.agent.relay import (
    SpecialistRelayEngine,
    SpecialistDomain,
    RelayBaton,
)
from antimatter.brain.arbiter import VRAMArbiter
from antimatter.agent.event_bus import AgentEventBus


def test_critic_syntax_error():
    critic = CodeCriticSpecialist()
    bad_code = "def invalid_syntax(x:\n    return x *"
    report = critic.audit_python_code(bad_code)

    assert report.verdict == CriticVerdict.REJECTED
    assert report.ast_valid is False
    assert len(report.issues) == 1
    assert report.issues[0].category == "syntax"


def test_critic_dangerous_eval_rejection():
    critic = CodeCriticSpecialist()
    dangerous_code = """
def run_user_input(code_str):
    eval(code_str)
"""
    report = critic.audit_python_code(dangerous_code)

    assert report.verdict == CriticVerdict.REJECTED
    assert report.security_score < 0.6
    assert any("eval()" in i.message for i in report.issues)


def test_critic_clean_code_approval():
    critic = CodeCriticSpecialist()
    clean_code = """
def calculate_trajectory(velocity: float, angle_rad: float) -> float:
    \"\"\"Compute ballistic range.\"\"\"
    g = 9.81
    import math
    return (velocity ** 2) * math.sin(2 * angle_rad) / g
"""
    report = critic.audit_python_code(clean_code)

    assert report.verdict == CriticVerdict.APPROVED
    assert report.security_score == 1.0
    assert "test_calculate_trajectory_happy_path" in report.suggested_tests


@pytest.mark.asyncio
async def test_critic_relay_consultation():
    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    event_bus = AgentEventBus()
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=event_bus)

    critic = CodeCriticSpecialist()
    relay.register_specialist(SpecialistDomain.CRITIC_TESTER, critic.handle_relay_baton)

    baton = RelayBaton(
        baton_id="baton-critic-audit-01",
        caller_id="coder_agent",
        domain=SpecialistDomain.CRITIC_TESTER,
        prompt="Audit proposed math utility function",
        context_data={"code": "def add(a, b): return a + b"},
    )

    result = await relay.consult(baton)

    assert result.status == "success"
    assert "Critic Verdict: APPROVED" in result.advice
    assert result.artifacts["verdict"] == "APPROVED"
    assert arbiter.allocated_vram_mb == 0
