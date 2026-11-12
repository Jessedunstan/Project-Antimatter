"""Comprehensive Verification Suite for the Production Autonomous Swarm Engine."""

import pytest
import sys
from pathlib import Path

from antimatter.agent.swarm import (
    SwarmStage,
    StageStatus,
    SwarmBlackboard,
    ScoutEvidence,
    PatchBlueprint,
    VerificationResult,
    ScoutSpecialist,
    ArchitectSpecialist,
    CoderSpecialist,
    VerifierSpecialist,
    SwarmOrchestrator,
)
from antimatter.brain.client import LocalLLMClient
from antimatter.core.config import get_settings, LLMBackendType
from antimatter.tools.fs_manager import FileSystemManager
from antimatter.tools.patcher import MultiLinePatcher
from antimatter.tools.terminal import TerminalSandbox


# ============================================================================
# 1. BLACKBOARD SYNCHRONIZATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_swarm_blackboard():
    bb = SwarmBlackboard()
    await bb.update_scout(ScoutEvidence(file_path="main.py", findings="Discovered bug"))
    await bb.update_blueprint(PatchBlueprint(file_path="main.py", target_block="a", replacement_block="b"))
    await bb.update_verification(VerificationResult(passed=True, exit_code=0))
    await bb.set_checkpoint("chk_12345")

    state = await bb.get_state()
    assert state["scout"]["file_path"] == "main.py"
    assert state["blueprint"]["replacement_block"] == "b"
    assert state["verification"]["passed"] is True
    assert state["checkpoint_id"] == "chk_12345"


# ============================================================================
# 2. MICRO-SPECIALIST UNIT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_scout_specialist(tmp_path):
    fs = FileSystemManager(tmp_path)
    fs.write_atomic("calc.py", "def add(x, y): return x - y # BUG")
    fs.write_atomic("test_calc.py", "def test_add(): assert add(2, 2) == 4")

    llm = LocalLLMClient()
    scout = ScoutSpecialist(fs, llm)

    evidence = await scout.execute(
        goal="Fix addition logic",
        target_file="calc.py",
        test_file="test_calc.py",
    )

    assert evidence.file_path == "calc.py"
    assert "def add(x, y)" in evidence.code_snippet
    assert "test_add" in evidence.test_code_snippet


@pytest.mark.asyncio
async def test_coder_specialist(tmp_path):
    fs = FileSystemManager(tmp_path)
    patcher = MultiLinePatcher(fs)
    coder = CoderSpecialist(patcher, fs)

    fs.write_atomic("calc.py", "def add(x, y):\n    return x - y\n")

    blueprint = PatchBlueprint(
        file_path="calc.py",
        target_block="    return x - y",
        replacement_block="    return x + y",
        rationale="Fix subtraction to addition",
    )

    res = await coder.execute(blueprint)
    assert res["success"] is True

    updated_code = fs.read_file("calc.py")
    assert "return x + y" in updated_code


@pytest.mark.asyncio
async def test_verifier_specialist(tmp_path):
    terminal = TerminalSandbox(tmp_path)
    verifier = VerifierSpecialist(terminal)

    res = await verifier.execute("echo 'Verifier Test Passed'")
    assert res.passed is True
    assert res.exit_code == 0
    assert "Verifier Test Passed" in res.stdout


# ============================================================================
# 3. END-TO-END AUTONOMOUS SWARM PIPELINE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_swarm_orchestrator_mock(tmp_path):
    # Setup test workspace
    fs = FileSystemManager(tmp_path)
    fs.write_atomic("math_mod.py", "def multiply(a, b):\n    return a + b  # BUG\n")
    fs.write_atomic("test_math.py", "from math_mod import multiply\ndef test_mul():\n    assert multiply(3, 4) == 12\n")

    settings = get_settings()
    settings.llm_backend = LLMBackendType.MOCK
    llm = LocalLLMClient(settings)

    swarm = SwarmOrchestrator(workspace_dir=tmp_path, llm_client=llm)

    # Mock the architect to return exact blueprint for deterministic test
    async def mock_architect_exec(goal, evidence):
        return PatchBlueprint(
            file_path="math_mod.py",
            target_block="    return a + b  # BUG",
            replacement_block="    return a * b",
            rationale="Change addition to multiplication",
        )
    swarm.architect.execute = mock_architect_exec

    stages_seen = []
    def on_stage(step):
        stages_seen.append(step.stage)

    test_cmd = f'"{sys.executable}" -m pytest test_math.py'
    trajectory = await swarm.run(
        goal="Fix multiplication bug",
        target_file="math_mod.py",
        test_file="test_math.py",
        test_command=test_cmd,
        on_stage_complete=on_stage,
    )

    assert trajectory.overall_status == StageStatus.PASSED
    assert SwarmStage.SCOUT in stages_seen
    assert SwarmStage.ARCHITECT in stages_seen
    assert SwarmStage.CODER in stages_seen
    assert SwarmStage.VERIFIER in stages_seen

    # Verify code was modified
    assert "return a * b" in fs.read_file("math_mod.py")


@pytest.mark.asyncio
async def test_swarm_rollback_on_irrecoverable_failure(tmp_path):
    fs = FileSystemManager(tmp_path)
    original_code = "def stable(): return 100\n"
    fs.write_atomic("service.py", original_code)

    swarm = SwarmOrchestrator(workspace_dir=tmp_path, max_retries=1)

    # Mock architect to return an invalid patch that fails test
    async def failing_architect(goal, evidence):
        return PatchBlueprint(
            file_path="service.py",
            target_block="def stable(): return 100",
            replacement_block="def stable(): return 0",
        )
    swarm.architect.execute = failing_architect

    # Command that will always fail
    failing_cmd = f'"{sys.executable}" -c "import service; assert service.stable() == 100"'

    trajectory = await swarm.run(
        goal="Attempt change but fail test",
        target_file="service.py",
        test_command=failing_cmd,
    )

    assert trajectory.overall_status == StageStatus.FAILED
    # Verify rollback restored the original file!
    assert fs.read_file("service.py") == original_code
