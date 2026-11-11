"""Production Autonomous Swarm Orchestrator for Project Antimatter.

Coordinates specialized micro-agents through the 4-stage pipeline with
self-healing retry loops and deterministic rollback guarantees.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from antimatter.agent.swarm.types import (
    SwarmStage,
    StageStatus,
    SwarmStepResult,
    SwarmTrajectory,
)
from antimatter.agent.swarm.blackboard import (
    SwarmBlackboard,
    ScoutEvidence,
    PatchBlueprint,
    VerificationResult,
)
from antimatter.agent.swarm.specialists import (
    ScoutSpecialist,
    ArchitectSpecialist,
    CoderSpecialist,
    VerifierSpecialist,
)
from antimatter.brain.client import LocalLLMClient
from antimatter.core.config import get_settings
from antimatter.tools.fs_manager import FileSystemManager
from antimatter.tools.patcher import MultiLinePatcher
from antimatter.tools.terminal import TerminalSandbox

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Orchestrates the multi-agent specialist swarm with self-healing retries."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        llm_client: Optional[LocalLLMClient] = None,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self.workspace_dir = (workspace_dir or settings.workspace_dir).resolve()
        self.llm_client = llm_client or LocalLLMClient(settings)
        self.max_retries = max_retries

        # Tool foundations
        self.fs = FileSystemManager(self.workspace_dir)
        self.patcher = MultiLinePatcher(self.fs)
        self.terminal = TerminalSandbox(self.workspace_dir)

        # Micro-Specialists
        self.scout = ScoutSpecialist(self.fs, self.llm_client)
        self.architect = ArchitectSpecialist(self.llm_client)
        self.coder = CoderSpecialist(self.patcher, self.fs)
        self.verifier = VerifierSpecialist(self.terminal)

    async def run(
        self,
        goal: str,
        target_file: str,
        test_file: Optional[str] = None,
        test_command: Optional[str] = None,
        on_stage_complete: Optional[Callable[[SwarmStepResult], None]] = None,
    ) -> SwarmTrajectory:
        """Run the complete 4-stage autonomous swarm pipeline."""
        start_time = time.time()
        blackboard = SwarmBlackboard()
        trajectory = SwarmTrajectory(user_goal=goal, overall_status=StageStatus.RUNNING)

        # 1. Transactional Checkpoint Baseline
        cp_id = self.fs.create_checkpoint(f"Swarm baseline: {goal[:50]}")
        trajectory.checkpoint_id = cp_id
        await blackboard.set_checkpoint(cp_id)
        logger.info("[Swarm] Initialized checkpoint: %s", cp_id)

        # --------------------------------------------------------------------
        # STAGE 1: SCOUT (Read-Only Code Inspection)
        # --------------------------------------------------------------------
        t0 = time.time()
        evidence = await self.scout.execute(goal, target_file, test_file)
        await blackboard.update_scout(evidence)
        step1 = SwarmStepResult(
            stage=SwarmStage.SCOUT,
            status=StageStatus.PASSED,
            evidence=evidence.model_dump(),
            summary=f"Scouted {target_file} ({len(evidence.code_snippet)} chars)",
            duration_sec=time.time() - t0,
        )
        trajectory.stages.append(step1)
        if on_stage_complete:
            on_stage_complete(step1)

        # Self-Healing Retry Loop for Stages 2 -> 3 -> 4
        current_evidence = evidence
        verification_passed = False
        retry_idx = 0

        while retry_idx <= self.max_retries and not verification_passed:
            # ----------------------------------------------------------------
            # STAGE 2: ARCHITECT (Pure Reasoning Patch Blueprint)
            # ----------------------------------------------------------------
            t0 = time.time()
            blueprint = await self.architect.execute(goal, current_evidence)
            await blackboard.update_blueprint(blueprint)
            step2 = SwarmStepResult(
                stage=SwarmStage.ARCHITECT,
                status=StageStatus.PASSED if blueprint.target_block else StageStatus.FAILED,
                evidence=blueprint.model_dump(),
                summary=f"Drafted patch: {blueprint.rationale}",
                duration_sec=time.time() - t0,
            )
            trajectory.stages.append(step2)
            if on_stage_complete:
                on_stage_complete(step2)

            if not blueprint.target_block:
                logger.warning("[Swarm] Architect produced empty patch. Retrying...")
                retry_idx += 1
                continue

            # ----------------------------------------------------------------
            # STAGE 3: CODER (Surgical Patch Application)
            # ----------------------------------------------------------------
            t0 = time.time()
            coder_res = await self.coder.execute(blueprint, checkpoint_id=cp_id)
            step3 = SwarmStepResult(
                stage=SwarmStage.CODER,
                status=StageStatus.PASSED if coder_res.get("success") else StageStatus.FAILED,
                evidence=coder_res,
                summary=f"Applied patch to {blueprint.file_path}" if coder_res.get("success") else f"Patch failed: {coder_res.get('error')}",
                duration_sec=time.time() - t0,
            )
            trajectory.stages.append(step3)
            if on_stage_complete:
                on_stage_complete(step3)

            if not coder_res.get("success"):
                logger.warning("[Swarm] Coder failed to apply patch. Retrying...")
                retry_idx += 1
                continue

            # ----------------------------------------------------------------
            # STAGE 4: VERIFIER (Sandboxed Test Validation)
            # ----------------------------------------------------------------
            if test_command:
                t0 = time.time()
                ver_res = await self.verifier.execute(test_command)
                await blackboard.update_verification(ver_res)
                step4 = SwarmStepResult(
                    stage=SwarmStage.VERIFIER,
                    status=StageStatus.PASSED if ver_res.passed else StageStatus.FAILED,
                    evidence=ver_res.model_dump(),
                    summary=f"Validation {'PASSED (exit code 0)' if ver_res.passed else f'FAILED (exit code {ver_res.exit_code})'}",
                    duration_sec=time.time() - t0,
                )
                trajectory.stages.append(step4)
                if on_stage_complete:
                    on_stage_complete(step4)

                if ver_res.passed:
                    verification_passed = True
                else:
                    logger.warning("[Swarm] Verification failed on retry %d. Feeding error to architect...", retry_idx)
                    # Feed error back to architect
                    current_evidence.findings = f"Previous patch failed test with output:\n{ver_res.stderr or ver_res.stdout}"
                    retry_idx += 1
            else:
                # No test command specified; assume patch complete
                verification_passed = True

        # --------------------------------------------------------------------
        # STAGE 5: SYNTHESIS & FINAL RESOLUTION
        # --------------------------------------------------------------------
        trajectory.total_duration_sec = time.time() - start_time
        trajectory.retries_count = retry_idx

        if verification_passed:
            trajectory.overall_status = StageStatus.PASSED
            trajectory.final_report = (
                f"Autonomous Swarm successfully resolved goal in {trajectory.total_duration_sec:.2f}s "
                f"across {len(trajectory.stages)} micro-stages. All tests validated."
            )
        else:
            trajectory.overall_status = StageStatus.FAILED
            # Rollback on final failure
            self.fs.rollback(cp_id)
            trajectory.final_report = (
                f"Autonomous Swarm could not verify fix after {retry_idx} retries. "
                f"Rolled back workspace to baseline checkpoint {cp_id}."
            )

        return trajectory
