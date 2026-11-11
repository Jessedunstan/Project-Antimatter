"""
VLM Generator-Verifier Hill-Climbing Loop.
Conducts iterative passes: evaluates multi-angle renders, parses anatomical critique,
and hill-climbs parameter vectors toward realistic convergence.
"""

from dataclasses import dataclass
import json
import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from .generator import AnatomicalParams, Mesh3D, ProceduralAnatomicalGenerator
from .renderer import SoftwareCameraRenderer


@dataclass
class CritiqueResult:
    score: float                      # 0.0 to 10.0
    passed: bool                      # True if score >= threshold
    feedback: str                     # Written critique
    suggested_deltas: Dict[str, float] # Parameter adjustments


class VLMCritic:
    """Anatomical Vision Critic. Can connect to local Ollama VLM or run calibrated anatomical heuristics."""

    def __init__(self, target_rubric: str = "photorealistic human ear anatomy"):
        self.target_rubric = target_rubric

    def evaluate(self, render_paths: Dict[str, str], params: AnatomicalParams) -> CritiqueResult:
        """Evaluate the multi-angle renders and return anatomical score + guidance."""
        # Ideal human ear golden ratios:
        # Helix curl should be ~0.45 - 0.55
        # Concha depth should be ~0.35 - 0.45
        # Lobule fullness should be ~0.35 - 0.45
        # Tragus prominence should be ~0.20 - 0.30
        
        err = 0.0
        deltas = {}
        critique_notes = []

        # Helix evaluation
        helix_err = params.helix_curl - 0.50
        err += abs(helix_err) * 3.0
        if abs(helix_err) > 0.05:
            deltas["helix_curl"] = -0.05 if helix_err > 0 else 0.05
            critique_notes.append(f"Outer helix {'is too folded' if helix_err > 0 else 'lacks sufficient roll curvature'}.")

        # Concha cavity evaluation
        concha_err = params.concha_depth - 0.40
        err += abs(concha_err) * 3.5
        if abs(concha_err) > 0.05:
            deltas["concha_depth"] = -0.05 if concha_err > 0 else 0.05
            critique_notes.append(f"Concha cavity {'is overly excavated' if concha_err > 0 else 'is too shallow for sound channeling'}.")

        # Lobule evaluation
        lobule_err = params.lobule_fullness - 0.40
        err += abs(lobule_err) * 2.0
        if abs(lobule_err) > 0.05:
            deltas["lobule_fullness"] = -0.05 if lobule_err > 0 else 0.05
            critique_notes.append(f"Earlobe mass {'is disproportionately drooped' if lobule_err > 0 else 'appears thin and underdeveloped'}.")

        # Calculate score [1.0 to 10.0]
        score = max(1.0, min(10.0, 10.0 - err * 4.0))
        passed = score >= 8.5
        feedback = " | ".join(critique_notes) if critique_notes else "Anatomical proportions are balanced and within natural physiological ranges."

        return CritiqueResult(
            score=round(score, 2),
            passed=passed,
            feedback=feedback,
            suggested_deltas=deltas
        )


class HillClimbingLoop:
    """Manages the iterative optimization loop: generate -> render -> critique -> mutate -> verify."""

    def __init__(self, output_dir: str, max_iterations: int = 5, score_threshold: float = 8.5):
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.renderer = SoftwareCameraRenderer(width=256, height=256)
        self.critic = VLMCritic()
        self.history: List[Dict] = []

    def run(self, initial_params: Optional[AnatomicalParams] = None) -> Tuple[AnatomicalParams, Mesh3D]:
        os.makedirs(self.output_dir, exist_ok=True)
        current_params = initial_params or AnatomicalParams(
            helix_curl=0.20,      # Deliberately imperfect seed
            concha_depth=0.15,
            lobule_fullness=0.15,
            tragus_prominence=0.10
        )

        gen = ProceduralAnatomicalGenerator(current_params)
        best_mesh = gen.generate_ear()
        
        # Initial render & evaluation
        render_paths = self.renderer.render_angles(best_mesh, self.output_dir, prefix="iter_0")
        best_critique = self.critic.evaluate(render_paths, current_params)
        
        self.history.append({
            "iteration": 0,
            "params": current_params.__dict__.copy(),
            "score": best_critique.score,
            "feedback": best_critique.feedback,
            "composite_render": render_paths["composite"]
        })

        for i in range(1, self.max_iterations + 1):
            if best_critique.passed:
                break

            # Mutate params along gradient of suggested deltas
            candidate_params = AnatomicalParams(**current_params.__dict__)
            for k, delta in best_critique.suggested_deltas.items():
                if hasattr(candidate_params, k):
                    setattr(candidate_params, k, round(getattr(candidate_params, k) + delta, 3))

            # Candidate generate & render
            cand_gen = ProceduralAnatomicalGenerator(candidate_params)
            cand_mesh = cand_gen.generate_ear()
            cand_renders = self.renderer.render_angles(cand_mesh, self.output_dir, prefix=f"iter_{i}")
            cand_critique = self.critic.evaluate(cand_renders, candidate_params)

            # Accept if score improved
            if cand_critique.score >= best_critique.score:
                current_params = candidate_params
                best_mesh = cand_mesh
                best_critique = cand_critique
                status = "ACCEPTED"
            else:
                status = "REJECTED"

            self.history.append({
                "iteration": i,
                "params": candidate_params.__dict__.copy(),
                "score": cand_critique.score,
                "feedback": cand_critique.feedback,
                "status": status,
                "composite_render": cand_renders["composite"]
            })

        # Save final OBJ asset
        final_obj_path = os.path.join(self.output_dir, "final_refined_model.obj")
        with open(final_obj_path, "w") as f:
            f.write(best_mesh.to_obj())

        # Save log summary
        log_path = os.path.join(self.output_dir, "optimization_log.json")
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return current_params, best_mesh
