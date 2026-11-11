"""
Tests for Scenario LAB-3D-01 (Procedural Blender + VLM Critic Loop).
"""

import os
import shutil
import pytest

from staging.3d_blender_vlm.generator import AnatomicalParams, ProceduralAnatomicalGenerator
from staging.3d_blender_vlm.renderer import SoftwareCameraRenderer
from staging.3d_blender_vlm.critic_loop import HillClimbingLoop, VLMCritic


@pytest.fixture
def temp_output_dir(tmp_path):
    out = tmp_path / "test_renders"
    out.mkdir()
    return str(out)


def test_generator_produces_valid_mesh():
    params = AnatomicalParams(subdivisions=12)
    gen = ProceduralAnatomicalGenerator(params)
    mesh = gen.generate_ear()

    assert len(mesh.vertices) == 12 * 12
    assert len(mesh.faces) > 0
    assert len(mesh.normals) == len(mesh.vertices)

    obj_str = mesh.to_obj()
    assert "v " in obj_str
    assert "f " in obj_str
    assert "vn " in obj_str


def test_renderer_generates_4_angles(temp_output_dir):
    params = AnatomicalParams(subdivisions=10)
    gen = ProceduralAnatomicalGenerator(params)
    mesh = gen.generate_ear()

    renderer = SoftwareCameraRenderer(width=64, height=64)
    paths = renderer.render_angles(mesh, temp_output_dir, prefix="test")

    assert "front" in paths
    assert "side" in paths
    assert "perspective_34" in paths
    assert "top" in paths
    assert "composite" in paths

    for p in paths.values():
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0


def test_vlm_critic_scores_and_deltas():
    critic = VLMCritic()
    params = AnatomicalParams(helix_curl=0.20, concha_depth=0.10)
    result = critic.evaluate({}, params)

    assert result.score < 8.0
    assert not result.passed
    assert "helix_curl" in result.suggested_deltas
    assert result.suggested_deltas["helix_curl"] > 0 # should request curl increase


def test_hill_climbing_loop_converges(temp_output_dir):
    loop = HillClimbingLoop(output_dir=temp_output_dir, max_iterations=4, score_threshold=8.0)
    final_params, final_mesh = loop.run()

    assert len(loop.history) >= 2
    # Verify score improved over initial iteration
    initial_score = loop.history[0]["score"]
    final_score = loop.history[-1]["score"]
    assert final_score >= initial_score

    # Verify OBJ export exists
    assert os.path.exists(os.path.join(temp_output_dir, "final_refined_model.obj"))
    assert os.path.exists(os.path.join(temp_output_dir, "optimization_log.json"))
