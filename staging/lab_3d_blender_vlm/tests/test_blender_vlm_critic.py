"""Tests for Procedural 3D Blender + VLM Critic Loop (LAB-3D-01)."""

import pytest
from pathlib import Path
from staging.lab_3d_blender_vlm.procedural_generator import ProceduralMeshBuilder
from staging.lab_3d_blender_vlm.multi_view_rasterizer import MultiViewRasterizer
from staging.lab_3d_blender_vlm.vlm_critic_loop import VLMCriticHillClimber


def test_procedural_mesh_building():
    mesh = ProceduralMeshBuilder.build_capsule_limb(length=2.0, radius=0.5, taper=0.8, segments=8, rings=4)
    assert len(mesh.vertices) == 8 * 4
    assert len(mesh.faces) == (4 - 1) * 8

    obj_str = mesh.to_obj()
    assert obj_str.startswith("# Antimatter 3D Procedural Mesh")
    assert "v " in obj_str
    assert "f " in obj_str

    bpy_script = mesh.to_bpy_script(Path("./renders"))
    assert "import bpy" in bpy_script
    assert "bpy.ops.render.render" in bpy_script


def test_multi_view_rasterizer(tmp_path: Path):
    mesh = ProceduralMeshBuilder.build_capsule_limb(length=1.5, radius=0.4, segments=8, rings=4)
    rasterizer = MultiViewRasterizer(resolution=256)
    renders = rasterizer.render_views(mesh, tmp_path)

    assert len(renders) == 4
    for view in ["front", "side", "top", "isometric"]:
        assert view in renders
        assert renders[view].exists()
        assert renders[view].stat().st_size > 0


def test_vlm_critic_hill_climbing_loop(tmp_path: Path):
    climber = VLMCriticHillClimber(output_dir=tmp_path)
    result = climber.run_optimization(iterations=2)

    assert result.score >= 7.0
    assert (tmp_path / "final" / "model.obj").exists()
    assert (tmp_path / "final" / "render_script.py").exists()
    assert (tmp_path / "final" / "proof_render_pass_isometric.png").exists()
