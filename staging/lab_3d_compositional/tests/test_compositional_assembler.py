"""
Tests for Scenario LAB-3D-05 (Unified Compositional 3D Assembler).
"""

import os
import pytest

from staging.lab_3d_compositional.assembler import UnifiedCompositionalAssembler


def test_assembler_evaluates_bust_sdf():
    assembler = UnifiedCompositionalAssembler(resolution=16)
    
    # Origin should be inside cranium (d < 0)
    d_center = assembler.evaluate_bust_sdf(0.0, 0.0, 0.0)
    assert d_center < 0.0

    # Far point outside should be positive (d > 0)
    d_outside = assembler.evaluate_bust_sdf(5.0, 5.0, 5.0)
    assert d_outside > 0.0


def test_assembler_extracts_and_renders(tmp_path):
    assembler = UnifiedCompositionalAssembler(resolution=22)
    mesh = assembler.assemble()

    assert len(mesh.vertices) > 200
    assert len(mesh.faces) > 200

    out_dir = str(tmp_path / "renders")
    paths = assembler.render_turntable(mesh, out_dir)

    assert "obj_model" in paths
    assert os.path.exists(paths["obj_model"])
    assert "composite" in paths
    assert os.path.exists(paths["composite"])
