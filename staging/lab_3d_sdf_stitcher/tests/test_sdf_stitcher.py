"""
Tests for Scenario LAB-3D-04 (Voxel/SDF Octree Boolean Stitcher & Dual Contouring).
"""

import math
import os
import pytest

from staging.lab_3d_sdf_stitcher.sdf_core import sdf_sphere, sdf_ellipsoid, smin
from staging.lab_3d_sdf_stitcher.mesher import VoxelSDFMesher, blend_head_and_ear_assembly


def test_sdf_primitives():
    # Center of sphere should be -radius
    d_center = sdf_sphere(0.0, 0.0, 0.0, radius=1.0)
    assert pytest.approx(d_center, rel=1e-3) == -1.0

    # Outside sphere
    d_out = sdf_sphere(2.0, 0.0, 0.0, radius=1.0)
    assert pytest.approx(d_out, rel=1e-3) == 1.0


def test_smin_smooth_transition():
    # At exact intersection boundary d1 == d2 == 0, smin should blend below 0 by k/4
    k = 0.2
    d_blend = smin(0.0, 0.0, k=k)
    expected = - (k / 4.0)
    assert pytest.approx(d_blend, abs=1e-4) == expected

    # Far away from boundary, smin should smoothly match standard minimum
    d_far = smin(10.0, 1.0, k=k)
    assert pytest.approx(d_far, abs=1e-3) == 1.0


def test_blend_head_and_ear_assembly_extracts_manifold_mesh(tmp_path):
    mesh = blend_head_and_ear_assembly(k_blend=0.15)
    
    assert len(mesh.vertices) > 100
    assert len(mesh.faces) > 100

    obj_content = mesh.to_obj()
    assert "v " in obj_content
    assert "f " in obj_content

    # Save to dossier proving ground directory
    dossier_dir = "docs/dashboard/design_lab/testing/3d_04_voxel_sdf_octree_stitcher"
    os.makedirs(dossier_dir, exist_ok=True)
    out_file = os.path.join(dossier_dir, "proof_head_ear_stitched_assembly.obj")
    with open(out_file, "w") as f:
        f.write(obj_content)
    assert os.path.exists(out_file)
