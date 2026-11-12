"""Tests for Microsoft TRELLIS Structured Latents (SLAT) Pipeline (LAB-3D-02)."""

import pytest
from pathlib import Path
from staging.lab_3d_trellis_latents.slat_engine import SLATEngine, SLATRepresentation
from staging.lab_3d_trellis_latents.pipeline import TrellisPipeline, TrellisGenerationResult


def test_slat_rectified_flow_sampling():
    engine = SLATEngine(resolution=12, feature_dim=16)
    cond = [0.5, -0.2, 0.8, 0.1]
    slat = engine.sample_rectified_flow(condition_vector=cond, steps=6)

    assert isinstance(slat, SLATRepresentation)
    assert slat.resolution == 12
    assert len(slat.active_coords) > 0
    assert slat.features.shape == (len(slat.active_coords), 16)


def test_slat_dual_decoding():
    engine = SLATEngine(resolution=10, feature_dim=16)
    slat = engine.sample_rectified_flow(condition_vector=[0.1, 0.2], steps=4)

    # 1. Gaussian Splats
    splats = engine.decode_to_gaussian_splats(slat)
    assert len(splats) == len(slat.active_coords)
    assert splats[0].opacity >= 0.0

    # 2. Polygon Mesh
    vertices, faces = engine.decode_to_mesh(slat)
    assert len(vertices) > 0
    assert len(faces) > 0
    assert all(len(f) == 4 for f in faces)  # Quads


def test_trellis_pipeline_end_to_end(tmp_path: Path):
    pipeline = TrellisPipeline(output_dir=tmp_path, resolution=12, feature_dim=16)
    result = pipeline.generate(prompt="anatomical human ear", steps=8)

    assert isinstance(result, TrellisGenerationResult)
    # Verify sub-15 second requirement from dossier
    assert result.generation_time_sec < 15.0
    assert Path(result.obj_path).exists()
    assert Path(result.preview_path).exists()
    assert result.num_vertices > 0
    assert result.num_splats > 0
