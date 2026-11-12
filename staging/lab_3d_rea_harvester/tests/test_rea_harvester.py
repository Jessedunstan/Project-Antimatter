"""Tests for REA Anatomical Kitbash Harvester (LAB-3D-03)."""

import pytest
from pathlib import Path
from staging.lab_3d_rea_harvester.asset_segmenter import AssetSegmenter, HarvestedSubPart
from staging.lab_3d_rea_harvester.part_registry import PartRegistry
from staging.lab_3d_blender_vlm.procedural_generator import ProceduralMeshBuilder


def test_asset_segmentation_by_slices():
    # Build a full character limb / capsule mesh
    mesh = ProceduralMeshBuilder.build_capsule_limb(length=3.0, radius=0.6, segments=8, rings=6)
    segmenter = AssetSegmenter()

    parts = segmenter.segment_by_height_slices(mesh.vertices, mesh.faces, slices=3)

    assert len(parts) == 3
    labels = [p.label for p in parts]
    assert "lower_limbs" in labels
    assert "torso" in labels
    assert "head_upper" in labels

    # Verify socket boundaries extracted
    for p in parts:
        assert isinstance(p.socket_boundary_indices, list)
        assert len(p.faces) > 0


def test_part_registry_and_scaling(tmp_path: Path):
    mesh = ProceduralMeshBuilder.build_capsule_limb(length=2.0, radius=0.5, segments=8, rings=4)
    segmenter = AssetSegmenter()
    parts = segmenter.segment_by_height_slices(mesh.vertices, mesh.faces, slices=2)

    registry = PartRegistry(library_dir=tmp_path / "lib")
    registry.register(parts[0])

    assert (tmp_path / "lib" / f"{parts[0].part_id}.obj").exists()

    # Scale to target dimensions (1.0, 1.0, 1.0)
    scaled_verts = registry.scale_part_to_target(parts[0].part_id, (1.0, 1.0, 1.0))
    assert len(scaled_verts) == len(parts[0].vertices)


def test_proof_rendering(tmp_path: Path):
    mesh = ProceduralMeshBuilder.build_capsule_limb(length=2.5, radius=0.5, segments=8, rings=4)
    segmenter = AssetSegmenter()
    parts = segmenter.segment_by_height_slices(mesh.vertices, mesh.faces, slices=3)

    proof_png = tmp_path / "proof_harvest_segmentation.png"
    PartRegistry.render_segmentation_proof(parts, proof_png)

    assert proof_png.exists()
    assert proof_png.stat().st_size > 0
