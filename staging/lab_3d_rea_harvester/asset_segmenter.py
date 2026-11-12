"""Reverse Engineering Analyst (REA) Anatomical 3D Asset Segmenter.

Ingests monolithic 3D meshes, decomposes them into anatomical sub-components
(limbs, joints, head, torso), extracts socket boundary loops, and registers them.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class HarvestedSubPart:
    """Anatomical sub-mesh extracted from a host asset."""
    part_id: str
    label: str  # e.g., 'head', 'torso', 'arm_left', 'ear'
    vertices: List[Tuple[float, float, float]]
    faces: List[List[int]]
    socket_boundary_indices: List[int] = field(default_factory=list)
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(
        default_factory=lambda: ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    )

    def to_obj(self) -> str:
        """Export sub-part to OBJ format."""
        lines = [f"# Antimatter REA Harvested Part: {self.label} ({self.part_id})"]
        for x, y, z in self.vertices:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        for f in self.faces:
            lines.append(f"f {' '.join(str(idx + 1) for idx in f)}")
        return "\n".join(lines)


class AssetSegmenter:
    """Decomposes 3D meshes into canonical sub-components."""

    @staticmethod
    def parse_obj(obj_content: str) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Simple Wavefront OBJ parser."""
        vertices = []
        faces = []
        for line in obj_content.splitlines():
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Handle v/vt/vn format
                face = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face)
        return vertices, faces

    def segment_by_height_slices(
        self,
        vertices: List[Tuple[float, float, float]],
        faces: List[List[int]],
        slices: int = 3,
    ) -> List[HarvestedSubPart]:
        """Decompose mesh into height-stratified anatomical parts (e.g. legs, torso, head)."""
        if not vertices:
            return []

        v_arr = np.array(vertices)
        z_min = float(np.min(v_arr[:, 2]))
        z_max = float(np.max(v_arr[:, 2]))
        z_span = max(1e-5, z_max - z_min)

        slice_labels = ["lower_limbs", "torso", "head_upper"]
        parts: List[HarvestedSubPart] = []

        for s in range(slices):
            label = slice_labels[s] if s < len(slice_labels) else f"section_{s}"
            t_low = z_min + (s / slices) * z_span
            t_high = z_min + ((s + 1) / slices) * z_span

            # Find vertices belonging to this vertical segment
            mask = (v_arr[:, 2] >= t_low) & (v_arr[:, 2] <= t_high)
            orig_indices = np.where(mask)[0]
            if len(orig_indices) == 0:
                continue

            orig_to_new = {orig: new for new, orig in enumerate(orig_indices)}
            sub_verts = [vertices[idx] for idx in orig_indices]

            # Find faces where all or majority vertices belong to segment
            sub_faces = []
            for face in faces:
                if all(idx in orig_to_new for idx in face):
                    sub_faces.append([orig_to_new[idx] for idx in face])

            # Extract socket boundary: vertices on the boundary height
            socket_indices = [
                new for new, orig in enumerate(orig_indices)
                if abs(vertices[orig][2] - t_low) < (0.1 * z_span / slices) or abs(vertices[orig][2] - t_high) < (0.1 * z_span / slices)
            ]

            sub_v_arr = np.array(sub_verts)
            bbox = (
                (float(np.min(sub_v_arr[:, 0])), float(np.min(sub_v_arr[:, 1])), float(np.min(sub_v_arr[:, 2]))),
                (float(np.max(sub_v_arr[:, 0])), float(np.max(sub_v_arr[:, 1])), float(np.max(sub_v_arr[:, 2]))),
            )

            parts.append(
                HarvestedSubPart(
                    part_id=f"rea_part_{label}",
                    label=label,
                    vertices=sub_verts,
                    faces=sub_faces,
                    socket_boundary_indices=socket_indices,
                    bounding_box=bbox,
                )
            )

        return parts
