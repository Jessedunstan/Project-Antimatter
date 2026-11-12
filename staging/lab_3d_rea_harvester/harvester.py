"""
REA Asset Harvester & Reverse-Engineering Transformer.
Extracts boundary loops, normalizes raw geometric assets, and scales parts into socket frames.
"""

from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Optional, Tuple

from .scaffold_registry import SocketFrame, AnatomicalProportionRegistry


@dataclass
class HarvestedPart:
    name: str
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]
    boundary_loop_indices: List[int] = field(default_factory=list)
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0,0,0), (0,0,0))


class REAAssetHarvester:
    """Reverse Engineer Analyst pipeline for decomposing and normalizing 3D meshes."""

    def __init__(self):
        self.registry = AnatomicalProportionRegistry()

    def parse_obj(self, obj_text: str, part_name: str = "part") -> HarvestedPart:
        """Parses an OBJ formatted string into a structured HarvestedPart."""
        vertices = []
        faces = []
        
        for line in obj_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                # Support face formats like v, v/vt, v/vt/vn
                v_indices = []
                for p in parts[1:4]:
                    idx = int(p.split("/")[0]) - 1 # 1-indexed to 0-indexed
                    v_indices.append(idx)
                faces.append((v_indices[0], v_indices[1], v_indices[2]))

        if not vertices:
            return HarvestedPart(name=part_name, vertices=[], faces=[])

        # Compute centroid & bounds
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        min_pt = (min(xs), min(ys), min(zs))
        max_pt = (max(xs), max(ys), max(zs))
        centroid = ((min_pt[0] + max_pt[0]) / 2.0, (min_pt[1] + max_pt[1]) / 2.0, (min_pt[2] + max_pt[2]) / 2.0)

        # Detect boundary vertices (edges belonging to only 1 face)
        edge_counts: Dict[Tuple[int, int], int] = {}
        for f in faces:
            edges = [
                tuple(sorted((f[0], f[1]))),
                tuple(sorted((f[1], f[2]))),
                tuple(sorted((f[2], f[0]))),
            ]
            for e in edges:
                edge_counts[e] = edge_counts.get(e, 0) + 1

        boundary_indices = set()
        for e, count in edge_counts.items():
            if count == 1:
                boundary_indices.add(e[0])
                boundary_indices.add(e[1])

        return HarvestedPart(
            name=part_name,
            vertices=vertices,
            faces=faces,
            boundary_loop_indices=list(boundary_indices),
            centroid=centroid,
            bounding_box=(min_pt, max_pt)
        )

    def fit_to_socket(self, part: HarvestedPart, socket: SocketFrame) -> HarvestedPart:
        """Transforms, rotates, and scales harvested part to conform to target socket frame."""
        if not part.vertices:
            return part

        min_pt, max_pt = part.bounding_box
        orig_dx = max(max_pt[0] - min_pt[0], 1e-5)
        orig_dy = max(max_pt[1] - min_pt[1], 1e-5)
        orig_dz = max(max_pt[2] - min_pt[2], 1e-5)

        target_sx, target_sy, target_sz = socket.scale
        # Desired dimension is 2 * radius
        scale_x = (2.0 * target_sx) / orig_dx
        scale_y = (2.0 * target_sy) / orig_dy
        scale_z = (2.0 * target_sz) / orig_dz

        yaw, pitch, roll = socket.rotation_euler
        tx, ty, tz = socket.position
        cx, cy, cz = part.centroid

        fitted_verts = []
        for v in part.vertices:
            # 1. Center at origin
            lx = (v[0] - cx) * scale_x
            ly = (v[1] - cy) * scale_y
            lz = (v[2] - cz) * scale_z

            # 2. Rotate
            # Yaw (Y)
            cy_a, sy_a = math.cos(yaw), math.sin(yaw)
            rx = lx * cy_a + lz * sy_a
            rz = -lx * sy_a + lz * cy_a
            # Pitch (X)
            cp_a, sp_a = math.cos(pitch), math.sin(pitch)
            ry = ly * cp_a - rz * sp_a
            rz = ly * sp_a + rz * cp_a
            # Roll (Z)
            cr_a, sr_a = math.cos(roll), math.sin(roll)
            fx = rx * cr_a - ry * sr_a
            fy = rx * sr_a + ry * cr_a

            # 3. Translate to target socket
            fitted_verts.append((fx + tx, fy + ty, rz + tz))

        return HarvestedPart(
            name=f"{part.name}_fitted_to_{socket.name}",
            vertices=fitted_verts,
            faces=part.faces,
            boundary_loop_indices=part.boundary_loop_indices,
            centroid=socket.position,
            bounding_box=((tx - target_sx, ty - target_sy, tz - target_sz), (tx + target_sx, ty + target_sy, tz + target_sz))
        )
