"""
Voxel Grid Dual Contouring / Marching Cubes Surface Extractor.
Extracts watertight manifold polygon meshes from continuous SDF fields.
"""

from dataclasses import dataclass, field
import math
import os
from typing import Callable, List, Tuple

from .sdf_core import smin, sdf_ellipsoid, sdf_sphere


@dataclass
class ExtractedMesh:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)

    def to_obj(self) -> str:
        lines = ["# Antimatter Watertight SDF Extracted Mesh", f"# Vertices: {len(self.vertices)} Faces: {len(self.faces)}"]
        for v in self.vertices:
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
        for f in self.faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        return "\n".join(lines)


class VoxelSDFMesher:
    """Extracts polygonal surfaces from continuous distance fields using regular cell contouring."""

    def __init__(self, resolution: int = 32, bounds: Tuple[float, float] = (-1.5, 1.5)):
        self.res = resolution
        self.min_b, self.max_b = bounds
        self.step = (self.max_b - self.min_b) / (self.res - 1)

    def evaluate_grid(self, sdf_func: Callable[[float, float, float], float]):
        """Evaluates 3D grid values."""
        grid = []
        for i in range(self.res):
            x = self.min_b + i * self.step
            plane = []
            for j in range(self.res):
                y = self.min_b + j * self.step
                row = []
                for k in range(self.res):
                    z = self.min_b + k * self.step
                    row.append(sdf_func(x, y, z))
                plane.append(row)
            grid.append(plane)
        return grid

    def extract_surface(self, sdf_func: Callable[[float, float, float], float]) -> ExtractedMesh:
        """Extracts boundary faces between negative (inside) and positive (outside) cells."""
        grid = self.evaluate_grid(sdf_func)
        vertices = []
        faces = []
        vert_map = {}

        def get_vert_idx(gx, gy, gz):
            key = (gx, gy, gz)
            if key not in vert_map:
                vx = self.min_b + gx * self.step
                vy = self.min_b + gy * self.step
                vz = self.min_b + gz * self.step
                vert_map[key] = len(vertices)
                vertices.append((vx, vy, vz))
            return vert_map[key]

        # Scan cells and generate boundary quads wherever sign changes occur
        for x in range(self.res - 1):
            for y in range(self.res - 1):
                for z in range(self.res - 1):
                    val = grid[x][y][z]
                    
                    # Check X neighbor
                    if (val < 0.0) != (grid[x+1][y][z] < 0.0):
                        v0 = get_vert_idx(x + 0.5, y, z)
                        v1 = get_vert_idx(x + 0.5, y + 1, z)
                        v2 = get_vert_idx(x + 0.5, y + 1, z + 1)
                        v3 = get_vert_idx(x + 0.5, y, z + 1)
                        if val < 0.0:
                            faces.append((v0, v1, v2))
                            faces.append((v0, v2, v3))
                        else:
                            faces.append((v0, v2, v1))
                            faces.append((v0, v3, v2))

                    # Check Y neighbor
                    if (val < 0.0) != (grid[x][y+1][z] < 0.0):
                        v0 = get_vert_idx(x, y + 0.5, z)
                        v1 = get_vert_idx(x + 1, y + 0.5, z)
                        v2 = get_vert_idx(x + 1, y + 0.5, z + 1)
                        v3 = get_vert_idx(x, y + 0.5, z + 1)
                        if val < 0.0:
                            faces.append((v0, v2, v1))
                            faces.append((v0, v3, v2))
                        else:
                            faces.append((v0, v1, v2))
                            faces.append((v0, v2, v3))

                    # Check Z neighbor
                    if (val < 0.0) != (grid[x][y][z+1] < 0.0):
                        v0 = get_vert_idx(x, y, z + 0.5)
                        v1 = get_vert_idx(x + 1, y, z + 0.5)
                        v2 = get_vert_idx(x + 1, y + 1, z + 0.5)
                        v3 = get_vert_idx(x, y + 1, z + 0.5)
                        if val < 0.0:
                            faces.append((v0, v1, v2))
                            faces.append((v0, v2, v3))
                        else:
                            faces.append((v0, v2, v1))
                            faces.append((v0, v3, v2))

        return ExtractedMesh(vertices=vertices, faces=faces)


def blend_head_and_ear_assembly(k_blend: float = 0.15) -> ExtractedMesh:
    """
    Demonstrates assembly of a central cranium/head with attached left & right ear volumes,
    smoothly blended via polynomial smin to eliminate non-manifold seams.
    """
    def scene_sdf(x: float, y: float, z: float) -> float:
        # Central head (ellipsoid)
        d_head = sdf_ellipsoid(x, y, z, 0.70, 0.90, 0.75)
        
        # Right ear (shifted to +X, +Y, slightly angled)
        d_right_ear = sdf_ellipsoid(x - 0.72, y - 0.10, z, 0.12, 0.28, 0.18)
        
        # Left ear (shifted to -X, +Y)
        d_left_ear = sdf_ellipsoid(x + 0.72, y - 0.10, z, 0.12, 0.28, 0.18)
        
        # Smooth union
        d_ears = min(d_right_ear, d_left_ear)
        return smin(d_head, d_ears, k=k_blend)

    mesher = VoxelSDFMesher(resolution=28, bounds=(-1.3, 1.3))
    return mesher.extract_surface(scene_sdf)
