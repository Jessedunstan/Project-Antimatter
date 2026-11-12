"""Microsoft TRELLIS Structured Latents (SLAT) Engine for Project Antimatter.

Implements sparse 3D token representations (SLAT), rectified flow latent sampling,
and dual decoding into 3D Gaussian Splats and watertight manifold polygon meshes.
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GaussianSplat:
    """3D Gaussian Splat primitive."""
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)
    opacity: float
    color_rgb: Tuple[float, float, float]


@dataclass
class SLATRepresentation:
    """Structured Latents (SLAT) 3D volumetric token representation."""
    resolution: int  # Lattice grid dimension (e.g. 16x16x16 or 32x32x32)
    feature_dim: int  # Feature channels per token (e.g. 64)
    active_coords: np.ndarray  # (N, 3) integer grid coordinates
    features: np.ndarray  # (N, feature_dim) latent vectors
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLATEngine:
    """Core rectified flow sampler and decoder for Structured Latents."""

    def __init__(self, resolution: int = 16, feature_dim: int = 32, seed: int = 42):
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)

    def sample_rectified_flow(
        self,
        condition_vector: np.ndarray,
        steps: int = 12,
        guidance_scale: float = 3.0,
    ) -> SLATRepresentation:
        """Simulate rectified flow ODE trajectory from standard normal prior to SLAT."""
        # 1. Define active sparse occupancy grid (e.g., ellipsoid or target shape envelope)
        grid_coords = []
        center = self.resolution / 2.0
        radius = self.resolution * 0.4

        for x in range(self.resolution):
            for y in range(self.resolution):
                for z in range(self.resolution):
                    # Distance from normalized center
                    dx = (x - center) / radius
                    dy = (y - center) / radius
                    dz = (z - center) / radius
                    if (dx**2 + dy**2 + dz**2) <= 1.0:
                        grid_coords.append([x, y, z])

        coords = np.array(grid_coords, dtype=np.int32)
        n_tokens = len(coords)

        # 2. Initialize Gaussian noise prior: x_0 ~ N(0, I)
        latents = self.rng.standard_normal((n_tokens, self.feature_dim))

        # 3. Rectified flow Euler integration from t=0 to t=1: dx = v(x, t) dt
        dt = 1.0 / steps
        for step in range(steps):
            t = step * dt
            # Velocity field v(x, t) guided by condition vector
            cond_proj = condition_vector[:self.feature_dim] if len(condition_vector) >= self.feature_dim else np.pad(condition_vector, (0, self.feature_dim - len(condition_vector)))
            velocity = cond_proj * guidance_scale - latents * (1.0 - t)
            latents += velocity * dt

        # Normalize latent variance
        latents = np.tanh(latents)

        return SLATRepresentation(
            resolution=self.resolution,
            feature_dim=self.feature_dim,
            active_coords=coords,
            features=latents,
            metadata={"steps": steps, "guidance_scale": guidance_scale, "tokens": n_tokens},
        )

    def decode_to_gaussian_splats(self, slat: SLATRepresentation) -> List[GaussianSplat]:
        """Decode SLAT tokens into 3D Gaussian Splats for real-time radiance rendering."""
        splats: List[GaussianSplat] = []
        scale_factor = 2.0 / slat.resolution

        for i in range(len(slat.active_coords)):
            coord = slat.active_coords[i]
            feat = slat.features[i]

            # Position in normalized [-1.0, 1.0] coordinates
            pos = (
                float((coord[0] - slat.resolution / 2.0) * scale_factor),
                float((coord[1] - slat.resolution / 2.0) * scale_factor),
                float((coord[2] - slat.resolution / 2.0) * scale_factor),
            )

            # Extract scale, rotation, opacity, and color from latent slice
            scale_val = float(0.05 + 0.04 * (1.0 + feat[0]))
            opacity = float(0.5 + 0.5 * (1.0 + feat[1]) / 2.0)

            # Derive RGB from first 3 latent channels
            r = float(0.5 + 0.5 * math.sin(feat[0] * 3.14))
            g = float(0.5 + 0.5 * math.cos(feat[1] * 3.14))
            b = float(0.5 + 0.5 * math.sin((feat[2] if len(feat) > 2 else 0.0) * 3.14))

            splats.append(
                GaussianSplat(
                    position=pos,
                    scale=(scale_val, scale_val, scale_val),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                    opacity=opacity,
                    color_rgb=(r, g, b),
                )
            )

        return splats

    def decode_to_mesh(self, slat: SLATRepresentation) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Decode SLAT tokens into a manifold surface mesh using dual boundary extraction."""
        coord_set = set(tuple(c) for c in slat.active_coords)
        scale = 2.0 / slat.resolution
        center = slat.resolution / 2.0

        vertices: List[Tuple[float, float, float]] = []
        faces: List[List[int]] = []
        vert_map: Dict[Tuple[int, int, int], int] = {}

        def get_vert_idx(gx: int, gy: int, gz: int) -> int:
            k = (gx, gy, gz)
            if k not in vert_map:
                vx = float((gx - center) * scale)
                vy = float((gy - center) * scale)
                vz = float((gz - center) * scale)
                vert_map[k] = len(vertices)
                vertices.append((vx, vy, vz))
            return vert_map[k]

        # 6-neighborhood offsets for cube faces
        # (offset, 4 quad vertex coordinate offsets)
        cube_faces = [
            # +X
            ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            # -X
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            # +Y
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            # -Y
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            # +Z
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            # -Z
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ]

        for x, y, z in slat.active_coords:
            for (dx, dy, dz), quad_verts in cube_faces:
                neighbor = (x + dx, y + dy, z + dz)
                # If neighbor is empty space, this is a boundary face
                if neighbor not in coord_set:
                    face_indices = [get_vert_idx(x + qx, y + qy, z + qz) for qx, qy, qz in quad_verts]
                    faces.append(face_indices)

        return vertices, faces
