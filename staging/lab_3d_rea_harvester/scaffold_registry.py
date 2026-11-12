"""
REA (Reverse Engineering Analyst) Anatomical Scaffold & Socket Registry.
Codifies anatomical proportion constraints, socket frames, and baseline geometry.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple


@dataclass
class SocketFrame:
    """Anatomical socket coordinate frame specifying position, scale, and orientation."""
    name: str
    position: Tuple[float, float, float]    # (x, y, z) in meters / world units
    scale: Tuple[float, float, float]       # (sx, sy, sz) bounding box radii
    rotation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (yaw, pitch, roll) in radians
    blend_radius: float = 0.12              # smin smoothing radius k at interface boundary
    is_symmetric: bool = False              # If True, mirror on -X axis


class AnatomicalProportionRegistry:
    """
    Authoritative spatial skeleton defining anthropometric human head proportions:
    - Cranium: Base ellipsoid (width: 1.4m, height: 1.8m, depth: 1.5m)
    - Ears: Centered at level with eyes, spanning from brow line to nose base (~0.6m high)
    - Nose: Originates at nasion (y=0.10, z=0.68), projects forward with nasal flare
    - Jaw / Chin: Tapered anterior mandible (y=-0.75, z=0.35)
    - Neck: Cylindrical support column (y=-1.0, z=-0.1)
    """

    @staticmethod
    def get_canonical_sockets() -> Dict[str, SocketFrame]:
        return {
            "cranium": SocketFrame(
                name="cranium",
                position=(0.0, 0.0, 0.0),
                scale=(0.70, 0.90, 0.75),
                blend_radius=0.15
            ),
            "right_ear": SocketFrame(
                name="right_ear",
                position=(0.72, -0.05, -0.05),
                scale=(0.14, 0.30, 0.18),
                rotation_euler=(0.15, -0.08, 0.05),
                blend_radius=0.14,
                is_symmetric=True
            ),
            "left_ear": SocketFrame(
                name="left_ear",
                position=(-0.72, -0.05, -0.05),
                scale=(0.14, 0.30, 0.18),
                rotation_euler=(-0.15, -0.08, -0.05),
                blend_radius=0.14,
                is_symmetric=True
            ),
            "nose": SocketFrame(
                name="nose",
                position=(0.0, -0.05, 0.76),
                scale=(0.16, 0.32, 0.22),
                rotation_euler=(0.0, -0.20, 0.0),
                blend_radius=0.10
            ),
            "jaw_chin": SocketFrame(
                name="jaw_chin",
                position=(0.0, -0.72, 0.38),
                scale=(0.34, 0.26, 0.32),
                rotation_euler=(0.0, 0.15, 0.0),
                blend_radius=0.18
            ),
            "neck": SocketFrame(
                name="neck",
                position=(0.0, -1.05, -0.10),
                scale=(0.42, 0.45, 0.42),
                blend_radius=0.22
            ),
        }

    @staticmethod
    def validate_proportions(sockets: Dict[str, SocketFrame]) -> List[str]:
        """Audits socket frames for anatomical violations."""
        issues = []
        if "right_ear" in sockets and "left_ear" in sockets:
            re = sockets["right_ear"]
            le = sockets["left_ear"]
            # Check bilateral symmetry
            if abs(re.position[0] + le.position[0]) > 1e-4:
                issues.append("Bilateral asymmetry: ear X positions do not mirror accurately.")
            if abs(re.position[1] - le.position[1]) > 1e-4:
                issues.append("Ear vertical misalignment: left and right ears are at different heights.")

        if "nose" in sockets and "cranium" in sockets:
            nose = sockets["nose"]
            cranium = sockets["cranium"]
            # Nose must project anteriorly (+Z) beyond cranium surface
            cranium_front = cranium.position[2] + cranium.scale[2]
            if nose.position[2] < cranium_front - 0.10:
                issues.append("Nose socket is recessed inside the cranium volume.")

        return issues
