"""
Signed Distance Field (SDF) Core Mathematical Engine.
Implements continuous geometric primitives and smooth polynomial Boolean blending.
"""

import math
from typing import Callable, List, Tuple


def length3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def sdf_sphere(x: float, y: float, z: float, radius: float) -> float:
    return length3(x, y, z) - radius


def sdf_ellipsoid(x: float, y: float, z: float, rx: float, ry: float, rz: float) -> float:
    # Approximate algebraic distance scaled to metric distance
    k0 = length3(x / rx, y / ry, z / rz)
    k1 = length3(x / (rx * rx), y / (ry * ry), z / (rz * rz))
    return k0 * (k0 - 1.0) / k1 if k1 > 1e-6 else k0 - 1.0


def sdf_cylinder(x: float, y: float, z: float, r: float, h: float) -> float:
    d_xy = math.sqrt(x * x + y * y) - r
    d_z = abs(z) - h
    outside = math.sqrt(max(d_xy, 0.0)**2 + max(d_z, 0.0)**2)
    inside = min(max(d_xy, d_z), 0.0)
    return outside + inside


def smin(d1: float, d2: float, k: float = 0.1) -> float:
    """
    Polynomial smooth minimum union: A ∪ B with smooth blending radius k.
    Eliminates non-manifold boundary seams and creasing artifacts.
    """
    if k <= 1e-6:
        return min(d1, d2)
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


def smax(d1: float, d2: float, k: float = 0.1) -> float:
    """Polynomial smooth maximum intersection / subtraction."""
    if k <= 1e-6:
        return max(d1, d2)
    h = max(k - abs(-d1 - d2), 0.0) / k
    return max(d1, d2) + h * h * k * 0.25
