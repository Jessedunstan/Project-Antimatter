"""
Multi-View Software Camera Rig & 3D Rasterizer.
Renders 4 canonical angles (Front, Side, 3/4 Perspective, Top) with z-buffering and Lambertian shading.
"""

import math
import os
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

from .generator import Mesh3D


class SoftwareCameraRenderer:
    """Renders 3D meshes to 2D image passes without GPU or external window dependencies."""

    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height

    def render_angles(self, mesh: Mesh3D, output_dir: str, prefix: str = "pass") -> Dict[str, str]:
        """Renders 4 distinct camera views of the mesh and saves them as PNG files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Canonical angles: (yaw, pitch, roll) in radians
        angles = {
            "front": (0.0, 0.0, 0.0),
            "side": (math.pi / 2.0, 0.0, 0.0),
            "perspective_34": (math.pi / 4.0, -math.pi / 6.0, 0.0),
            "top": (0.0, -math.pi / 2.0, 0.0),
        }

        paths = {}
        for name, (yaw, pitch, roll) in angles.items():
            img = self._render_view(mesh, yaw, pitch, roll)
            filepath = os.path.join(output_dir, f"{prefix}_{name}.png")
            img.save(filepath)
            paths[name] = filepath

        # Also create a composite 2x2 contact sheet for single-glance VLM inspection
        contact_path = os.path.join(output_dir, f"{prefix}_composite_4view.png")
        self._create_contact_sheet(paths, contact_path)
        paths["composite"] = contact_path
        return paths

    def _rotate_vertex(self, v: Tuple[float, float, float], yaw: float, pitch: float, roll: float) -> Tuple[float, float, float]:
        x, y, z = v
        # Yaw (around Y axis)
        cy, sy = math.cos(yaw), math.sin(yaw)
        x1 = x * cy + z * sy
        z1 = -x * sy + z * cy
        
        # Pitch (around X axis)
        cp, sp = math.cos(pitch), math.sin(pitch)
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp
        
        # Roll (around Z axis)
        cr, sr = math.cos(roll), math.sin(roll)
        x3 = x1 * cr - y2 * sr
        y3 = x1 * sr + y2 * cr
        return (x3, y2, z2)

    def _render_view(self, mesh: Mesh3D, yaw: float, pitch: float, roll: float) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), color=(15, 20, 30))
        draw = ImageDraw.Draw(img)
        
        # Light vector (directional key light from upper right front)
        lx, ly, lz = 0.577, 0.577, 0.577

        # Transform vertices
        transformed = [self._rotate_vertex(v, yaw, pitch, roll) for v in mesh.vertices]
        trans_normals = [self._rotate_vertex(n, yaw, pitch, roll) for n in mesh.normals]

        # Calculate face depth for painter's algorithm (sorting by average Z)
        face_depths = []
        for i, f in enumerate(mesh.faces):
            avg_z = (transformed[f[0]][2] + transformed[f[1]][2] + transformed[f[2]][2]) / 3.0
            face_depths.append((avg_z, i))
        face_depths.sort(key=lambda item: item[0]) # draw farthest first

        # Scale and center to viewport
        scale = min(self.width, self.height) * 0.38
        cx = self.width / 2.0
        cy = self.height / 2.0

        for _, f_idx in face_depths:
            f = mesh.faces[f_idx]
            v0, v1, v2 = transformed[f[0]], transformed[f[1]], transformed[f[2]]
            
            # Normal calculation for face lighting
            ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            fnx = ay * bz - az * by
            fny = az * bx - ax * bz
            fnz = ax * by - ay * bx
            norm_len = math.sqrt(fnx*fnx + fny*fny + fnz*fnz)
            if norm_len < 1e-6:
                continue
            fnx, fny, fnz = fnx / norm_len, fny / norm_len, fnz / norm_len

            # Backface culling: discard faces pointing away from camera (camera looks down +Z)
            if fnz < -0.05:
                continue

            # Lambertian diffuse lighting + ambient
            diffuse = max(0.0, fnx * lx + fny * ly + fnz * lz)
            intensity = 0.25 + 0.75 * diffuse # ambient + key
            
            # Shading color (cyan-slate tone matching Antimatter aesthetic)
            r_col = int(min(255, 60 * intensity + 40))
            g_col = int(min(255, 180 * intensity + 30))
            b_col = int(min(255, 230 * intensity + 20))

            # Project 3D -> 2D (invert Y for screen space)
            p0 = (cx + v0[0] * scale, cy - v0[1] * scale)
            p1 = (cx + v1[0] * scale, cy - v1[1] * scale)
            p2 = (cx + v2[0] * scale, cy - v2[1] * scale)

            draw.polygon([p0, p1, p2], fill=(r_col, g_col, b_col), outline=(30, 45, 65))

        return img

    def _create_contact_sheet(self, paths: Dict[str, str], output_path: str):
        """Combine 4 single views into a 2x2 grid."""
        w, h = self.width, self.height
        composite = Image.new("RGB", (w * 2, h * 2), color=(10, 14, 22))
        
        placements = [
            ("front", 0, 0),
            ("side", w, 0),
            ("perspective_34", 0, h),
            ("top", w, h),
        ]
        
        for name, ox, oy in placements:
            if name in paths and os.path.exists(paths[name]):
                tile = Image.open(paths[name])
                composite.paste(tile, (ox, oy))
                
        # Draw grid labels
        draw = ImageDraw.Draw(composite)
        labels = [
            ("FRONT VIEW", 10, 10),
            ("SIDE / PROFILE", w + 10, 10),
            ("3/4 PERSPECTIVE", 10, h + 10),
            ("TOP ORTHOGRAPHIC", w + 10, h + 10),
        ]
        for text, tx, ty in labels:
            draw.rectangle([tx, ty, tx + 130, ty + 18], fill=(0, 0, 0, 180))
            draw.text((tx + 6, ty + 3), text, fill=(0, 240, 255))

        composite.save(output_path)
