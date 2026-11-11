"""Multi-View Software Rasterizer for Procedural 3D Proving Grounds.

Renders 4-angle views (Front, Side, Top, Isometric) directly into PNG files
using pure Python + Pillow without requiring GPU drivers or external Blender binaries.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

from staging.3d_blender_vlm.procedural_generator import Mesh3D


class MultiViewRasterizer:
    """Renders 4-angle orthographic multi-pass views of a 3D mesh."""

    def __init__(self, resolution: int = 512, bg_color: Tuple[int, int, int] = (24, 24, 32)):
        self.resolution = resolution
        self.bg_color = bg_color

    def project_vertex(
        self,
        v: Tuple[float, float, float],
        view: str,
        scale: float = 120.0,
    ) -> Tuple[int, int, float]:
        """Project 3D vertex to 2D image coordinates with depth."""
        x, y, z = v
        cx = self.resolution // 2
        cy = self.resolution // 2

        if view == "front":
            px = cx + int(x * scale)
            py = cy - int(z * scale)
            depth = y
        elif view == "side":
            px = cx + int(y * scale)
            py = cy - int(z * scale)
            depth = -x
        elif view == "top":
            px = cx + int(x * scale)
            py = cy + int(y * scale)
            depth = -z
        elif view == "isometric":
            # Standard 30-degree isometric projection
            iso_x = (x - y) * math.cos(math.radians(30))
            iso_y = (x + y) * math.sin(math.radians(30)) - z
            px = cx + int(iso_x * scale)
            py = cy + int(iso_y * scale)
            depth = x + y + z
        else:
            px, py, depth = cx, cy, 0.0

        return px, py, depth

    def render_views(self, mesh: Mesh3D, output_dir: Path) -> Dict[str, Path]:
        """Render Front, Side, Top, and Isometric passes to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        views = ["front", "side", "top", "isometric"]
        rendered_paths: Dict[str, Path] = {}

        for view in views:
            img = Image.new("RGB", (self.resolution, self.resolution), color=self.bg_color)
            draw = ImageDraw.Draw(img)

            # Project all faces with simple depth sorting
            face_draw_list = []
            for face in mesh.faces:
                projected = [self.project_vertex(mesh.vertices[idx], view) for idx in face]
                avg_depth = sum(p[2] for p in projected) / len(projected)
                coords = [(p[0], p[1]) for p in projected]
                face_draw_list.append((avg_depth, coords))

            # Sort back-to-front
            face_draw_list.sort(key=lambda item: item[0])

            # Draw shaded faces and wireframe highlights
            for i, (_, coords) in enumerate(face_draw_list):
                # Simple pseudo-diffuse lighting
                shade = 80 + (i * 120) // max(1, len(face_draw_list))
                fill_color = (shade // 2, shade, min(255, shade + 30))
                draw.polygon(coords, fill=fill_color, outline=(200, 220, 255))

            # Add watermark text / view label
            draw.text((20, 20), f"Antimatter Proving Ground Pass: {view.upper()}", fill=(220, 220, 240))
            draw.text((20, 40), f"Mesh: {mesh.name} | Faces: {len(mesh.faces)}", fill=(140, 160, 180))

            file_path = output_dir / f"proof_render_pass_{view}.png"
            img.save(file_path, format="PNG")
            rendered_paths[view] = file_path

        return rendered_paths
