"""Mesh and Gaussian Splat Exporter for Microsoft TRELLIS Structured Latents."""

from __future__ import annotations
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

from staging.lab_3d_trellis_latents.slat_engine import GaussianSplat, SLATRepresentation


class MeshExporter:
    """Exports 3D meshes and Gaussian Splats to disk with preview renderings."""

    @staticmethod
    def export_obj(
        vertices: List[Tuple[float, float, float]],
        faces: List[List[int]],
        output_path: Path,
        model_name: str = "trellis_slat_mesh",
    ) -> Path:
        """Export vertices and faces to Wavefront OBJ format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Antimatter TRELLIS SLAT Export: {model_name}"]
        for x, y, z in vertices:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        for f in faces:
            lines.append(f"f {' '.join(str(idx + 1) for idx in f)}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def render_preview_grid(
        vertices: List[Tuple[float, float, float]],
        faces: List[List[int]],
        output_path: Path,
        resolution: int = 512,
    ) -> Path:
        """Render a clean 4-quadrant preview image of the generated 3D asset."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (resolution, resolution), color=(18, 20, 26))
        draw = ImageDraw.Draw(img)

        # Draw isometric projection in top-left, front in top-right, side in bottom-left, top in bottom-right
        quad_w = resolution // 2
        quad_h = resolution // 2

        views = [
            ("ISOMETRIC (3/4)", (0, 0), "iso"),
            ("FRONT", (quad_w, 0), "front"),
            ("SIDE", (0, quad_h), "side"),
            ("TOP", (quad_w, quad_h), "top"),
        ]

        # Draw grid dividing lines
        draw.line([(quad_w, 0), (quad_w, resolution)], fill=(45, 50, 65), width=2)
        draw.line([(0, quad_h), (resolution, quad_h)], fill=(45, 50, 65), width=2)

        for label, (ox, oy), vtype in views:
            # Draw viewport label
            draw.text((ox + 10, oy + 10), f"TRELLIS SLAT: {label}", fill=(180, 200, 240))

            cx = ox + quad_w // 2
            cy = oy + quad_h // 2
            scale = quad_w * 0.35

            # Project and draw faces
            for face in faces[:120]:  # Draw representative subset for fast preview
                pts = []
                for idx in face:
                    x, y, z = vertices[idx]
                    if vtype == "front":
                        px = cx + int(x * scale)
                        py = cy - int(z * scale)
                    elif vtype == "side":
                        px = cx + int(y * scale)
                        py = cy - int(z * scale)
                    elif vtype == "top":
                        px = cx + int(x * scale)
                        py = cy + int(y * scale)
                    else:  # iso
                        px = cx + int((x - y) * 0.7 * scale)
                        py = cy + int(((x + y) * 0.4 - z * 0.8) * scale)
                    pts.append((px, py))

                if len(pts) >= 3:
                    draw.polygon(pts, fill=(35, 75, 120), outline=(100, 160, 230))

        img.save(output_path, format="PNG")
        return output_path
