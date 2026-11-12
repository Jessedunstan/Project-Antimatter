"""Part Library Registry & Parametric Sizing for REA Asset Harvester."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

from staging.lab_3d_rea_harvester.asset_segmenter import HarvestedSubPart


class PartRegistry:
    """Manages harvested sub-parts and scales them to fit skeletal sockets."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._parts: Dict[str, HarvestedSubPart] = {}

    def register(self, part: HarvestedSubPart) -> None:
        """Register harvested part in memory and write OBJ to library."""
        self._parts[part.part_id] = part
        obj_file = self.library_dir / f"{part.part_id}.obj"
        obj_file.write_text(part.to_obj(), encoding="utf-8")

    def scale_part_to_target(
        self,
        part_id: str,
        target_dimensions: Tuple[float, float, float],
    ) -> List[Tuple[float, float, float]]:
        """Scale sub-part vertices to match target socket bounding volume."""
        part = self._parts.get(part_id)
        if not part or not part.vertices:
            return []

        (min_x, min_y, min_z), (max_x, max_y, max_z) = part.bounding_box
        curr_w = max(1e-4, max_x - min_x)
        curr_h = max(1e-4, max_y - min_y)
        curr_d = max(1e-4, max_z - min_z)

        target_w, target_h, target_d = target_dimensions
        sx = target_w / curr_w
        sy = target_h / curr_h
        sz = target_d / curr_d

        return [(x * sx, y * sy, z * sz) for x, y, z in part.vertices]

    @staticmethod
    def render_segmentation_proof(
        parts: List[HarvestedSubPart],
        output_path: Path,
        resolution: int = 512,
    ) -> Path:
        """Render multi-color visualization showing segmented anatomical decomposition."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (resolution, resolution), color=(20, 22, 28))
        draw = ImageDraw.Draw(img)

        colors = [
            (240, 90, 80),   # Red: Lower limbs
            (70, 180, 120),  # Green: Torso
            (60, 150, 240),  # Blue: Head/upper
            (240, 200, 60),  # Yellow
        ]

        cx = resolution // 2
        cy = resolution // 2
        scale = resolution * 0.22

        # Draw title
        draw.text((20, 20), "Antimatter REA Harvester: Anatomical Segmentation", fill=(230, 230, 240))

        legend_y = 50
        for i, part in enumerate(parts):
            col = colors[i % len(colors)]
            draw.rectangle([20, legend_y, 35, legend_y + 12], fill=col)
            draw.text((45, legend_y), f"{part.label} (Faces: {len(part.faces)})", fill=(200, 200, 210))
            legend_y += 20

            # Draw 3/4 isometric projected faces
            for face in part.faces:
                pts = []
                for idx in face:
                    x, y, z = part.vertices[idx]
                    iso_x = cx + int((x - y) * 0.7 * scale)
                    iso_y = cy + int(((x + y) * 0.4 - z * 0.8) * scale)
                    pts.append((iso_x, iso_y))

                if len(pts) >= 3:
                    draw.polygon(pts, fill=col, outline=(30, 30, 35))

        img.save(output_path, format="PNG")
        return output_path
