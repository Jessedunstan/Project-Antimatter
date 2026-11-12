"""
Unified Compositional 3D Assembler.
Coordinates multi-part anatomical generation, enforces global proportion cages,
executes multi-part SDF smooth Boolean fusion, and outputs watertight character busts.
"""

from dataclasses import dataclass
import math
import os
from typing import Dict, List, Tuple

from staging.lab_3d_rea_harvester.scaffold_registry import AnatomicalProportionRegistry, SocketFrame
from staging.lab_3d_sdf_stitcher.sdf_core import smin, sdf_ellipsoid, sdf_sphere
from staging.lab_3d_sdf_stitcher.mesher import VoxelSDFMesher, ExtractedMesh
from staging.lab_3d_blender_vlm.renderer import SoftwareCameraRenderer, Mesh3D


class UnifiedCompositionalAssembler:
    """
    Coordinates multi-part anatomical generation and smooth geometric fusion:
    1. Loads canonical socket constraints from AnatomicalProportionRegistry.
    2. Builds continuous analytical + mesh distance representations for each part.
    3. Blends cranial vault, bilateral ears, anterior nasal bridge, mental chin, and cervical neck.
    4. Extracts watertight manifold surface via high-resolution cell contouring.
    """

    def __init__(self, resolution: int = 36, bounds: Tuple[float, float] = (-1.5, 1.5)):
        self.resolution = resolution
        self.bounds = bounds
        self.registry = AnatomicalProportionRegistry()
        self.sockets = self.registry.get_canonical_sockets()

    def evaluate_bust_sdf(self, x: float, y: float, z: float) -> float:
        """Evaluates the composite distance field of the complete character bust."""
        soc = self.sockets

        # 1. Cranium / Skull Vault
        cr = soc["cranium"]
        d_cranium = sdf_ellipsoid(
            x - cr.position[0], y - cr.position[1], z - cr.position[2],
            cr.scale[0], cr.scale[1], cr.scale[2]
        )

        # 2. Right Ear
        re = soc["right_ear"]
        d_re = sdf_ellipsoid(
            x - re.position[0], y - re.position[1], z - re.position[2],
            re.scale[0], re.scale[1], re.scale[2]
        )

        # 3. Left Ear
        le = soc["left_ear"]
        d_le = sdf_ellipsoid(
            x - le.position[0], y - le.position[1], z - le.position[2],
            le.scale[0], le.scale[1], le.scale[2]
        )

        # 4. Nose Projection (anterior pyramid / ellipsoid)
        no = soc["nose"]
        d_nose = sdf_ellipsoid(
            x - no.position[0], y - no.position[1], z - no.position[2],
            no.scale[0], no.scale[1], no.scale[2]
        )

        # 5. Mandible / Chin prominence
        jw = soc["jaw_chin"]
        d_jaw = sdf_ellipsoid(
            x - jw.position[0], y - jw.position[1], z - jw.position[2],
            jw.scale[0], jw.scale[1], jw.scale[2]
        )

        # 6. Cervical Neck Support
        nk = soc["neck"]
        d_neck = sdf_ellipsoid(
            x - nk.position[0], y - nk.position[1], z - nk.position[2],
            nk.scale[0], nk.scale[1], nk.scale[2]
        )

        # Smooth Hierarchical Fusion:
        # Cranium + Neck (smooth blend k=0.22)
        d_core = smin(d_cranium, d_neck, k=nk.blend_radius)
        # Core + Mandible (smooth blend k=0.18)
        d_head = smin(d_core, d_jaw, k=jw.blend_radius)
        # Head + Nose (sharper blend k=0.10 to preserve nasal bridge)
        d_with_nose = smin(d_head, d_nose, k=no.blend_radius)
        # Head + Bilateral Ears (clean anatomical fold k=0.14)
        d_ears = min(d_re, d_le)
        d_final = smin(d_with_nose, d_ears, k=re.blend_radius)

        return d_final

    def assemble(self) -> ExtractedMesh:
        """Extracts watertight polygon mesh from composite bust SDF."""
        mesher = VoxelSDFMesher(resolution=self.resolution, bounds=self.bounds)
        return mesher.extract_surface(self.evaluate_bust_sdf)

    def render_turntable(self, mesh: ExtractedMesh, output_dir: str) -> Dict[str, str]:
        """Renders 8-angle turntable inspection passes and composite contact sheets."""
        os.makedirs(output_dir, exist_ok=True)
        m3d = Mesh3D(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=[(0.0, 0.0, 1.0)] * len(mesh.vertices)
        )
        
        renderer = SoftwareCameraRenderer(width=320, height=320)
        paths = renderer.render_angles(m3d, output_dir, prefix="bust_assembly")
        
        # Save production OBJ
        obj_path = os.path.join(output_dir, "final_assembled_character_bust.obj")
        with open(obj_path, "w") as f:
            f.write(mesh.to_obj())
        paths["obj_model"] = obj_path

        return paths
