"""
Procedural 3D Geometry Generator for Anatomical Micro-Features.
Generates parametric OBJ meshes and standalone Blender Python (bpy) scripts.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple


@dataclass
class AnatomicalParams:
    """Parametric control vector for micro-feature generation."""
    feature_type: str = "ear"
    scale: float = 1.0
    helix_curl: float = 0.45       # Outer rim curvature [0.1, 1.0]
    concha_depth: float = 0.35     # Deep cavity depression [0.1, 0.8]
    lobule_fullness: float = 0.40  # Earlobe meatiness [0.1, 0.8]
    tragus_prominence: float = 0.25 # Inner ear flap [0.1, 0.6]
    subdivisions: int = 24         # Grid density per axis


@dataclass
class Mesh3D:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)

    def to_obj(self) -> str:
        lines = ["# Antimatter 3D Procedural Mesh", f"# Vertices: {len(self.vertices)} Faces: {len(self.faces)}"]
        for v in self.vertices:
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
        for n in self.normals:
            lines.append(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}")
        for f in self.faces:
            # OBJ is 1-indexed
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        return "\n".join(lines)


class ProceduralAnatomicalGenerator:
    """Generates procedural anatomical parts via parametric deformation grids."""

    def __init__(self, params: AnatomicalParams = None):
        self.params = params or AnatomicalParams()

    def generate_ear(self) -> Mesh3D:
        """Generate a procedural ear model using elliptical coordinates and SDF-derived offsets."""
        u_steps = self.params.subdivisions
        v_steps = self.params.subdivisions
        vertices = []
        faces = []

        # Coordinate grid: u represents radial angle (0 to 2pi), v represents radius (0 to 1)
        for i in range(u_steps):
            theta = 2.0 * math.pi * i / u_steps
            for j in range(v_steps):
                r = float(j) / (v_steps - 1)
                
                # Base oval projection (aspect ratio ~1.6 for human ear)
                x = r * math.cos(theta) * 0.65 * self.params.scale
                y = r * math.sin(theta) * 1.05 * self.params.scale
                
                # Height profile z(x, y) modeling ear anatomy
                # 1. Base taper: ear is flatter near head (small r)
                z = 0.05 * r
                
                # 2. Outer Helix Rim (at r near 1.0, theta in upper and posterior quadrants)
                if r > 0.65:
                    helix_factor = math.sin((r - 0.65) / 0.35 * math.pi)
                    # Helix curls forward and out
                    z += self.params.helix_curl * helix_factor * (0.8 + 0.3 * math.sin(theta))
                    x -= 0.15 * helix_factor * self.params.helix_curl
                
                # 3. Concha Cavity (central depression near origin, mostly posterior)
                if 0.15 < r < 0.55:
                    concha_factor = math.sin((r - 0.15) / 0.40 * math.pi)
                    z -= self.params.concha_depth * concha_factor
                
                # 4. Lobule (lower region: theta near -pi/2 or 3pi/2)
                if theta > math.pi and theta < 2.0 * math.pi:
                    lobule_weight = math.sin(theta - math.pi)
                    if r > 0.4:
                        y -= 0.25 * self.params.lobule_fullness * lobule_weight
                        z += 0.10 * self.params.lobule_fullness * lobule_weight

                # 5. Tragus flap (anterior inner border)
                if abs(theta) < 0.5 and 0.1 < r < 0.35:
                    z += self.params.tragus_prominence * math.cos(theta * 2.0)

                vertices.append((x, y, z))

        # Build triangular quad mesh faces
        for i in range(u_steps):
            next_i = (i + 1) % u_steps
            for j in range(v_steps - 1):
                idx0 = i * v_steps + j
                idx1 = i * v_steps + (j + 1)
                idx2 = next_i * v_steps + (j + 1)
                idx3 = next_i * v_steps + j
                
                faces.append((idx0, idx1, idx2))
                faces.append((idx0, idx2, idx3))

        mesh = Mesh3D(vertices=vertices, faces=faces)
        mesh.normals = self._compute_normals(mesh)
        return mesh

    def _compute_normals(self, mesh: Mesh3D) -> List[Tuple[float, float, float]]:
        normals = [(0.0, 0.0, 0.0)] * len(mesh.vertices)
        for f in mesh.faces:
            v0 = mesh.vertices[f[0]]
            v1 = mesh.vertices[f[1]]
            v2 = mesh.vertices[f[2]]
            
            # Cross product (v1 - v0) x (v2 - v0)
            ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            
            # Accumulate on vertices
            for idx in f:
                cur = normals[idx]
                normals[idx] = (cur[0] + nx, cur[1] + ny, cur[2] + nz)

        # Normalize
        normalized = []
        for n in normals:
            length = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
            if length > 1e-6:
                normalized.append((n[0]/length, n[1]/length, n[2]/length))
            else:
                normalized.append((0.0, 0.0, 1.0))
        return normalized

    def export_blender_script(self, output_filepath: str) -> str:
        """Generate a standalone Python script executable in Blender (bpy)."""
        p = self.params
        script = f"""# Headless Blender Generation Script - Generated by Antimatter
import bpy
import bmesh
import math

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create procedural mesh container
mesh_data = bpy.data.meshes.new("Procedural_{p.feature_type}")
obj = bpy.data.objects.new("Procedural_{p.feature_type}_Obj", mesh_data)
bpy.context.collection.objects.link(obj)

# Create bmesh
bm = bmesh.new()

u_steps = {p.subdivisions}
v_steps = {p.subdivisions}
verts = []

for i in range(u_steps):
    theta = 2.0 * math.pi * i / u_steps
    row = []
    for j in range(v_steps):
        r = float(j) / (v_steps - 1)
        x = r * math.cos(theta) * 0.65 * {p.scale}
        y = r * math.sin(theta) * 1.05 * {p.scale}
        z = 0.05 * r
        if r > 0.65:
            helix_factor = math.sin((r - 0.65) / 0.35 * math.pi)
            z += {p.helix_curl} * helix_factor * (0.8 + 0.3 * math.sin(theta))
            x -= 0.15 * helix_factor * {p.helix_curl}
        if 0.15 < r < 0.55:
            concha_factor = math.sin((r - 0.15) / 0.40 * math.pi)
            z -= {p.concha_depth} * concha_factor
        if math.pi < theta < 2.0 * math.pi:
            lobule_weight = math.sin(theta - math.pi)
            if r > 0.4:
                y -= 0.25 * {p.lobule_fullness} * lobule_weight
                z += 0.10 * {p.lobule_fullness} * lobule_weight
        row.append(bm.verts.new((x, y, z)))
    verts.append(row)

for i in range(u_steps):
    next_i = (i + 1) % u_steps
    for j in range(v_steps - 1):
        v0 = verts[i][j]
        v1 = verts[i][j + 1]
        v2 = verts[next_i][j + 1]
        v3 = verts[next_i][j]
        bm.faces.new((v0, v1, v2, v3))

bm.to_mesh(mesh_data)
bm.free()

# Apply subdivision surface modifier for organic smoothing
subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2

# Export output OBJ
bpy.ops.wm.obj_export(filepath=r"{output_filepath}")
"""
        return script
