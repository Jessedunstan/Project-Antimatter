# Proving Ground Dossier: Scenario LAB-3D-04
## Hierarchical Voxel/SDF Octree Boolean Stitcher & Dual Contouring

### Objective
Build and verify the spatial assembly and seam-healing engine inside `staging/3d_sdf_stitcher/`:
1. Ingest separately generated sub-part meshes and place them according to the kinematic skeleton matrix.
2. Rasterize input meshes into high-resolution Signed Distance Fields (SDFs) / OpenVDB grids.
3. Perform smooth Boolean union ($A \cup B$) with radius-controlled blending at junction seams (e.g. ear-to-skull, neck-to-torso).
4. Run Dual Contouring / Marching Cubes to generate a unified, watertight, manifold mesh, followed by Quad Remeshing.

### Required Artifacts in This Dossier
- `README.md` (this file)
- `scenario_config.json`
- `proof_boolean_blend_*.png`
- Hand-back block in `HANDOFF_STATE.md`
