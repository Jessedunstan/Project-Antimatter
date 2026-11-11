# Proving Ground Dossier: Scenario LAB-3D-03
## REA (Reverse Engineering Analyst) Anatomical Kitbash Harvester

### Objective
Build and verify the REA 3D asset extraction pipeline inside `staging/3d_rea_harvester/`:
1. Search and ingest open 3D assets (Objaverse, Thingiverse, Sketchfab, local GLB/OBJ repository).
2. Segment canonical anatomical sub-parts (ears, hands, noses, torsos) using geometric clustering / semantic segmentation.
3. Normalize coordinates, extract socket boundary loops, and register into a local part vector library.
4. Provide parametric scaling that maps harvested parts onto target proportion skeletons.

### Required Artifacts in This Dossier
- `README.md` (this file)
- `scenario_config.json`
- `proof_harvest_segmentation_*.png`
- Hand-back block in `HANDOFF_STATE.md`
