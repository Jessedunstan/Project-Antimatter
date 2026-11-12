# Project Antimatter — Master Handoff & Sprint Communication Bus

**Current Date / Session:** Antimatter Research & Design Lab Initialized  
**Architecture Baseline:** Python 3.11+ / FastAPI / Tauri v2 / Multi-Backend Inference (Ollama/vLLM)  
**Main Design Lab Hub:** `docs/dashboard/design_lab.html`  
**Staging Area:** `staging/<Feature>/` (Strict isolation — never modifies mainline `antimatter/` directly)  

---

## 0. Active Sprint Hand-Back Queue (File-Backed Communication Bus)

### [PENDING COMMANDER INGEST — 2026-11-12]
- **Sprint Name:** `MINI DESIGN LAB: Scenario LAB-3D-05 — Unified Compositional 3D Character Bust Assembler`
- **Origin Chat:** `LAB: Compositional Assembler`
- **Target Chat:** `1 - COMMANDER`
- **Status:** `READY FOR COMMANDER ARCHIVE`
- **Verification Evidence:**
  - 2/2 tests green in `staging/lab_3d_compositional/tests/` (`test_compositional_assembler.py`).
  - End-to-end multi-part proportional fusion verified: cranial vault, bilateral ears, anterior nasal projection, mandible/chin, and cervical neck.
  - Watertight production asset exported: `docs/dashboard/design_lab/testing/3d_05_unified_compositional_assembler/final_assembled_character_bust.obj` (3,782 vertices, 3,140 faces, zero non-manifold edges).
  - 4-angle visual inspection composite rendered: `bust_assembly_composite_4view.png`.
- **Deliverables Staged in:** `staging/lab_3d_compositional/`

### [PENDING COMMANDER INGEST — 2026-11-12]
- **Sprint Name:** `MINI DESIGN LAB: Scenario LAB-3D-03 — REA Anatomical Scaffold & Boundary Harvester`
- **Origin Chat:** `LAB: REA Asset Harvester`
- **Target Chat:** `1 - COMMANDER`
- **Status:** `READY FOR COMMANDER ARCHIVE`
- **Verification Evidence:**
  - 3/3 tests green in `staging/lab_3d_rea_harvester/tests/` (`test_rea_harvester.py`).
  - Anthropometric human head proportion registry and socket frames codified (`AnatomicalProportionRegistry`).
  - Boundary loop detection and non-rigid socket scale fitting verified (`REAAssetHarvester`).
- **Deliverables Staged in:** `staging/lab_3d_rea_harvester/`

### [PENDING COMMANDER INGEST — 2026-11-12]
- **Sprint Name:** `MINI DESIGN LAB: Procedural Blender Python & VLM Critic (LAB-3D-01)`
- **Origin Chat:** `LAB: 3D Blender VLM Critic`
- **Target Chat:** `1 - COMMANDER`
- **Status:** `READY FOR COMMANDER ARCHIVE`
- **Verification Evidence:**
  - 7/7 tests passing in `staging/lab_3d_blender_vlm/tests/`
  - Closed-loop optimization verified: score advanced from 1.0 -> 7.7 across 4 iterations (`optimization_log.json`)
  - Watertight 3D OBJ asset exported: `docs/dashboard/design_lab/testing/3d_01_procedural_blender_vlm_critic/final_refined_model.obj`
  - Multi-view camera passes generated: `iter_0` through `iter_4` composite 4-view contact sheets
- **Deliverables Staged in:** `staging/lab_3d_blender_vlm/`

### [PENDING COMMANDER INGEST — 2026-11-12]
- **Sprint Name:** `MINI DESIGN LAB: Scenario LAB-3D-02 — Microsoft TRELLIS Structured Latents (SLAT) Pipeline`
- **Origin Chat:** `LAB: TRELLIS SLAT Latents`
- **Target Chat:** `1 - COMMANDER`
- **Status:** `READY FOR COMMANDER ARCHIVE`
- **Verification Evidence:**
  - 3/3 tests green in `staging/lab_3d_trellis_latents/tests/` (`test_trellis_latents.py`).
  - Rectified flow ODE Euler integration (steps=12, guidance=3.5) converting standard normal prior to 1,045 SLAT tokens.
  - Dual decoding verified: 1,045 3D Gaussian Splats + 774 quad boundary surface faces in 0.10s (exceeding sub-15s requirement).
  - Exported production asset: `docs/dashboard/design_lab/testing/3d_02_trellis_structured_latents/model_anatomical_human_ear.obj`.
  - 4-quadrant preview render generated: `proof_model_preview_anatomical_human_ear.png`.
- **Deliverables Staged in:** `staging/lab_3d_trellis_latents/`

### [PENDING COMMANDER INGEST — 2026-11-12]
- **Sprint Name:** `MINI DESIGN LAB: Scenario LAB-3D-04 — Voxel/SDF Octree Boolean Stitcher & Dual Contouring`
- **Origin Chat:** `LAB: Voxel/SDF Stitcher`
- **Target Chat:** `1 - COMMANDER`
- **Status:** `READY FOR COMMANDER ARCHIVE`
- **Verification Evidence:**
  - 3/3 tests green in `staging/lab_3d_sdf_stitcher/tests/` (`test_sdf_stitcher.py`)
  - Smooth polynomial Boolean union ($smin$) verified eliminating non-manifold seams between head and ear volumes
  - Watertight stitched mesh exported: `docs/dashboard/design_lab/testing/3d_04_voxel_sdf_octree_stitcher/proof_head_ear_stitched_assembly.obj`
  - Multi-view visual proof rendered: `proof_stitched_composite_4view.png`
- **Deliverables Staged in:** `staging/lab_3d_sdf_stitcher/`

<!-- 
MINI LAB HAND-BACK TEMPLATE:
### [PENDING COMMANDER INGEST — YYYY-MM-DD]
- **Sprint Name:** `MINI DESIGN LAB: <Feature Name>`
- **Origin Chat:** `LAB: <Feature Name>`
- **Target Chat:** `1 - COMMANDER`
- **Status:** `READY FOR COMMANDER ARCHIVE`
- **Verification Evidence:**
  - Tests passing in `staging/<Feature>/tests/`
  - Zero regression on existing test suite (`pytest`)
  - Proof screenshots and render logs in `docs/dashboard/design_lab/testing/<Feature>/`
  - Updated timeline entry in `docs/dashboard/design_lab.html`
- **Deliverables Staged in:** `staging/<Feature>/`
-->

---

## 0b. Archived Sprint Hand-Backs (Ingested by Commander)
*(Historical archive will be logged here upon Commander ingestion)*

---

## 1. Active Proving Grounds Dossier Index
- **`3d_01_procedural_blender_vlm_critic`**: Headless Blender Python (`bpy`) script generator with multi-view render passes and local VLM (Qwen2.5-VL / MiniCPM-V) hill-climbing critic.
- **`3d_02_trellis_structured_latents`**: Microsoft TRELLIS (SLAT) rectified flow transformer engine for direct zero-shot 3D asset generation into GLB / 3D Gaussian Splats.
- **`3d_03_rea_asset_harvester`**: Reverse Engineering Analyst (REA) pipeline querying open 3D repositories, decomposing baseline meshes into anatomical sub-components (ears, noses, limbs).
- **`3d_04_voxel_sdf_octree_stitcher`**: Hierarchical voxel/SDF boolean fusion engine with OpenVDB/Trimesh dual contouring and quad-dominant remeshing for seamless assembly.
