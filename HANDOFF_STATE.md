# Project Antimatter — Master Handoff & Sprint Communication Bus

**Current Date / Session:** Antimatter Research & Design Lab Initialized  
**Architecture Baseline:** Python 3.11+ / FastAPI / Tauri v2 / Multi-Backend Inference (Ollama/vLLM)  
**Main Design Lab Hub:** `docs/dashboard/design_lab.html`  
**Staging Area:** `staging/<Feature>/` (Strict isolation — never modifies mainline `antimatter/` directly)  

---

## 0. Active Sprint Hand-Back Queue (File-Backed Communication Bus)
*(Queue clean — ready for Mini Design Lab side-chats to file staging proofs)*

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
