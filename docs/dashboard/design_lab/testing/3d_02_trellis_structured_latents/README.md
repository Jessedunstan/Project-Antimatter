# Proving Ground Dossier: Scenario LAB-3D-02
## Microsoft TRELLIS Structured Latents (SLAT) Local Weights Pipeline

### Objective
Deploy and verify the TRELLIS 3D foundation model inside `staging/3d_trellis_latents/`:
1. Load TRELLIS weights via PyTorch/CUDA in isolated staging environment.
2. Given image or text prompt for individual anatomical parts, generate Structured Latents (SLAT).
3. Decode SLAT into watertight `.glb` meshes with full PBR textures and 3D Gaussian Splats.
4. Verify sub-15-second generation speed and export manifold mesh for assembly.

### Required Artifacts in This Dossier
- `README.md` (this file)
- `scenario_config.json`
- `proof_model_preview_*.png`
- Hand-back block in `HANDOFF_STATE.md`
