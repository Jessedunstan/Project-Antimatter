# Proving Ground Dossier: Scenario LAB-3D-01
## Procedural Blender Python (`bpy`) & VLM Critic Hill-Climbing Loop

### Objective
Build and verify an autonomous generator-verifier loop inside `staging/3d_blender_vlm/`:
1. Code LLM writes procedural Blender Python scripts (`bpy`) modeling micro-parts (e.g. ear, nose, hand).
2. Headless Blender renders 4-angle views (Front, Side, Top, Isometric).
3. Vision LLM (Qwen2.5-VL / MiniCPM-V or API) inspects renders, scores realism/anatomy (1-10), and produces code diff feedback.
4. If score improves, save checkpoint; if it drops, rollback and re-sample.

### Required Artifacts in This Dossier
- `README.md` (this file)
- `scenario_config.json`
- `proof_render_pass_*.png` (render outputs)
- Entry in `docs/dashboard/design_lab.html` timeline
- Hand-back block in `HANDOFF_STATE.md`
