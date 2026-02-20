CHANGE_ID: add-disenmood-integration
TASK_ID: 1.6
REF: R6
RUN_NUMBER: 8 (0008)

Goal
- Document the DisenMoOD two-stage workflow:
  - Stage 1: `DisentangledVAE` pretraining (`scripts/train_vae.py`)
  - Stage 2: BranchDiffusion latent diffusion training (`scripts/train_diffusion_joint.py` with `disenmood_mode: true`)
- Ensure documentation states legacy 3D diffusion path is disabled in DisenMoOD mode (no fallback route).

How to run
- Bash: `bash run.sh`
- Windows: `run.bat`

