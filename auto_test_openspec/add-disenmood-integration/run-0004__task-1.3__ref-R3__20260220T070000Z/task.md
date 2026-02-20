CHANGE_ID: add-disenmood-integration
TASK_ID: 1.3
REF: R3
RUN_NUMBER: 4 (0004)

Goal
- Exercise DisenMoOD mode in `scripts/train_diffusion_joint.py`:
  - Use `BranchDiffusion` to compute latent-space diffusion loss from `DisentangledVAE` latents.
  - Skip any 3D coordinate diffusion/noise path.
  - Write `outputs/train_step_metrics.json` with `latent_diffusion_loss`.

How to run
- Bash: `bash run.sh`
- Windows: `run.bat`

