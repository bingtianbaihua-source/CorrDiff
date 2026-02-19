## Task 1.2 (R2) — Disentangled VAE bundle

### How to run (CLI)
- macOS/Linux: `bash run.sh`
- Windows: `run.bat`

### What it does
- Generates a small synthetic Protein/Ligand mini-batch and saves it to `inputs/mini_batch.pt`.
- Runs `models/guide_model.py:DisentangledVAE` forward to produce `z_shared` and per-property `z_pi`.
- Computes and writes `outputs/vae_losses.json` containing:
  - `recon_loss`, `tc_loss`, `mi_loss`
  - `z_shared_shape`, `z_pi_shapes`

### Success criteria (machine-checkable)
- `run.sh` / `run.bat` exits with code 0.
- `outputs/vae_losses.json` contains keys: `recon_loss`, `tc_loss`, `mi_loss`, `z_shared_shape`, `z_pi_shapes`.
- `z_shared_shape[-1] == z_shared_dim` and each `z_pi_shapes[prop][-1] == z_pi_dim` as configured in `run_vae_bundle.py`.
