# DisenMoOD two-stage workflow (VAE → latent diffusion)

This project supports a **two-stage DisenMoOD workflow**:

- **Stage 1 (VAE pretraining):** train a `DisentangledVAE` encoder/decoder on (toy) protein–ligand batches and write a checkpoint.
- **Stage 2 (BranchDiffusion latent diffusion):** train / run latent-space diffusion using `BranchDiffusion` over the VAE latents. In DisenMoOD mode, the legacy **3D coordinate diffusion path is disabled** (no fallback).

## Stage 1: VAE pretraining

<!-- checklist: stage_1_vae_pretraining -->

Script:
- `scripts/train_vae.py`

Example config:
- `configs/vae_toy_config.yml`

Outputs:
- A VAE checkpoint `.pt` containing `encoder_state_dict` and `decoder_state_dict`.

Example command (CPU):
```bash
python scripts/train_vae.py \
  --config configs/vae_toy_config.yml \
  --output outputs/vae_checkpoint.pt
```

Notes:
- Stage 2 consumes **VAE latents** (`z_shared` and `z_pi_*`) produced by the `DisentangledVAE` encoder. In a full run, you should load the Stage 1 checkpoint weights before encoding latents.

## Stage 2: BranchDiffusion latent diffusion training (DisenMoOD mode)

<!-- checklist: stage_2_latent_diffusion -->

Script:
- `scripts/train_diffusion_joint.py`

Example config:
- `configs/train_config_disenmood.yml`

Key behavior:
- Uses **latent diffusion** (BranchDiffusion) over `(z_shared, z_pi_list)`.
- **Skips 3D coordinate diffusion** when `disenmood_mode: true`.

Example command (CPU):
```bash
python scripts/train_diffusion_joint.py \
  --config configs/train_config_disenmood.yml \
  --device cpu
```

## DisenMoOD switch and legacy 3D diffusion gate

<!-- checklist: disenmood_mode_switch -->

Enable DisenMoOD mode by setting this at the YAML top level:

```yml
disenmood_mode: true
```

When `disenmood_mode: true`, `scripts/train_diffusion_joint.py` runs the DisenMoOD latent-diffusion branch and **returns early** without entering the legacy training code path.

<!-- checklist: legacy_3d_path_disabled -->

**Legacy 3D diffusion path status in DisenMoOD mode: DISABLED**
- There is **no fallback route** to coordinate-space diffusion when `disenmood_mode: true`.
- Even if legacy 3D-diffusion settings are present in a config, the DisenMoOD branch is selected and the legacy pipeline is not executed.

## Checklist keys (for CLI verification)

The CLI doc-verifier for task 1.6 checks for these keys:
- `stage_1_vae_pretraining`
- `stage_2_latent_diffusion`
- `disenmood_mode_switch`
- `legacy_3d_path_disabled`

