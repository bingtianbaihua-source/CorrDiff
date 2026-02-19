# Task 1.3 (Ref R3) — Backbone/Branch Latent Diffusion (CLI Bundle)

## How to run (macOS/Linux)

From the repo root:

```bash
PYTHONPATH="$PWD" bash auto_test_openspec/add-disenmood-framework/run-0004__task-1.3__ref-R3__20260219T134302Z/run.sh
```

This writes:
- `inputs/diffusion_config.json`
- `outputs/sample_latents.npz`

## How to run (Windows)

From the repo root (cmd.exe):

```bat
set PYTHONPATH=%CD%
call auto_test_openspec\add-disenmood-framework\run-0004__task-1.3__ref-R3__20260219T134302Z\run.bat
```

## Pass/fail criteria (checked by scripts)

- `outputs/sample_latents.npz` contains key `z_shared` and one key per property name in `property_names`.
- `z_shared` has shape `[batch_size, z_shared_dim]`.
- Each property latent has shape `[batch_size, z_pi_dim]`.

