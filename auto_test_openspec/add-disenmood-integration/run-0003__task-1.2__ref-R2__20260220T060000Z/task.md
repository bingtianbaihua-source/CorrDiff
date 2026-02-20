# Task 1.2 (R2) Validation Bundle — Latent Encoding Modes

## Goal
Provide a latent encoding path for BranchDiffusion training supporting:
- **offline** latent cache (**default**)
- **online** per-batch encoding (**optional switch**)

## Pass/Fail Criteria
PASS if the CLI run:
1) Prints the selected mode (e.g. `Latent encoding mode: offline`)
2) Writes `outputs/latent_batch.npz`
3) `latent_batch.npz` contains `z_shared` and at least one `z_pi_*`
4) Shapes match the config:  
   - `z_shared.shape == (batch_size, z_shared_dim)`  
   - `z_pi_0.shape == (batch_size, z_pi_dim)`

## How to run
- macOS/Linux: `bash run.sh`
- Windows: `run.bat`

## Notes
- To try online mode, set `latent_encoding_mode: online` in `inputs/latent_mode_config.yml`.

