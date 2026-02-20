# Task 1.1 Validation Bundle (R1)

This bundle validates the DisentangledVAE pretraining entrypoint.

## Pass criteria
- `scripts/train_vae.py` runs with the provided toy config and writes `outputs/vae_checkpoint.pt`.
- The checkpoint is a `.pt` file that contains non-empty `encoder_state_dict` and `decoder_state_dict`.

## How to run
- macOS/Linux: run `run.sh`
- Windows: run `run.bat`

