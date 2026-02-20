# Task 1.1 (R1): DisentangledVAE pretraining entrypoint

This bundle validates the VAE pretraining entrypoint via a toy dataset.

## What it does
- Runs `scripts/train_vae.py` for 1 epoch on a tiny synthetic protein/ligand dataset
- Writes a checkpoint containing `encoder_state_dict` and `decoder_state_dict`

## How to run
- macOS/Linux: `bash run.sh`
- Windows: `run.bat`

