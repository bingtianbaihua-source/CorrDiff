## Task 1.4 (R4): Latent decode + SMILES helper

This bundle validates a toy latent decode path:

- Input: `inputs/sample_latents.npz` containing `z_shared` and `z_pi_0`
- Script: `scripts/decode_latents.py`
- Output: `outputs/decoded_ligand.npz` containing:
  - `xyz`: `(N, 3)` float32 coordinates
  - `atomic_nums`: `(N,)` int64 atomic numbers
  - `smiles`: SMILES string when optional deps are available; otherwise placeholder

Notes:
- SMILES export is best-effort via `utils.smiles_export.reconstruct_smiles()`, which attempts to use
  `utils.reconstruct.reconstruct_from_generated()` and falls back when optional deps are missing.

