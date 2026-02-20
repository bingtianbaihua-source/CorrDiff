# Task 1.4 (R4): latent decode + SMILES export helper

This bundle validates that `DisentangledVAE.decode()` can decode latent vectors into:

- `xyz` 3D coordinates with shape `(N, 3)`
- `atomic_nums` atom types with shape `(N,)`

It also validates that a non-empty SMILES representation is exported, accepting either:

- a non-empty SMILES string, or
- the documented placeholder string: `SMILES_UNAVAILABLE_IN_TOY_MODE`

Run with `bash run.sh` (or `run.bat` on Windows).

