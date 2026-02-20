# Decoder Output Contract (Reconstruction Compatibility)

This document defines the **minimum output contract** for decoded ligand samples that are intended to be reconstructed into an RDKit/OpenBabel molecule.

The contract is aligned with the reconstruction entry function:

- `utils.reconstruct.reconstruct_from_generated(xyz, atomic_nums, aromatic=None, atom_affinity=[], basic_mode=True)`

## Scope

This contract applies to:

- The output of `models.guide_model.DisentangledVAE.decode(...)` (returned as `(xyz, atomic_nums)`), and
- Any *sampling output* (e.g., `.npz` or dict) that feeds into reconstruction utilities.

## Required fields

Sampling/decoded outputs **MUST** provide:

1. `xyz`
   - Meaning: 3D coordinates for ligand atoms
   - Shape:
     - single molecule: `(N, 3)`
     - batched: `(B, N, 3)`
   - Dtype:
     - in torch: `float32`
     - when serialized (NumPy): `float32` recommended

2. `atomic_nums`
   - Meaning: atomic numbers (e.g., 6 for carbon, 8 for oxygen)
   - Shape:
     - single molecule: `(N,)`
     - batched: `(B, N)`
   - Dtype:
     - in torch: `int64` (`torch.long`)
     - when serialized (NumPy): `int64` recommended

## Optional fields

Sampling/decoded outputs **MAY** also provide:

1. `aromatic`
   - Meaning: per-atom aromaticity flags/indicators used by reconstruction in non-basic mode
   - Shape: `(N,)` (or `(B, N)` if batched)
   - Dtype: `bool`
   - Reconstruction mapping: passed as `aromatic=` to `reconstruct_from_generated(...)`

2. `atom_affinity`
   - Meaning: per-atom affinity / weighting scores used by reconstruction
   - Shape: `(N,)` (or `(B, N)` if batched)
   - Dtype: `float32`
   - Reconstruction mapping: passed as `atom_affinity=` to `reconstruct_from_generated(...)`

## Reconstruction call mapping

For a single molecule, reconstruction should be called as:

```python
from utils.reconstruct import reconstruct_from_generated

mol = reconstruct_from_generated(
    xyz=xyz,
    atomic_nums=atomic_nums,
    aromatic=aromatic,            # optional
    atom_affinity=atom_affinity,  # optional
)
```

