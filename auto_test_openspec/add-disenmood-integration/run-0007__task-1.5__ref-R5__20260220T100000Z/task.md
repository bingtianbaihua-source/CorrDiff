CHANGE_ID: add-disenmood-integration
TASK_ID: 1.5
REF: R5
RUN_NUMBER: 7 (0007)

Goal
- Sampling stage decodes generated latents:
  - Use `BranchDiffusion.sample()` to generate `z_shared` and `z_pi`.
  - Call `DisentangledVAE.decode(z_shared, z_pi)` to get 3D coordinates + atom types.
  - Reuse `utils.smiles_export.reconstruct_smiles()` to write `outputs/generated_molecules.smi`.
  - Write `outputs/generated_molecules_3d.npz` for 1-to-1 correspondence checks.

How to run
- Bash: `bash run.sh`
- Windows: `run.bat`

