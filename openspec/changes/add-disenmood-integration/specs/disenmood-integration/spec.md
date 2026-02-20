## ADDED Requirements

### Requirement: VAE Pretraining Entrypoint
The system SHALL provide a dedicated training entrypoint to pretrain the DisentangledVAE on molecule-pocket data and persist model checkpoints.

#### Scenario: VAE pretrain run
- **WHEN** the user runs the VAE pretraining script with a valid config
- **THEN** training completes an epoch and writes a checkpoint to the configured output path

### Requirement: Latent Encoding Path For Diffusion Training
The system SHALL support producing z_shared and {z_pi} latents for the training dataset by offline cache (default) or online encoding (optional), and make them available to BranchDiffusion during training.

#### Scenario: Online encoding in training loop
- **WHEN** latent caching is disabled
- **THEN** the training loop encodes each batch into z_shared and {z_pi} before diffusion loss is computed

### Requirement: BranchDiffusion As Main Training Path
The system SHALL compute diffusion loss in latent space using BranchDiffusion instead of directly adding noise in 3D coordinate space when DisenMoOD mode is enabled.

#### Scenario: DisenMoOD mode enabled
- **WHEN** the training config enables DisenMoOD latent training
- **THEN** the loss is computed from BranchDiffusion on z_shared and {z_pi} and 3D coordinate diffusion is skipped

### Requirement: Latent Decoding To 3D And SMILES
The system SHALL decode sampled latents into 3D ligand coordinates and atomic types required for molecule reconstruction, and provide a helper to export the decoded structure as SMILES (or a documented equivalent representation).

#### Scenario: Sampling pipeline decoding
- **WHEN** sampling produces z_shared and {z_pi}
- **THEN** the decoder returns 3D coordinates and atomic types sufficient for reconstruction, and the pipeline exports a molecular representation

### Requirement: Decoding Output Fields
The decoder output SHALL include at minimum `xyz` (N_atoms x 3) and `atomic_nums` (N_atoms) so that `utils.reconstruct.reconstruct_from_generated` can rebuild an RDKit molecule. Optional fields MAY include `aromatic` and `atom_affinity` when available.

#### Scenario: Minimal reconstruction-compatible decode
- **WHEN** the decoder returns only `xyz` and `atomic_nums`
- **THEN** reconstruction succeeds without requiring optional fields

### Requirement: Two-Stage Training Workflow
The system SHALL document and support a two-stage workflow where DisentangledVAE is trained first, followed by BranchDiffusion training that consumes the pretrained VAE.

#### Scenario: Two-stage execution
- **WHEN** the user follows the two-stage instructions
- **THEN** the diffusion training run loads the pretrained VAE and proceeds without re-training it
