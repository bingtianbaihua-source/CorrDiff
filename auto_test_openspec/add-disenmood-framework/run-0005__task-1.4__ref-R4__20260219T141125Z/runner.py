import json
import os
from pathlib import Path

import numpy as np
import torch


def _make_synth_minibatch(*, batch_size: int, protein_dim: int, ligand_dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    protein_pos = []
    protein_feat = []
    batch_protein = []

    ligand_pos = []
    ligand_feat = []
    batch_ligand = []

    for b in range(batch_size):
        n_prot = 6 + b
        n_lig = 4 + b

        protein_pos.append(torch.randn(n_prot, 3, generator=g))
        protein_feat.append(torch.randn(n_prot, protein_dim, generator=g))
        batch_protein.append(torch.full((n_prot,), b, dtype=torch.long))

        ligand_pos.append(torch.randn(n_lig, 3, generator=g))
        ligand_feat.append(torch.randn(n_lig, ligand_dim, generator=g))
        batch_ligand.append(torch.full((n_lig,), b, dtype=torch.long))

    return {
        "protein_pos": torch.cat(protein_pos, dim=0),
        "protein_atom_feature": torch.cat(protein_feat, dim=0),
        "batch_protein": torch.cat(batch_protein, dim=0),
        "ligand_pos": torch.cat(ligand_pos, dim=0),
        "ligand_atom_feature": torch.cat(ligand_feat, dim=0),
        "batch_ligand": torch.cat(batch_ligand, dim=0),
    }


def main() -> int:
    root = Path(os.environ["BUNDLE_DIR"])
    inputs_dir = root / "inputs"
    outputs_dir = root / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    mini_batch_path = inputs_dir / "mini_batch.pt"
    if not mini_batch_path.exists():
        batch = _make_synth_minibatch(batch_size=4, protein_dim=27, ligand_dim=13, seed=0)
        torch.save(batch, mini_batch_path)

    batch = torch.load(mini_batch_path, map_location="cpu")

    from models.egnn import CorrelationMatrixModule
    from models.guide_model import DisentangledVAE

    # Pool "graph features" from molecule/pocket graphs (as requested): mean over nodes per graph.
    batch_size = int(batch["batch_protein"].max().item()) + 1
    mol_pooled = []
    pocket_pooled = []
    for b in range(batch_size):
        mol_pooled.append(batch["ligand_atom_feature"][batch["batch_ligand"] == b].float().mean(dim=0))
        pocket_pooled.append(batch["protein_atom_feature"][batch["batch_protein"] == b].float().mean(dim=0))
    mol_pooled = torch.stack(mol_pooled, dim=0)
    pocket_pooled = torch.stack(pocket_pooled, dim=0)

    n_props = 10
    corr = CorrelationMatrixModule(mol_dim=13, pocket_dim=27, n_props=n_props, hidden_dim=64, num_layers=2).eval()
    with torch.no_grad():
        c_t = corr(mol_pooled=mol_pooled, pocket_pooled=pocket_pooled)

    c = c_t.detach().cpu().numpy()
    np.save(outputs_dir / "correlation_matrix.npy", c)

    symmetric = np.allclose(c, np.transpose(c, (0, 2, 1)), atol=1e-5)
    diag = np.diagonal(c, axis1=1, axis2=2)
    diagonal_ones = np.allclose(diag, 1.0, atol=1e-5)

    with torch.no_grad():
        mask_05 = corr.get_branch_mask(c_t, threshold=0.5)
        mask_09 = corr.get_branch_mask(c_t, threshold=0.9)

    mask_nnz = int(mask_05.sum().item())
    mask_nnz_hi = int(mask_09.sum().item())

    g = torch.Generator().manual_seed(0)
    mu_stack = torch.randn(batch_size, n_props, 7, generator=g)
    mu_mixed_05, stats_05 = DisentangledVAE.apply_branch_interaction(mu_stack, branch_mask=mask_05, strength=1.0)
    mu_mixed_09, stats_09 = DisentangledVAE.apply_branch_interaction(mu_stack, branch_mask=mask_09, strength=1.0)

    mean_abs_delta_mu = float(stats_05["mean_abs_delta_mu"].item())
    mean_abs_delta_mu_hi = float(stats_09["mean_abs_delta_mu"].item())

    out_json = {
        "symmetric": bool(symmetric),
        "diagonal_ones": bool(diagonal_ones),
        "mask_nnz": mask_nnz,
        "threshold": 0.5,
        "interaction_strength": 1.0,
        "mean_abs_delta_mu": mean_abs_delta_mu,
        "mask_nnz_threshold_0p9": mask_nnz_hi,
        "mean_abs_delta_mu_threshold_0p9": mean_abs_delta_mu_hi,
    }
    (outputs_dir / "branch_mask_stats.json").write_text(json.dumps(out_json, indent=2) + "\n")

    if not (symmetric and diagonal_ones):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
