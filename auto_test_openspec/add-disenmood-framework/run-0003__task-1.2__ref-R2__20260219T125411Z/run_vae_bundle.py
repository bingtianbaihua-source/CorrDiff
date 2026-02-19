import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from models.guide_model import DisentangledVAE


def main() -> int:
    root = Path(os.environ["BUNDLE_DIR"])
    inputs_dir = root / "inputs"
    outputs_dir = root / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    batch_size = 4
    n_protein = 6
    n_ligand = 5
    protein_atom_feature_dim = 8
    ligand_atom_feature_dim = 10

    protein_pos = torch.randn(batch_size * n_protein, 3)
    ligand_pos = torch.randn(batch_size * n_ligand, 3)
    protein_atom_feature = torch.randn(batch_size * n_protein, protein_atom_feature_dim)
    ligand_atom_feature = torch.randn(batch_size * n_ligand, ligand_atom_feature_dim)

    batch_protein = torch.arange(batch_size).repeat_interleave(n_protein)
    batch_ligand = torch.arange(batch_size).repeat_interleave(n_ligand)

    mini_batch = {
        "protein_pos": protein_pos,
        "protein_atom_feature": protein_atom_feature,
        "ligand_pos": ligand_pos,
        "ligand_atom_feature": ligand_atom_feature,
        "batch_protein": batch_protein,
        "batch_ligand": batch_ligand,
    }
    torch.save(mini_batch, inputs_dir / "mini_batch.pt")

    property_names = [f"p{i}" for i in range(10)]
    cfg = SimpleNamespace(
        hidden_dim=64,
        num_layers=2,
        edge_feat_dim=4,
        num_r_gaussian=20,
        knn=16,
        r_max=10.0,
        cutoff_mode="knn",
        update_x=False,
        act_fn="silu",
        norm=False,
        z_shared_dim=12,
        z_pi_dim=7,
        property_names=property_names,
        beta_kl=1.0,
        tc_weight=1.0,
        mi_weight=1.0,
    )

    model = DisentangledVAE(
        cfg,
        protein_atom_feature_dim=protein_atom_feature_dim,
        ligand_atom_feature_dim=ligand_atom_feature_dim,
    )
    model.eval()

    with torch.no_grad():
        out = model(
            protein_pos=mini_batch["protein_pos"],
            protein_atom_feature=mini_batch["protein_atom_feature"],
            ligand_pos=mini_batch["ligand_pos"],
            ligand_atom_feature=mini_batch["ligand_atom_feature"],
            batch_protein=mini_batch["batch_protein"],
            batch_ligand=mini_batch["batch_ligand"],
            return_losses=True,
        )

    losses = out["losses"]
    payload = {
        "recon_loss": float(losses["recon_loss"].item()),
        "tc_loss": float(losses["tc_loss"].item()),
        "mi_loss": float(losses["mi_loss"].item()),
        "z_shared_shape": list(out["z_shared"].shape),
        "z_pi_shapes": {k: list(v.shape) for k, v in out["z_pi"].items()},
    }
    (outputs_dir / "vae_losses.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Validate keys + shapes.
    required_keys = {"recon_loss", "tc_loss", "mi_loss", "z_shared_shape", "z_pi_shapes"}
    if not required_keys.issubset(payload.keys()):
        raise KeyError(f"Missing required keys: {sorted(required_keys - set(payload.keys()))}")

    z_shared_shape = payload["z_shared_shape"]
    if len(z_shared_shape) != 2 or z_shared_shape[0] != batch_size or z_shared_shape[1] != cfg.z_shared_dim:
        raise AssertionError(f"Unexpected z_shared shape: {z_shared_shape}, expected [{batch_size}, {cfg.z_shared_dim}]")

    z_pi_shapes = payload["z_pi_shapes"]
    if set(z_pi_shapes.keys()) != set(property_names):
        raise AssertionError(f"Unexpected z_pi keys: {sorted(z_pi_shapes.keys())}")
    for name in property_names:
        shp = z_pi_shapes[name]
        if len(shp) != 2 or shp[0] != batch_size or shp[1] != cfg.z_pi_dim:
            raise AssertionError(f"Unexpected z_pi[{name}] shape: {shp}, expected [{batch_size}, {cfg.z_pi_dim}]")

    if not (payload["recon_loss"] > 0.0):
        raise AssertionError("recon_loss is not positive/non-trivial")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
