#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, Sequence


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping, got: {type(data).__name__}")
    return data


def _cfg_get(mapping: Dict[str, Any], key: str, default: Any) -> Any:
    v = mapping.get(key, default)
    return default if v is None else v


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample DisenMoOD latents with BranchDiffusion and decode into 3D + SMILES (toy)."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config with model.* and sampling.*")
    parser.add_argument("--out-smi", type=str, required=True, help="Output .smi (one SMILES per line)")
    parser.add_argument("--out-3d-npz", type=str, required=True, help="Output .npz (xyz, atomic_nums)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu)")
    args = parser.parse_args(argv)

    cfg = _load_yaml(Path(args.config).expanduser().resolve())
    model_cfg = cfg.get("model", {}) or {}
    sampling_cfg = cfg.get("sampling", {}) or {}
    if not isinstance(model_cfg, dict):
        raise TypeError("config.model must be a mapping")
    if not isinstance(sampling_cfg, dict):
        raise TypeError("config.sampling must be a mapping")

    import numpy as np
    import torch

    from models.guide_model import DisentangledVAE
    from models.molopt_score_model import BranchDiffusion
    from utils.smiles_export import reconstruct_smiles

    seed = int(_cfg_get(sampling_cfg, "seed", 0))
    num_molecules = int(_cfg_get(sampling_cfg, "num_molecules", 3))
    if num_molecules < 1:
        raise ValueError(f"sampling.num_molecules must be >= 1, got {num_molecules}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(str(args.device))

    model_cfg = dict(model_cfg)
    model_cfg.setdefault("hidden_dim", 16)
    model_cfg.setdefault("z_shared_dim", 32)
    model_cfg.setdefault("z_pi_dim", 16)
    model_cfg.setdefault("property_names", ["p0"])
    model_cfg.setdefault("decoder_num_atoms", 8)

    vae = DisentangledVAE(model_cfg, protein_atom_feature_dim=4, ligand_atom_feature_dim=4).to(device)
    vae.eval()

    bd = BranchDiffusion(
        z_shared_dim=int(model_cfg["z_shared_dim"]),
        z_pi_dim=int(model_cfg["z_pi_dim"]),
        property_names=list(model_cfg["property_names"]),
        num_steps=int(_cfg_get(sampling_cfg, "diffusion_num_steps", 20)),
        time_emb_dim=int(_cfg_get(sampling_cfg, "diffusion_time_emb_dim", 64)),
        hidden_dim=int(_cfg_get(sampling_cfg, "diffusion_hidden_dim", 64)),
        num_layers=int(_cfg_get(sampling_cfg, "diffusion_num_layers", 2)),
        sigma_min=float(_cfg_get(sampling_cfg, "sigma_min", 0.01)),
        sigma_max=float(_cfg_get(sampling_cfg, "sigma_max", 1.0)),
    ).to(device)
    bd.eval()

    with torch.no_grad():
        sample_out = bd.sample(batch_size=num_molecules, device=device)
        z_shared = sample_out["z_shared"]
        z_pi = sample_out["z_pi"]
        xyz, atomic_nums = vae.decode(z_shared, z_pi)

    out_smi = Path(args.out_smi).expanduser().resolve()
    out_npz = Path(args.out_3d_npz).expanduser().resolve()
    _ensure_parent(out_smi)
    _ensure_parent(out_npz)

    xyz_np = xyz.detach().cpu().to(dtype=torch.float32).numpy()
    atomic_np = atomic_nums.detach().cpu().to(dtype=torch.long).numpy()
    np.savez(out_npz, xyz=xyz_np, atomic_nums=atomic_np)

    with out_smi.open("w", encoding="utf-8") as f:
        for i in range(num_molecules):
            result = reconstruct_smiles(xyz_np[i].tolist(), atomic_np[i].tolist(), basic_mode=True)
            f.write(str(result.smiles).strip() + "\n")

    n_lines = _count_nonempty_lines(out_smi)
    print(f"Wrote: {out_smi} ({n_lines} non-empty lines)")
    print(f"Wrote: {out_npz} (xyz={xyz_np.shape}, atomic_nums={atomic_np.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
