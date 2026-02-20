import argparse
import os
from dataclasses import dataclass
from typing import Any

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping, got: {type(data).__name__}")
    return data


@dataclass(frozen=True)
class ToyDataConfig:
    num_samples: int
    num_protein_atoms: int
    num_ligand_atoms: int
    protein_atom_feature_dim: int
    ligand_atom_feature_dim: int


@dataclass(frozen=True)
class EncodeConfig:
    device: str
    batch_size: int
    mode: str
    cache_path: str | None


def _parse_toy_data_config(cfg: dict[str, Any]) -> ToyDataConfig:
    data = cfg.get("data", {}) or {}
    if not isinstance(data, dict):
        raise TypeError("config.data must be a mapping")

    def _gi(key: str, default: int) -> int:
        v = data.get(key, default)
        return int(v)

    return ToyDataConfig(
        num_samples=_gi("num_samples", 8),
        num_protein_atoms=_gi("num_protein_atoms", 3),
        num_ligand_atoms=_gi("num_ligand_atoms", 3),
        protein_atom_feature_dim=_gi("protein_atom_feature_dim", 4),
        ligand_atom_feature_dim=_gi("ligand_atom_feature_dim", 4),
    )


def _parse_encode_config(cfg: dict[str, Any]) -> EncodeConfig:
    device = str(cfg.get("device", "cpu"))
    mode = str(cfg.get("latent_encoding_mode", "offline"))
    enc = cfg.get("encode", {}) or {}
    if not isinstance(enc, dict):
        raise TypeError("config.encode must be a mapping")

    cache_path = enc.get("cache_path", None)
    if cache_path is not None:
        cache_path = str(cache_path)
    return EncodeConfig(
        device=device,
        batch_size=int(enc.get("batch_size", 2)),
        mode=mode,
        cache_path=cache_path,
    )


class ToyProteinLigandDataset:
    def __init__(self, *, cfg: ToyDataConfig, seed: int):
        self._cfg = cfg
        self._seed = int(seed)

    def __len__(self) -> int:
        return int(self._cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np

        rng = np.random.default_rng(self._seed + int(idx))
        protein_pos = rng.normal(size=(self._cfg.num_protein_atoms, 3)).astype("float32")
        ligand_pos = rng.normal(size=(self._cfg.num_ligand_atoms, 3)).astype("float32")
        protein_feat = rng.normal(size=(self._cfg.num_protein_atoms, self._cfg.protein_atom_feature_dim)).astype(
            "float32"
        )
        ligand_feat = rng.normal(size=(self._cfg.num_ligand_atoms, self._cfg.ligand_atom_feature_dim)).astype(
            "float32"
        )
        return {
            "protein_pos": protein_pos,
            "protein_feat": protein_feat,
            "ligand_pos": ligand_pos,
            "ligand_feat": ligand_feat,
        }


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    protein_pos_list = []
    protein_feat_list = []
    ligand_pos_list = []
    ligand_feat_list = []
    batch_protein_list = []
    batch_ligand_list = []

    for i, s in enumerate(samples):
        protein_pos = torch.as_tensor(s["protein_pos"], dtype=torch.float32)
        protein_feat = torch.as_tensor(s["protein_feat"], dtype=torch.float32)
        ligand_pos = torch.as_tensor(s["ligand_pos"], dtype=torch.float32)
        ligand_feat = torch.as_tensor(s["ligand_feat"], dtype=torch.float32)

        protein_pos_list.append(protein_pos)
        protein_feat_list.append(protein_feat)
        ligand_pos_list.append(ligand_pos)
        ligand_feat_list.append(ligand_feat)

        batch_protein_list.append(torch.full((protein_pos.shape[0],), i, dtype=torch.long))
        batch_ligand_list.append(torch.full((ligand_pos.shape[0],), i, dtype=torch.long))

    return {
        "protein_pos": torch.cat(protein_pos_list, dim=0),
        "protein_feat": torch.cat(protein_feat_list, dim=0),
        "ligand_pos": torch.cat(ligand_pos_list, dim=0),
        "ligand_feat": torch.cat(ligand_feat_list, dim=0),
        "batch_protein": torch.cat(batch_protein_list, dim=0),
        "batch_ligand": torch.cat(batch_ligand_list, dim=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode toy batches into DisentangledVAE latents (offline cache / online encode).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output", type=str, required=True, help="Path to .npz to write (latent_batch.npz).")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    toy_cfg = _parse_toy_data_config(cfg)
    enc_cfg = _parse_encode_config(cfg)
    model_cfg = cfg.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        raise TypeError("config.model must be a mapping")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from models.guide_model import DisentangledVAE
    from utils.latent_cache import encode_batch_latents, get_or_build_latent_cache_npz

    seed = int(cfg.get("seed", cfg.get("train", {}).get("seed", 0)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(enc_cfg.device)
    dataset = ToyProteinLigandDataset(cfg=toy_cfg, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=enc_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
        drop_last=False,
    )

    vae = DisentangledVAE(
        model_cfg,
        protein_atom_feature_dim=toy_cfg.protein_atom_feature_dim,
        ligand_atom_feature_dim=toy_cfg.ligand_atom_feature_dim,
    ).to(device)
    property_names = list(getattr(vae, "property_names", []))
    if not property_names:
        raise RuntimeError("VAE has empty property_names; cannot produce z_pi")

    mode_norm = str(enc_cfg.mode).strip().lower()
    if mode_norm not in ("offline", "online"):
        raise ValueError(f"latent_encoding_mode must be 'offline' or 'online', got: {enc_cfg.mode!r}")
    print(f"Latent encoding mode: {mode_norm}")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cache_path = enc_cfg.cache_path
    if cache_path is None:
        cache_path = os.path.join(output_dir or ".", "latent_cache.npz")

    if mode_norm == "offline":
        cache = get_or_build_latent_cache_npz(
            mode="offline",
            cache_path=cache_path,
            vae=vae,
            loader=loader,
            property_names=property_names,
            device=device,
        )
        batch_n = min(enc_cfg.batch_size, int(cache.z_shared.shape[0]))
        out = {"z_shared": cache.z_shared[:batch_n].detach().cpu().numpy()}
        for i, t in enumerate(cache.z_pi):
            out[f"z_pi_{i}"] = t[:batch_n].detach().cpu().numpy()
        out["property_names"] = np.asarray(list(cache.property_names), dtype=object)
        np.savez(args.output, **out)
        return 0

    # online: encode first batch only
    first_batch = next(iter(loader))
    with torch.no_grad():
        vae.eval()
        z_shared, z_pi_list = encode_batch_latents(
            vae=vae, batch=first_batch, property_names=property_names, device=device
        )
    batch_n = int(z_shared.shape[0])
    _ = batch_n

    out = {"z_shared": z_shared.detach().cpu().numpy()}
    for i, t in enumerate(z_pi_list):
        out[f"z_pi_{i}"] = t.detach().cpu().numpy()
    out["property_names"] = np.asarray(list(property_names), dtype=object)
    np.savez(args.output, **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
