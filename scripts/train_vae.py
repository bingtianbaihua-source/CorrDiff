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
class TrainConfig:
    seed: int
    device: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float


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


def _parse_train_config(cfg: dict[str, Any]) -> TrainConfig:
    train = cfg.get("train", {}) or {}
    if not isinstance(train, dict):
        raise TypeError("config.train must be a mapping")
    device = str(cfg.get("device", train.get("device", "cpu")))
    return TrainConfig(
        seed=int(train.get("seed", 0)),
        device=device,
        epochs=int(train.get("epochs", 1)),
        batch_size=int(train.get("batch_size", 2)),
        lr=float(train.get("lr", train.get("learning_rate", 1e-3))),
        weight_decay=float(train.get("weight_decay", 0.0)),
    )


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def _split_encoder_decoder_state_dict(*, model_state_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder_prefixes = (
        "protein_atom_emb.",
        "ligand_atom_emb.",
        "encoder_gnn.",
        "shared_head.",
        "pi_heads.",
        "corr_module.",
    )
    encoder_state = {k: v for k, v in model_state_dict.items() if k.startswith(encoder_prefixes)}
    decoder_state = {k[len("decoder.") :]: v for k, v in model_state_dict.items() if k.startswith("decoder.")}
    return encoder_state, decoder_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone DisentangledVAE pretraining (toy dataset).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output", type=str, required=True, help="Path to .pt checkpoint to write.")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    toy_cfg = _parse_toy_data_config(cfg)
    train_cfg = _parse_train_config(cfg)
    model_cfg = cfg.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        raise TypeError("config.model must be a mapping")

    import torch
    from torch.utils.data import DataLoader

    from models.guide_model import DisentangledVAE

    _set_seed(train_cfg.seed)

    device = torch.device(train_cfg.device)
    dataset = ToyProteinLigandDataset(cfg=toy_cfg, seed=train_cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        drop_last=False,
    )

    model = DisentangledVAE(
        model_cfg,
        protein_atom_feature_dim=toy_cfg.protein_atom_feature_dim,
        ligand_atom_feature_dim=toy_cfg.ligand_atom_feature_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    model.train()
    for epoch in range(int(train_cfg.epochs)):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["protein_pos"].to(device),
                batch["protein_feat"].to(device),
                batch["ligand_pos"].to(device),
                batch["ligand_feat"].to(device),
                batch["batch_protein"].to(device),
                batch["batch_ligand"].to(device),
                return_losses=True,
            )
            losses = out.get("losses", {})
            if not isinstance(losses, dict) or not losses:
                raise RuntimeError("Model did not return a non-empty losses dict")
            loss = sum(v for v in losses.values())
            loss.backward()
            optimizer.step()

    model_state = model.state_dict()
    encoder_state, decoder_state = _split_encoder_decoder_state_dict(model_state_dict=model_state)
    if len(encoder_state) == 0:
        raise RuntimeError("No encoder weights found to save (expected encoder_* modules in state_dict).")
    if len(decoder_state) == 0:
        raise RuntimeError("No decoder weights found to save (expected decoder.* keys in state_dict).")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {
            "config": cfg,
            "model_state_dict": model_state,
            "encoder_state_dict": encoder_state,
            "decoder_state_dict": decoder_state,
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
