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


def _is_real_mode(cfg: dict[str, Any]) -> bool:
    """真实数据模式：data.path 存在且非空。"""
    return bool((cfg.get("data") or {}).get("path"))


# ── 玩具模式 ─────────────────────────────────────────────────────────────────

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


# ── 真实数据模式 ──────────────────────────────────────────────────────────────

def _lmdb_worker_init_fn(worker_id: int) -> None:
    """Close the LMDB handle inherited from the main process after fork.

    LMDB is not fork-safe: if the main process opens a DB handle (e.g. via
    __len__) before DataLoader forks workers, each worker inherits the same
    file descriptor and crashes with a segmentation fault.  This init function
    closes the inherited handle so each worker reopens it lazily on first
    __getitem__ access.
    """
    import torch.utils.data
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Traverse Subset / ConcatDataset wrappers used by torch_geometric
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "db") and dataset.db is not None:
        try:
            dataset.db.close()
        except Exception:
            pass
        dataset.db = None
    if hasattr(dataset, "keys"):
        dataset.keys = None


def _build_real_loaders(cfg: dict[str, Any], train_cfg: TrainConfig):
    """
    构建真实数据的 train / val DataLoader。
    返回 (train_loader, val_loader, protein_feat_dim, ligand_vocab_size)。
    """
    import torch
    from easydict import EasyDict
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Compose

    import utils.transforms as trans
    from datasets import get_dataset
    from datasets.pl_data import FOLLOW_BATCH

    data_cfg = EasyDict(cfg["data"])
    transform_cfg = data_cfg.get("transform", EasyDict())

    ligand_atom_mode = str(transform_cfg.get("ligand_atom_mode", "add_aromatic"))
    random_rot = bool(transform_cfg.get("random_rot", False))

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
        trans.NormalizeVina(data_cfg.name),
    ]
    if random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    dataset, subsets = get_dataset(config=data_cfg, transform=transform)

    if data_cfg.name in ("pl", "pl_chem"):
        train_set, val_set = subsets["train"], subsets["test"]
    elif data_cfg.name == "pdbbind":
        train_set, val_set = subsets["train"], subsets["val"]
    else:
        raise ValueError(f"Unknown dataset: {data_cfg.name}")

    num_workers = int((cfg.get("train") or {}).get("num_workers", 4))
    # rmsd / pk / rmsd<2 are affinity-injection metadata not used by the VAE;
    # exclude them so batches with missing fields collate without error.
    collate_excl = ["ligand_nbh_list", "rmsd", "pk", "rmsd<2"]

    worker_init = _lmdb_worker_init_fn if num_workers > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_excl,
        drop_last=True,        # 保证 batch_size>=2，VAE 协方差惩罚才有意义
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_excl,
        worker_init_fn=worker_init,
    )

    protein_feat_dim: int = protein_featurizer.feature_dim
    ligand_vocab_size: int = ligand_featurizer.feature_dim
    return train_loader, val_loader, protein_feat_dim, ligand_vocab_size


def _real_batch_to_vae_inputs(batch, device, ligand_vocab_size: int) -> dict[str, Any]:
    """将 torch_geometric batch 转为 VAE forward 所需的关键字参数。"""
    import torch
    import torch.nn.functional as F

    lig_feat = F.one_hot(batch.ligand_atom_feature_full.long(), num_classes=ligand_vocab_size).float()
    return dict(
        protein_pos=batch.protein_pos.to(device),
        protein_atom_feature=batch.protein_atom_feature.float().to(device),
        ligand_pos=batch.ligand_pos.to(device),
        ligand_atom_feature=lig_feat.to(device),
        batch_protein=batch.protein_element_batch.to(device),
        batch_ligand=batch.ligand_element_batch.to(device),
    )


# ── 公共工具 ─────────────────────────────────────────────────────────────────

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
    decoder_state = {k[len("decoder."):]: v for k, v in model_state_dict.items() if k.startswith("decoder.")}
    return encoder_state, decoder_state


def _save_checkpoint(*, model, cfg, epoch, val_loss, path: str) -> None:
    import torch
    model_state = model.state_dict()
    encoder_state, decoder_state = _split_encoder_decoder_state_dict(model_state_dict=model_state)
    torch.save({
        "config": cfg,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model_state,
        "encoder_state_dict": encoder_state,
        "decoder_state_dict": decoder_state,
    }, path)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DisentangledVAE pretraining — toy or real data."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for the final .pt checkpoint.")
    parser.add_argument("--logdir", type=str, default="",
                        help="Log / intermediate-checkpoint directory (real mode only).")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Resume from an existing checkpoint (real mode only).")
    parser.add_argument("--val-freq", type=int, default=1,
                        help="Validation frequency in epochs (real mode, default: every epoch).")
    parser.add_argument("--save-freq", type=int, default=10,
                        help="Periodic checkpoint saving frequency in epochs (real mode).")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    train_cfg = _parse_train_config(cfg)
    model_cfg = cfg.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        raise TypeError("config.model must be a mapping")

    import torch
    from models.guide_model import DisentangledVAE

    _set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    # ── 分支：玩具模式 vs. 真实数据模式 ──────────────────────────────────────
    if _is_real_mode(cfg):
        return _main_real(args, cfg, train_cfg, model_cfg, device)
    else:
        return _main_toy(args, cfg, train_cfg, model_cfg, device)


# ── 玩具模式主函数 ────────────────────────────────────────────────────────────

def _main_toy(args, cfg, train_cfg, model_cfg, device) -> int:
    import torch
    from torch.utils.data import DataLoader
    from models.guide_model import DisentangledVAE

    toy_cfg = _parse_toy_data_config(cfg)
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

    _save_checkpoint(
        model=model, cfg=cfg, epoch=int(train_cfg.epochs) - 1,
        val_loss=None, path=args.output,
    )
    return 0


# ── 真实数据模式主函数 ────────────────────────────────────────────────────────

def _main_real(args, cfg, train_cfg, model_cfg, device) -> int:
    import shutil
    import torch
    import torch.utils.tensorboard
    from torch.nn.utils import clip_grad_norm_
    from models.guide_model import DisentangledVAE
    import utils.misc as misc

    # ── 日志目录 ──────────────────────────────────────────────────────────────
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind(".")]
    if args.logdir:
        log_dir = misc.get_new_log_dir(args.logdir, prefix="vae_" + config_name)
    else:
        log_dir = misc.get_new_log_dir("logs", prefix="vae_" + config_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger("train_vae", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(f"Config: {args.config}")
    logger.info(f"Log dir: {log_dir}")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    train_loader, val_loader, protein_feat_dim, ligand_vocab_size = _build_real_loaders(cfg, train_cfg)
    train_n = len(train_loader.dataset)
    val_n = len(val_loader.dataset)
    logger.info(
        f"  protein_feat_dim={protein_feat_dim}  ligand_vocab_size={ligand_vocab_size}"
    )
    logger.info(f"  train={train_n}  val={val_n}")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    logger.info("Building DisentangledVAE...")
    model = DisentangledVAE(
        model_cfg,
        protein_atom_feature_dim=protein_feat_dim,
        ligand_atom_feature_dim=ligand_vocab_size,
    ).to(device)
    logger.info(
        f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M"
    )

    max_grad_norm = float((cfg.get("train") or {}).get("max_grad_norm", 1.0))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── 断点续训 ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss: float | None = None
    if args.ckpt:
        logger.info(f"Resuming from checkpoint: {args.ckpt}")
        ckpt_data = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        start_epoch = int(ckpt_data.get("epoch", 0)) + 1
        best_val_loss = ckpt_data.get("val_loss")
        logger.info(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss}")

    epochs = int(train_cfg.epochs)
    val_freq = int(args.val_freq)
    save_freq = int(args.save_freq)

    logger.info(f"Training for {epochs} epochs (start={start_epoch})")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    from tqdm.auto import tqdm

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = total_recon = total_tc = total_mi = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
            for batch in pbar:
                optimizer.zero_grad(set_to_none=True)
                vae_inputs = _real_batch_to_vae_inputs(batch, device, ligand_vocab_size)
                out = model(**vae_inputs, return_losses=True)
                losses = out["losses"]
                loss = sum(losses.values())
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += float(loss)
                total_recon += float(losses.get("recon_loss", 0))
                total_tc += float(losses.get("tc_loss", 0))
                total_mi += float(losses.get("mi_loss", 0))
                n_batches += 1

                pbar.set_postfix(loss=f"{float(loss):.4f}", recon=f"{float(losses.get('recon_loss', 0)):.4f}")
            pbar.close()

            avg = lambda x: x / max(n_batches, 1)
            logger.info(
                "[Train] Epoch %d/%d | loss=%.4f (recon=%.4f tc=%.4f mi=%.4f) | lr=%.2e" % (
                    epoch + 1, epochs,
                    avg(total_loss), avg(total_recon), avg(total_tc), avg(total_mi),
                    optimizer.param_groups[0]["lr"],
                )
            )
            writer.add_scalar("train/loss", avg(total_loss), epoch)
            writer.add_scalar("train/recon_loss", avg(total_recon), epoch)
            writer.add_scalar("train/tc_loss", avg(total_tc), epoch)
            writer.add_scalar("train/mi_loss", avg(total_mi), epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

            # ── 验证 ──────────────────────────────────────────────────────────
            if (epoch + 1) % val_freq == 0 or epoch == epochs - 1:
                model.eval()
                val_total = val_recon = val_tc = val_mi = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        vae_inputs = _real_batch_to_vae_inputs(batch, device, ligand_vocab_size)
                        out = model(**vae_inputs, return_losses=True)
                        losses = out["losses"]
                        val_total += float(sum(losses.values()))
                        val_recon += float(losses.get("recon_loss", 0))
                        val_tc += float(losses.get("tc_loss", 0))
                        val_mi += float(losses.get("mi_loss", 0))
                        val_batches += 1

                avgv = lambda x: x / max(val_batches, 1)
                avg_val = avgv(val_total)
                logger.info(
                    "[Val]   Epoch %d/%d | loss=%.4f (recon=%.4f tc=%.4f mi=%.4f)" % (
                        epoch + 1, epochs,
                        avg_val, avgv(val_recon), avgv(val_tc), avgv(val_mi),
                    )
                )
                writer.add_scalar("val/loss", avg_val, epoch)
                writer.add_scalar("val/recon_loss", avgv(val_recon), epoch)
                writer.add_scalar("val/tc_loss", avgv(val_tc), epoch)
                writer.add_scalar("val/mi_loss", avgv(val_mi), epoch)
                writer.flush()

                scheduler.step(avg_val)

                if best_val_loss is None or avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    _save_checkpoint(
                        model=model, cfg=cfg, epoch=epoch,
                        val_loss=best_val_loss, path=best_path,
                    )
                    logger.info(f"  Best val loss {best_val_loss:.4f} -> saved {best_path}")

            # ── 周期性保存 ────────────────────────────────────────────────────
            if (epoch + 1) % save_freq == 0:
                periodic_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
                _save_checkpoint(
                    model=model, cfg=cfg, epoch=epoch,
                    val_loss=best_val_loss, path=periodic_path,
                )
                logger.info(f"  Periodic checkpoint -> {periodic_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted — saving current state...")

    # ── 最终 checkpoint ───────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _save_checkpoint(
        model=model, cfg=cfg, epoch=epochs - 1,
        val_loss=best_val_loss, path=args.output,
    )
    logger.info(f"Final checkpoint saved: {args.output}")
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
