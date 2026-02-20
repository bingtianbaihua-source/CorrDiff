from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


LatentEncodingMode = str  # "offline" | "online"


@dataclass(frozen=True)
class LatentCache:
    """
    Simple latent cache container.

    Tensors are stored on CPU to make it easy to save/load and to use across devices.
    """

    z_shared: Any  # torch.Tensor (N, D_shared)
    z_pi: Sequence[Any]  # list[torch.Tensor] each (N, D_pi)
    property_names: Sequence[str]


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def encode_batch_latents(
    *,
    vae: Any,
    batch: dict[str, Any],
    property_names: Sequence[str],
    device: Any,
) -> tuple[Any, list[Any]]:
    """
    Returns:
      z_shared: (B, z_shared_dim)
      z_pi_list: list of (B, z_pi_dim) in property_names order
    """
    import torch

    b = _to_device(batch, device)
    enc = vae.encode(
        protein_pos=b["protein_pos"],
        protein_atom_feature=b["protein_feat"],
        ligand_pos=b["ligand_pos"],
        ligand_atom_feature=b["ligand_feat"],
        batch_protein=b["batch_protein"],
        batch_ligand=b["batch_ligand"],
    )

    z_shared = enc["z_shared"]
    z_pi_dict = enc["z_pi"]
    if not isinstance(z_pi_dict, dict):
        raise TypeError("VAE encode() must return dict z_pi")

    z_pi_list: list[torch.Tensor] = []
    for name in property_names:
        if name not in z_pi_dict:
            raise KeyError(f"Missing z_pi for property '{name}'")
        z_pi_list.append(z_pi_dict[name])
    return z_shared, z_pi_list


def build_latent_cache(
    *,
    vae: Any,
    loader: Iterable[dict[str, Any]],
    property_names: Sequence[str],
    device: Any,
) -> LatentCache:
    import torch

    vae.eval()
    z_shared_list: list[torch.Tensor] = []
    z_pi_lists: list[list[torch.Tensor]] = [[] for _ in range(len(property_names))]

    with torch.no_grad():
        for batch in loader:
            z_shared, z_pi_list = encode_batch_latents(
                vae=vae, batch=batch, property_names=property_names, device=device
            )
            z_shared_list.append(z_shared.detach().cpu())
            for i, zpi in enumerate(z_pi_list):
                z_pi_lists[i].append(zpi.detach().cpu())

    if not z_shared_list:
        raise RuntimeError("No batches encoded; latent cache would be empty")

    z_shared_cat = torch.cat(z_shared_list, dim=0)
    z_pi_cat = [torch.cat(parts, dim=0) for parts in z_pi_lists]
    return LatentCache(z_shared=z_shared_cat, z_pi=z_pi_cat, property_names=list(property_names))


def save_latent_cache_npz(*, cache: LatentCache, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    arrays: dict[str, np.ndarray] = {"z_shared": cache.z_shared.detach().cpu().numpy()}
    for i, t in enumerate(cache.z_pi):
        arrays[f"z_pi_{i}"] = t.detach().cpu().numpy()
    arrays["property_names"] = np.asarray(list(cache.property_names), dtype=object)

    np.savez(path, **arrays)


def load_latent_cache_npz(*, path: str) -> LatentCache:
    import torch

    obj = np.load(path, allow_pickle=True)
    if "z_shared" not in obj:
        raise KeyError("latent cache missing key 'z_shared'")

    z_pi_keys = sorted([k for k in obj.files if k.startswith("z_pi_")], key=lambda s: int(s.split("_")[-1]))
    if not z_pi_keys:
        raise KeyError("latent cache missing any 'z_pi_*' keys")

    property_names = obj.get("property_names", None)
    if property_names is None:
        property_names_list = [f"p{i}" for i in range(len(z_pi_keys))]
    else:
        property_names_list = [str(x) for x in property_names.tolist()]

    z_shared = torch.as_tensor(obj["z_shared"])
    z_pi = [torch.as_tensor(obj[k]) for k in z_pi_keys]
    return LatentCache(z_shared=z_shared, z_pi=z_pi, property_names=property_names_list)


def get_or_build_latent_cache_npz(
    *,
    mode: LatentEncodingMode,
    cache_path: str,
    vae: Any,
    loader: Iterable[dict[str, Any]],
    property_names: Sequence[str],
    device: Any,
) -> LatentCache:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("offline", "online"):
        raise ValueError(f"latent_encoding_mode must be 'offline' or 'online', got: {mode!r}")

    if mode_norm == "online":
        cache = build_latent_cache(vae=vae, loader=loader, property_names=property_names, device=device)
        return cache

    if os.path.exists(cache_path):
        return load_latent_cache_npz(path=cache_path)

    cache = build_latent_cache(vae=vae, loader=loader, property_names=property_names, device=device)
    save_latent_cache_npz(cache=cache, path=cache_path)
    return cache

