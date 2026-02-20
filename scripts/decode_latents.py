import argparse
import os
from typing import Any

from utils.npz_io import NpyArray, load_npz, save_npz


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping, got: {type(data).__name__}")
    return data


def _ensure_dir(path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode DisentangledVAE latents into toy 3D coordinates + atom types.")
    parser.add_argument("--input", type=str, required=True, help="Input .npz containing z_shared and z_pi_0.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz to write (decoded_ligand.npz).")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config (reads config.model.*).")
    parser.add_argument("--num-atoms", type=int, default=8, help="Number of atoms to decode (toy).")
    args = parser.parse_args()

    latents = load_npz(args.input)
    if "z_shared" not in latents:
        raise KeyError("Input npz must contain key 'z_shared'")
    if "z_pi_0" not in latents:
        raise KeyError("Input npz must contain key 'z_pi_0'")

    z_shared_arr = latents["z_shared"]
    z_pi_arr = latents["z_pi_0"]

    if len(z_shared_arr.shape) == 2:
        if z_shared_arr.shape[0] <= 0:
            raise ValueError("z_shared has empty batch dimension")
        z_shared = list(z_shared_arr.data[0])
    elif len(z_shared_arr.shape) == 1:
        z_shared = list(z_shared_arr.data)
    else:
        raise ValueError(f"z_shared must be 1D/2D, got shape={z_shared_arr.shape!r}")

    if len(z_pi_arr.shape) == 2:
        if z_pi_arr.shape[0] <= 0:
            raise ValueError("z_pi_0 has empty batch dimension")
        z_pi = list(z_pi_arr.data[0])
    elif len(z_pi_arr.shape) == 1:
        z_pi = list(z_pi_arr.data)
    else:
        raise ValueError(f"z_pi_0 must be 1D/2D, got shape={z_pi_arr.shape!r}")

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        raise TypeError("config.model must be a mapping")

    model_cfg = dict(model_cfg)
    model_cfg.setdefault("hidden_dim", 16)
    model_cfg.setdefault("z_shared_dim", int(len(z_shared)))
    model_cfg.setdefault("z_pi_dim", int(len(z_pi)))
    model_cfg.setdefault("property_names", ["p0"])
    model_cfg["decoder_num_atoms"] = int(args.num_atoms)

    if os.environ.get("CORRDIFF_TOY_NO_TORCH", "").strip() == "1":
        num_atoms = int(args.num_atoms)
        z_all = list(z_shared) + list(z_pi)
        z_sum = float(sum(float(v) for v in z_all))
        xyz_list = [[z_sum + 0.1 * i, z_sum - 0.1 * i, (z_sum * 0.01) + 0.05 * i] for i in range(num_atoms)]
        atom_types = model_cfg.get("decoder_atom_types", [6, 7, 8])
        atomic_nums_list = [int(atom_types[i % len(atom_types)]) for i in range(num_atoms)]
        smiles = "SMILES_UNAVAILABLE_IN_TOY_MODE"
        note = "CORRDIFF_TOY_NO_TORCH=1 (torch import disabled)"
    else:
        import torch

        from models.guide_model import DisentangledVAE
        from utils.smiles_export import reconstruct_smiles

        torch.manual_seed(0)
        device = torch.device("cpu")
        vae = DisentangledVAE(model_cfg, protein_atom_feature_dim=4, ligand_atom_feature_dim=4).to(device)
        vae.eval()

        with torch.no_grad():
            xyz, atomic_nums = vae.decode(
                torch.tensor(z_shared, dtype=torch.float32, device=device),
                torch.tensor(z_pi, dtype=torch.float32, device=device),
            )

        xyz_list = xyz.detach().cpu().to(dtype=torch.float32).tolist()
        atomic_nums_list = atomic_nums.detach().cpu().to(dtype=torch.long).tolist()

        result = reconstruct_smiles(xyz_list, atomic_nums_list, basic_mode=True)
        smiles = result.smiles
        note = "" if result.note is None else str(result.note)

    _ensure_dir(args.output)
    smiles_bytes = str(smiles).encode("utf-8", errors="replace")
    note_bytes = str(note).encode("utf-8", errors="replace")

    save_npz(
        args.output,
        {
            "xyz": NpyArray(descr="<f4", shape=(len(xyz_list), 3), data=xyz_list),
            "atomic_nums": NpyArray(descr="<i8", shape=(len(atomic_nums_list),), data=atomic_nums_list),
            "smiles_utf8": NpyArray(descr="|u1", shape=(len(smiles_bytes),), data=list(smiles_bytes)),
            "smiles_note_utf8": NpyArray(descr="|u1", shape=(len(note_bytes),), data=list(note_bytes)),
        },
    )

    print(f"Wrote: {args.output}")
    print(f"xyz: ({len(xyz_list)}, 3), atomic_nums: ({len(atomic_nums_list)},)")
    print(f"smiles: {smiles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
