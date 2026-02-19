import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from models.molopt_score_model import BranchDiffusion


def main() -> int:
    root = Path(os.environ["BUNDLE_DIR"])
    cfg = json.loads((root / "inputs" / "diffusion_config.json").read_text())

    z_shared_dim = int(cfg["z_shared_dim"])
    z_pi_dim = int(cfg["z_pi_dim"])
    property_names = list(cfg["property_names"])
    num_steps = int(cfg["num_steps"])
    batch_size = int(cfg["batch_size"])

    diffusion = BranchDiffusion(
        z_shared_dim=z_shared_dim,
        z_pi_dim=z_pi_dim,
        property_names=property_names,
        num_steps=num_steps,
    )

    latents = diffusion.sample(batch_size=batch_size)

    out_path = root / "outputs" / "sample_latents.npz"
    arrays = {"z_shared": latents["z_shared"].detach().cpu().numpy()}
    for name, z in latents["z_pi"].items():
        arrays[str(name)] = z.detach().cpu().numpy()
    np.savez(out_path, **arrays)

    loaded = np.load(out_path)
    if "z_shared" not in loaded.files:
        raise AssertionError("Missing key 'z_shared' in sample_latents.npz")
    if loaded["z_shared"].shape != (batch_size, z_shared_dim):
        raise AssertionError(
            f"z_shared shape mismatch: expected {(batch_size, z_shared_dim)}, got {loaded['z_shared'].shape}"
        )

    for name in property_names:
        if name not in loaded.files:
            raise AssertionError(f"Missing z_pi key '{name}' in sample_latents.npz")
        if loaded[name].shape != (batch_size, z_pi_dim):
            raise AssertionError(
                f"{name} shape mismatch: expected {(batch_size, z_pi_dim)}, got {loaded[name].shape}"
            )

    print(f"Wrote {out_path}")
    print("Verified keys and shapes:", ["z_shared"] + property_names)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise

