#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

INPUTS_DIR="${RUN_DIR}/inputs"
OUTPUTS_DIR="${RUN_DIR}/outputs"
export SAMPLE_LATENTS_NPZ="${INPUTS_DIR}/sample_latents.npz"
export DECODED_LIGAND_NPZ="${OUTPUTS_DIR}/decoded_ligand.npz"

mkdir -p "${INPUTS_DIR}" "${OUTPUTS_DIR}"

python - <<'PY'
import os

from utils.npz_io import NpyArray, save_npz

z_shared = [[0.1, -0.2, 0.3, 0.4]]
z_pi_0 = [[-0.5, 0.6, -0.7, 0.8]]

save_npz(
    os.environ["SAMPLE_LATENTS_NPZ"],
    {
        "z_shared": NpyArray(descr="<f4", shape=(1, 4), data=z_shared),
        "z_pi_0": NpyArray(descr="<f4", shape=(1, 4), data=z_pi_0),
    },
)
print("Wrote inputs/sample_latents.npz")
PY

CORRDIFF_TOY_NO_TORCH=1 \
  python "${REPO_ROOT}/scripts/decode_latents.py" \
    --input "${SAMPLE_LATENTS_NPZ}" \
    --output "${DECODED_LIGAND_NPZ}" \
    --num-atoms 8

python - <<'PY'
import os

from utils.npz_io import load_npz

out = load_npz(os.environ["DECODED_LIGAND_NPZ"])

xyz = out["xyz"]
atomic_nums = out["atomic_nums"]
smiles_utf8 = out["smiles_utf8"]

if xyz.shape[1:] != (3,):
    raise AssertionError(f"xyz must have shape (N,3), got {xyz.shape!r}")
if atomic_nums.shape != (xyz.shape[0],):
    raise AssertionError(f"atomic_nums must have shape (N,), got {atomic_nums.shape!r} (N={xyz.shape[0]})")

smiles_bytes = bytes(int(b) for b in smiles_utf8.data)
smiles = smiles_bytes.decode("utf-8", errors="replace")

if smiles == "":
    raise AssertionError("smiles must be non-empty")
if smiles == "SMILES_UNAVAILABLE_IN_TOY_MODE":
    print("OK: SMILES placeholder present (toy/optional-deps mode)")
else:
    print("OK: SMILES present")

print(f"OK: xyz.shape={xyz.shape}, atomic_nums.shape={atomic_nums.shape}, smiles={smiles!r}")
PY

python - <<'PY'
import builtins

from utils.smiles_export import reconstruct_smiles

orig_import = builtins.__import__

def _no_rdkit(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "rdkit" or name.startswith("rdkit."):
        raise ImportError("forced rdkit-unavailable for fallback test")
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = _no_rdkit
try:
    xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    atomic_nums = [6, 6]
    res = reconstruct_smiles(xyz, atomic_nums, basic_mode=True)
finally:
    builtins.__import__ = orig_import

if res.smiles != "SMILES_UNAVAILABLE_IN_TOY_MODE":
    raise AssertionError(f"Expected placeholder fallback, got: {res.smiles!r}")
print("OK: smiles_export fallback placeholder (forced rdkit-unavailable)")
PY
