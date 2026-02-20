#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SCRIPT_DIR}/inputs" "${SCRIPT_DIR}/outputs"

python - <<'PY'
import random

from utils.npz_io import NpyArray, save_npz

random.seed(0)
z_shared = [random.gauss(0.0, 1.0) for _ in range(4)]
z_pi_0 = [random.gauss(0.0, 1.0) for _ in range(2)]

save_npz(
    "auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z/inputs/sample_latents.npz",
    {
        "z_shared": NpyArray(descr="<f4", shape=(4,), data=z_shared),
        "z_pi_0": NpyArray(descr="<f4", shape=(2,), data=z_pi_0),
    },
)
print("Wrote inputs/sample_latents.npz (no numpy)")
PY

python -m scripts.decode_latents \
  --input "${SCRIPT_DIR}/inputs/sample_latents.npz" \
  --output "${SCRIPT_DIR}/outputs/decoded_ligand.npz" \
  --num-atoms 8 \
|| CORRDIFF_TOY_NO_TORCH=1 python -m scripts.decode_latents \
  --input "${SCRIPT_DIR}/inputs/sample_latents.npz" \
  --output "${SCRIPT_DIR}/outputs/decoded_ligand.npz" \
  --num-atoms 8

python - <<'PY'
from utils.npz_io import load_npz

out = load_npz("auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z/outputs/decoded_ligand.npz")
for k in ("xyz", "atomic_nums", "smiles_utf8"):
    if k not in out:
        raise KeyError(f"Missing key in decoded_ligand.npz: {k}")

xyz = out["xyz"]
atomic_nums = out["atomic_nums"]
smiles_utf8 = out["smiles_utf8"]

if xyz.shape[1] != 3:
    raise AssertionError(f"xyz must be (N, 3), got: {xyz.shape}")
if atomic_nums.shape[0] != xyz.shape[0]:
    raise AssertionError(f"atomic_nums must be (N,) matching xyz, got: {atomic_nums.shape} vs {xyz.shape}")
if any((not (v == v)) for row in xyz.data for v in row):  # NaN check without numpy
    raise AssertionError("xyz contains NaN")
if any(v <= 0 for v in atomic_nums.data):
    raise AssertionError("atomic_nums must be positive")

smiles = bytes(smiles_utf8.data).decode("utf-8", errors="replace")
if not smiles:
    raise AssertionError("smiles_utf8 decodes to empty string")

print("OK: decoded_ligand.npz contains xyz (N,3), atomic_nums (N,), and exportable representation")
PY
