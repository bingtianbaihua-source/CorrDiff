#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${HERE}/../../.." && pwd)"

mkdir -p "${HERE}/logs" "${HERE}/outputs"

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=${HERE}"
  echo "uname=$(uname -a || true)"
  echo "python=$(command -v python || true)"
  python -V || true
  echo "uv=$(command -v uv || true)"
} > "${HERE}/logs/env.txt"

if command -v uv >/dev/null 2>&1; then
  if [ ! -d "${HERE}/.venv" ]; then
    uv venv "${HERE}/.venv" >> "${HERE}/logs/env.txt" 2>&1
  fi
  # shellcheck disable=SC1091
  source "${HERE}/.venv/bin/activate"
  python -V >> "${HERE}/logs/env.txt" 2>&1 || true
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export BUNDLE_DIR="${HERE}"

python "${ROOT_DIR}/scripts/augment_admet_labels.py" \
  --input "${HERE}/inputs/smiles_list.txt" \
  --output "${HERE}/outputs/augmented_labels.json" \
  --cache "${HERE}/outputs/admet_cache.json"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BUNDLE_DIR"])
out_path = root / "outputs" / "augmented_labels.json"
in_path = root / "inputs" / "smiles_list.txt"

required = [
  "PAMPA_NCATS",
  "BBB_Martins",
  "logP",
  "Clearance_Microsome_AZ",
  "hERG",
  "affinity",
  "QED",
  "SA",
  "AMES",
  "lipinski",
]

smiles = [ln.strip() for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
rows = json.loads(out_path.read_text(encoding="utf-8"))

assert isinstance(rows, list), "augmented_labels.json must be a list"
assert len(rows) == len(smiles), f"expected {len(smiles)} entries, got {len(rows)}"
for i, row in enumerate(rows):
  assert isinstance(row, dict), f"row {i} must be an object"
  for k in required:
    assert k in row, f"row {i} missing key: {k}"
PY

echo "OK: outputs/augmented_labels.json generated"

