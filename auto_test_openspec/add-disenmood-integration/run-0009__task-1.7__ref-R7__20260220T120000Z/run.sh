#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}/inputs" "${RUN_DIR}/outputs"

cat > "${RUN_DIR}/inputs/decode_contract.json" <<'JSON'
{
  "required": ["xyz", "atomic_nums"],
  "optional": ["aromatic", "atom_affinity"],
  "reconstruction_fn": "utils.reconstruct.reconstruct_from_generated"
}
JSON

python "${RUN_DIR}/check_decode_contract.py"

OUT="${RUN_DIR}/outputs/decode_contract_check.txt"
test -f "${OUT}"
grep -q "REQUIRED:" "${OUT}"
grep -q "RECONSTRUCTION_FN:" "${OUT}"
grep -q "COMPATIBILITY: PASS" "${OUT}"

echo "Wrote: ${OUT}"

