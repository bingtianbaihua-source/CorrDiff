#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG_PATH="$SCRIPT_DIR/inputs/eval_config.json"
OUTPUT_PATH="$SCRIPT_DIR/outputs/eval_report.json"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

python scripts/eval_disenmood.py --config "$CONFIG_PATH" --output "$OUTPUT_PATH"

if [[ ! -f "$OUTPUT_PATH" ]]; then
  echo "Missing output: $OUTPUT_PATH" >&2
  exit 1
fi

python - "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
data = json.loads(out_path.read_text(encoding="utf-8"))
required = ["HV", "Sparsity", "correlation_error", "intervention_result", "docking_scores"]
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(f"Missing keys: {missing}")
print("OK: required keys present")
PY

