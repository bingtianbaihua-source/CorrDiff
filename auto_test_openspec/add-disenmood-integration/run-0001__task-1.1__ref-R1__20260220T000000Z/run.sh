#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"

CONFIG_PATH="${BUNDLE_DIR}/inputs/vae_toy_config.yml"
OUTPUT_PATH="${BUNDLE_DIR}/outputs/vae_checkpoint.pt"

mkdir -p "${BUNDLE_DIR}/outputs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${REPO_ROOT}"

python scripts/train_vae.py --config "${CONFIG_PATH}" --output "${OUTPUT_PATH}"

OUTPUT_PATH="${OUTPUT_PATH}" python - <<'PY'
import os
import sys

out_path = os.environ.get("OUTPUT_PATH")
if not out_path:
    raise SystemExit("OUTPUT_PATH not set")

if not os.path.isfile(out_path):
    raise SystemExit(f"missing checkpoint: {out_path}")

import torch

ckpt = torch.load(out_path, map_location="cpu")
if not isinstance(ckpt, dict):
    raise SystemExit("checkpoint must be a dict")

enc = ckpt.get("encoder_state_dict", None)
dec = ckpt.get("decoder_state_dict", None)
if not isinstance(enc, dict) or len(enc) == 0:
    raise SystemExit("encoder_state_dict missing/empty")
if not isinstance(dec, dict) or len(dec) == 0:
    raise SystemExit("decoder_state_dict missing/empty")

print(f"OK: wrote {out_path} (encoder_keys={len(enc)}, decoder_keys={len(dec)})")
PY
