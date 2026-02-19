#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE_DIR="$HERE"

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export KMP_DISABLE_SHM=1

python3 "$BUNDLE_DIR/runner.py"
