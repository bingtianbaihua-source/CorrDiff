#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE_DIR="$HERE"
export KMP_DISABLE_SHM=1
export KMP_USE_SHM=0
export OMP_NUM_THREADS=1

SHIM_SRC="$BUNDLE_DIR/shm_shim.c"
SHIM_LIB="$BUNDLE_DIR/libshm_shim.dylib"
clang -O2 -dynamiclib -o "$SHIM_LIB" "$SHIM_SRC"
export DYLD_INSERT_LIBRARIES="$SHIM_LIB${DYLD_INSERT_LIBRARIES:+:$DYLD_INSERT_LIBRARIES}"
export DYLD_FORCE_FLAT_NAMESPACE=1

python3 "$BUNDLE_DIR/run_vae_bundle.py"
