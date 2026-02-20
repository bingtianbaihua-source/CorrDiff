#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Avoid OpenMP shared-memory issues on some macOS setups (e.g. SHM2 errors).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_USE_SHM=0

CONFIG_PATH="${BUNDLE_DIR}/inputs/vae_toy_config.yml"
OUTPUT_PATH="${BUNDLE_DIR}/outputs/vae_checkpoint.pt"

mkdir -p "${BUNDLE_DIR}/logs" "${BUNDLE_DIR}/outputs"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  SHIM_C="${BUNDLE_DIR}/logs/shm_shim.c"
  SHIM_DYLIB="${BUNDLE_DIR}/logs/shm_shim.dylib"
  cat > "${SHIM_C}" <<'C'
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DYLD_INTERPOSE(_replacement, _replacee) \
  __attribute__((used)) static struct { \
    const void* replacement; \
    const void* replacee; \
  } _interpose_##_replacee \
  __attribute__((section("__DATA,__interpose"))) = { \
    (const void*)(unsigned long)&_replacement, \
    (const void*)(unsigned long)&_replacee \
  };

static int build_path(const char *name, char *out, size_t out_sz) {
  const char *tmp = getenv("TMPDIR");
  if (!tmp || tmp[0] == '\0') tmp = "/tmp";

  const char *n = name;
  while (n && *n == '/') n++;
  if (!n || *n == '\0') n = "unnamed";

  int written = snprintf(out, out_sz, "%s/codex_shm_%s", tmp, n);
  if (written < 0) return -1;
  if ((size_t)written >= out_sz) return -1;
  return 0;
}

static int shim_shm_open(const char *name, int oflag, mode_t mode) {
  char path[PATH_MAX];
  if (build_path(name, path, sizeof(path)) != 0) {
    return -1;
  }
  int flags = oflag;
  if (!(flags & O_CREAT)) flags |= O_CREAT;
  return open(path, flags, mode);
}

static int shim_shm_unlink(const char *name) {
  char path[PATH_MAX];
  if (build_path(name, path, sizeof(path)) != 0) {
    return -1;
  }
  return unlink(path);
}

DYLD_INTERPOSE(shim_shm_open, shm_open)
DYLD_INTERPOSE(shim_shm_unlink, shm_unlink)
C

  clang -dynamiclib -o "${SHIM_DYLIB}" "${SHIM_C}"
  export DYLD_INSERT_LIBRARIES="${SHIM_DYLIB}"
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/train_vae.py --config "${CONFIG_PATH}" --output "${OUTPUT_PATH}"

OUTPUT_PATH="${OUTPUT_PATH}" "${PYTHON_BIN}" - <<'PY'
import os

import torch

ckpt_path = os.environ.get("OUTPUT_PATH")
if not ckpt_path:
    raise SystemExit("OUTPUT_PATH not set")
if not os.path.exists(ckpt_path):
    raise SystemExit(f"Missing checkpoint: {ckpt_path}")
obj = torch.load(ckpt_path, map_location="cpu")
if not isinstance(obj, dict):
    raise SystemExit("Checkpoint must be a dict")
missing = [k for k in ("encoder_state_dict", "decoder_state_dict") if k not in obj]
if missing:
    raise SystemExit(f"Missing keys in checkpoint: {missing}")
if not obj["encoder_state_dict"] or not obj["decoder_state_dict"]:
    raise SystemExit("encoder_state_dict/decoder_state_dict must be non-empty")
print("OK: checkpoint contains encoder_state_dict + decoder_state_dict")
PY
