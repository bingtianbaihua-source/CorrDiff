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

CONFIG_PATH="${BUNDLE_DIR}/inputs/latent_mode_config.yml"
OUTPUT_PATH="${BUNDLE_DIR}/outputs/latent_batch.npz"
LOG_PATH="${BUNDLE_DIR}/logs/encode_latents.log"

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
"${PYTHON_BIN}" scripts/encode_latents.py --config "${CONFIG_PATH}" --output "${OUTPUT_PATH}" | tee "${LOG_PATH}"

grep -q "Latent encoding mode: offline" "${LOG_PATH}"

CONFIG_PATH="${CONFIG_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" "${PYTHON_BIN}" - <<'PY'
import os

import numpy as np
import yaml

cfg_path = os.environ["CONFIG_PATH"]
out_path = os.environ["OUTPUT_PATH"]

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

batch_size = int((cfg.get("encode") or {}).get("batch_size", 2))
z_shared_dim = int((cfg.get("model") or {}).get("z_shared_dim"))
z_pi_dim = int((cfg.get("model") or {}).get("z_pi_dim"))

obj = np.load(out_path, allow_pickle=True)
if "z_shared" not in obj:
    raise SystemExit("Missing key z_shared in output")
z_pi_keys = [k for k in obj.files if k.startswith("z_pi_")]
if not z_pi_keys:
    raise SystemExit("Missing any z_pi_* key in output")

z_shared = obj["z_shared"]
z_pi0 = obj[sorted(z_pi_keys, key=lambda s: int(s.split('_')[-1]))[0]]

if tuple(z_shared.shape) != (batch_size, z_shared_dim):
    raise SystemExit(f"z_shared shape mismatch: got {z_shared.shape}, expected {(batch_size, z_shared_dim)}")
if tuple(z_pi0.shape) != (batch_size, z_pi_dim):
    raise SystemExit(f"z_pi_0 shape mismatch: got {z_pi0.shape}, expected {(batch_size, z_pi_dim)}")

print("OK: latent_batch.npz contains z_shared and z_pi_* with expected shapes")
PY
