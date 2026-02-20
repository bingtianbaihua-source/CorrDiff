#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_USE_SHM=0
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${BUNDLE_DIR}"
mkdir -p outputs

if [[ "$(uname -s)" == "Darwin" ]]; then
  SHIM_C="${BUNDLE_DIR}/logs/shm_shim.c"
  SHIM_DYLIB="${BUNDLE_DIR}/logs/shm_shim.dylib"
  mkdir -p "${BUNDLE_DIR}/logs"
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

python "${REPO_ROOT}/scripts/sample_disenmood.py" \
  --config "${BUNDLE_DIR}/inputs/sample_config.yml" \
  --device "cpu" \
  --out-smi "${BUNDLE_DIR}/outputs/generated_molecules.smi" \
  --out-3d-npz "${BUNDLE_DIR}/outputs/generated_molecules_3d.npz"

python - <<'PY'
import os, sys
import numpy as np

smi_path = os.path.join("outputs", "generated_molecules.smi")
npz_path = os.path.join("outputs", "generated_molecules_3d.npz")

if not os.path.exists(smi_path):
    print(f"Missing: {smi_path}", file=sys.stderr)
    sys.exit(1)
if not os.path.exists(npz_path):
    print(f"Missing: {npz_path}", file=sys.stderr)
    sys.exit(1)

with open(smi_path, "r", encoding="utf-8") as f:
    smiles_lines = [ln.strip() for ln in f if ln.strip()]
if len(smiles_lines) < 1:
    print("Expected at least one non-empty SMILES line", file=sys.stderr)
    sys.exit(1)

obj = np.load(npz_path)
xyz = obj["xyz"]
atomic_nums = obj["atomic_nums"]

if xyz.ndim != 3 or xyz.shape[-1] != 3:
    print(f"Expected xyz shape (B, N, 3), got {xyz.shape}", file=sys.stderr)
    sys.exit(1)
if atomic_nums.ndim != 2:
    print(f"Expected atomic_nums shape (B, N), got {atomic_nums.shape}", file=sys.stderr)
    sys.exit(1)
if xyz.shape[0] != len(smiles_lines):
    print(f"Count mismatch: smiles={len(smiles_lines)} vs xyz_batch={xyz.shape[0]}", file=sys.stderr)
    sys.exit(1)
if atomic_nums.shape[0] != xyz.shape[0] or atomic_nums.shape[1] != xyz.shape[1]:
    print(
        f"Shape mismatch: xyz={xyz.shape} atomic_nums={atomic_nums.shape}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
