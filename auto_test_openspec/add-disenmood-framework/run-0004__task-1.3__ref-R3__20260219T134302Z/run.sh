#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE_DIR="$HERE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_USE_SHM=0

mkdir -p "${BUNDLE_DIR}/inputs" "${BUNDLE_DIR}/outputs" "${BUNDLE_DIR}/logs"

cat > "${BUNDLE_DIR}/inputs/diffusion_config.json" <<'JSON'
{
  "z_shared_dim": 12,
  "z_pi_dim": 7,
  "property_names": ["qed", "sa", "logp", "affinity"],
  "num_steps": 12,
  "batch_size": 4
}
JSON

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
  DYLD_INSERT_LIBRARIES="${SHIM_DYLIB}" "${PYTHON_BIN}" "${BUNDLE_DIR}/runner.py"
else
  "${PYTHON_BIN}" "${BUNDLE_DIR}/runner.py"
fi
