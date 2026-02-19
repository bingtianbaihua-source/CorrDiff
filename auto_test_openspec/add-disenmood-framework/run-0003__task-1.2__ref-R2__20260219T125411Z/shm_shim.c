#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
// A tiny shim to emulate shm_open/shm_unlink using regular files under /tmp.
//
// This is useful in sandboxed environments where shm_open is blocked (EPERM),
// but libraries (e.g. OpenMP runtimes) expect it to work.

static void shm_build_path(const char *name, char *out_path, size_t out_size) {
  const char *base = "/tmp/codex_shm";
  // Ensure directory exists.
  (void)mkdir(base, 0700);

  // Normalize name into a safe filename component.
  char normalized[NAME_MAX];
  size_t n = 0;
  for (const char *p = name; *p && n + 1 < sizeof(normalized); ++p) {
    char c = *p;
    if (c == '/') {
      c = '_';
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
      normalized[n++] = c;
    } else {
      normalized[n++] = '_';
    }
  }
  normalized[n] = '\0';
  if (normalized[0] == '\0') {
    strncpy(normalized, "shm", sizeof(normalized) - 1);
    normalized[sizeof(normalized) - 1] = '\0';
  }

  (void)snprintf(out_path, out_size, "%s/%s", base, normalized);
}

static int shim_shm_open_impl(const char *name, int oflag, mode_t mode) {
  if (name == NULL) {
    errno = EINVAL;
    return -1;
  }
  char path[PATH_MAX];
  shm_build_path(name, path, sizeof(path));
  // Be permissive: some runtimes probe / open shared segments multiple times.
  // If they omit O_CREAT on a subsequent open, treat it as "create if missing".
  int flags = oflag | O_CREAT;
  return open(path, flags, mode);
}

static int shim_shm_unlink(const char *name) {
  if (name == NULL) {
    errno = EINVAL;
    return -1;
  }
  char path[PATH_MAX];
  shm_build_path(name, path, sizeof(path));
  return unlink(path);
}

// Forward declarations for dyld interpose.
__attribute__((visibility("default"))) int shim_shm_open(const char *name, int oflag, ...);

// dyld interpose works with macOS two-level namespaces.
#define DYLD_INTERPOSE(_replacement, _replacee)                                   \
  __attribute__((used)) static struct {                                           \
    const void *replacement;                                                      \
    const void *replacee;                                                         \
  } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {     \
      (const void *)(uintptr_t)&_replacement, (const void *)(uintptr_t)&_replacee \
  };

DYLD_INTERPOSE(shim_shm_open, shm_open)
DYLD_INTERPOSE(shim_shm_unlink, shm_unlink)

// Also export the symbols in case a consumer resolves them dynamically.
__attribute__((visibility("default"))) int shim_shm_open(const char *name, int oflag, ...) {
  mode_t mode = 0;
  if (oflag & O_CREAT) {
    va_list ap;
    va_start(ap, oflag);
    mode = (mode_t)va_arg(ap, int);
    va_end(ap);
  }
  return shim_shm_open_impl(name, oflag, mode);
}

__attribute__((visibility("default"))) int shm_open(const char *name, int oflag, ...) {
  mode_t mode = 0;
  if (oflag & O_CREAT) {
    va_list ap;
    va_start(ap, oflag);
    mode = (mode_t)va_arg(ap, int);
    va_end(ap);
  }
  return shim_shm_open_impl(name, oflag, mode);
}

__attribute__((visibility("default"))) int shm_unlink(const char *name) {
  return shim_shm_unlink(name);
}
