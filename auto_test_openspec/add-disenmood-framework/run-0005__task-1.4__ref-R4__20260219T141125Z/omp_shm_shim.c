#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
// Interpose shim for sandboxed macOS environments where OpenMP's shared-memory
// probes can fail. Redirect suspicious SHM paths/names to regular files under
// /tmp/codex_shm and be permissive about O_CREAT.

static void build_redirect_path(const char *name_or_path, char *out_path, size_t out_size) {
  const char *base = "/tmp/codex_shm";
  (void)mkdir(base, 0700);

  char normalized[NAME_MAX];
  size_t n = 0;
  for (const char *p = name_or_path; *p && n + 1 < sizeof(normalized); ++p) {
    char c = *p;
    if (c == '/') c = '_';
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

static bool looks_like_kmp_shm(const char *s) {
  if (s == NULL) return false;
  // Heuristic: only redirect OpenMP-ish shared memory probes/paths.
  if (strstr(s, "KMP") == NULL && strstr(s, "iomp") == NULL && strstr(s, "omp") == NULL) return false;
  if (strstr(s, "SHM") == NULL && strstr(s, "shm") == NULL) return false;
  return true;
}

static int open_redirect_impl(const char *path, int oflag, mode_t mode) {
  if (path == NULL) {
    errno = EINVAL;
    return -1;
  }
  char redirected[PATH_MAX];
  build_redirect_path(path, redirected, sizeof(redirected));
  int flags = oflag | O_CREAT;
  return open(redirected, flags, mode);
}

static int shm_open_redirect_impl(const char *name, int oflag, mode_t mode) {
  if (name == NULL) {
    errno = EINVAL;
    return -1;
  }
  char redirected[PATH_MAX];
  build_redirect_path(name, redirected, sizeof(redirected));
  int flags = oflag | O_CREAT;
  return open(redirected, flags, mode);
}

static int shim_open_common(const char *path, int oflag, va_list ap) {
  mode_t mode = 0;
  if (oflag & O_CREAT) {
    mode = (mode_t)va_arg(ap, int);
  }
  if (looks_like_kmp_shm(path)) {
    return open_redirect_impl(path, oflag, mode);
  }
  return open(path, oflag, mode);
}

static int shim_openat_common(int fd, const char *path, int oflag, va_list ap) {
  mode_t mode = 0;
  if (oflag & O_CREAT) {
    mode = (mode_t)va_arg(ap, int);
  }
  if (looks_like_kmp_shm(path)) {
    char redirected[PATH_MAX];
    build_redirect_path(path, redirected, sizeof(redirected));
    int flags = oflag | O_CREAT;
    return open(redirected, flags, mode);
  }
  return openat(fd, path, oflag, mode);
}

// Forward declarations for dyld interpose.
__attribute__((visibility("default"))) int shim_shm_open(const char *name, int oflag, ...);
__attribute__((visibility("default"))) int shim_shm_unlink(const char *name);
__attribute__((visibility("default"))) int shim_open(const char *path, int oflag, ...);
__attribute__((visibility("default"))) int shim_openat(int fd, const char *path, int oflag, ...);

#define DYLD_INTERPOSE(_replacement, _replacee)                                   \
  __attribute__((used)) static struct {                                           \
    const void *replacement;                                                      \
    const void *replacee;                                                         \
  } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {     \
      (const void *)(uintptr_t)&_replacement, (const void *)(uintptr_t)&_replacee \
  };

DYLD_INTERPOSE(shim_shm_open, shm_open)
DYLD_INTERPOSE(shim_shm_unlink, shm_unlink)
DYLD_INTERPOSE(shim_open, open)
DYLD_INTERPOSE(shim_openat, openat)

__attribute__((visibility("default"))) int shim_shm_open(const char *name, int oflag, ...) {
  mode_t mode = 0;
  if (oflag & O_CREAT) {
    va_list ap;
    va_start(ap, oflag);
    mode = (mode_t)va_arg(ap, int);
    va_end(ap);
  }
  return shm_open_redirect_impl(name, oflag, mode);
}

__attribute__((visibility("default"))) int shim_shm_unlink(const char *name) {
  if (name == NULL) {
    errno = EINVAL;
    return -1;
  }
  // Mirror unlink on redirected path.
  char redirected[PATH_MAX];
  build_redirect_path(name, redirected, sizeof(redirected));
  return unlink(redirected);
}

__attribute__((visibility("default"))) int shim_open(const char *path, int oflag, ...) {
  va_list ap;
  va_start(ap, oflag);
  int rc = shim_open_common(path, oflag, ap);
  va_end(ap);
  return rc;
}

__attribute__((visibility("default"))) int shim_openat(int fd, const char *path, int oflag, ...) {
  va_list ap;
  va_start(ap, oflag);
  int rc = shim_openat_common(fd, path, oflag, ap);
  va_end(ap);
  return rc;
}
