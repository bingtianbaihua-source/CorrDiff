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
