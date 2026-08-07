#!/usr/bin/env bash
# Install to a clean prefix and load the result from there.
#
# This exists because the build tree cannot show the failure it checks for.
# A library built by CMake keeps an rpath pointing back at its own build
# directory, so it finds its neighbours no matter how the install is
# configured; `cmake --install` strips that rpath, and the installed copy then
# fails to load with "Library not loaded: @rpath/libkenlm.dylib". Every test
# passed while that was true, because every test loaded the build-tree copy.
#
# Anything that ships is the installed artifact, so that is what gets loaded
# here.
#
#     ./scripts/check-install.sh [build-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build/check-install}"
[ $# -gt 0 ] && shift  # anything further is passed to cmake when configuring
PREFIX="$(mktemp -d)"
trap 'rm -rf "$PREFIX"' EXIT

PYTHON="${PYTHON:-python3}"

# Reuse a build directory that is already configured rather than making one.
#
# Configuring afresh here silently chose different options from the build
# under test — it took the default for CTCBD_WITH_KENLM, so a deliberate
# -DCTCBD_WITH_KENLM=OFF build was checked as though it were an ON one, and
# failed to configure at all where the KenLM sources had not been fetched.
# Whatever was built is what should be installed and loaded.
if [ -f "$BUILD/CMakeCache.txt" ]; then
  echo "==> Using the existing build in $BUILD"
else
  echo "==> Configuring $BUILD"
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@" > /dev/null
fi
cmake --build "$BUILD" --config Release \
  -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" > /dev/null

echo "==> Installing to $PREFIX"
cmake --install "$BUILD" --prefix "$PREFIX" > /dev/null

case "$(uname -s)" in
  Darwin) LIB="$PREFIX/lib/libctc_beam_decoder.dylib" ;;
  Linux)  LIB="$PREFIX/lib/libctc_beam_decoder.so" ;;
  *)      LIB="$PREFIX/bin/ctc_beam_decoder.dll" ;;
esac

if [[ ! -f "$LIB" ]]; then
  echo "error: nothing installed at $LIB" >&2
  ls -R "$PREFIX" >&2
  exit 1
fi

echo "==> Loading $LIB"
# Deliberately no library search path set: if the installed decoder needs one
# to find KenLM sitting beside it, that is the bug this script is for.
"$PYTHON" - "$LIB" <<'PY'
import ctypes
import sys

path = sys.argv[1]
lib = ctypes.CDLL(path)          # raises OSError if a dependency is unresolved
lib.ctcbd_abi_version.restype = ctypes.c_int
lib.ctcbd_has_kenlm.restype = ctypes.c_int
print(f"    loaded, ABI {lib.ctcbd_abi_version()}, kenlm={bool(lib.ctcbd_has_kenlm())}")

# Decode something, so this covers the KenLM symbols actually being resolvable
# rather than only the library opening.
err = ctypes.create_string_buffer(512)
labels = ["▁a", "▁b", "c"]
arr = (ctypes.c_char_p * len(labels))(*[s.encode() for s in labels])
lib.ctcbd_create.restype = ctypes.c_void_p
dec = lib.ctcbd_create(arr, len(labels), err, 512)
if not dec:
    raise SystemExit(f"ctcbd_create failed: {err.value!r}")
lib.ctcbd_n_columns.argtypes = [ctypes.c_void_p]
lib.ctcbd_n_columns.restype = ctypes.c_int32
cols = lib.ctcbd_n_columns(ctypes.c_void_p(dec))

frames = 4
data = (ctypes.c_float * (frames * cols))(*([-1.0] * (frames * cols)))
lib.ctcbd_decode.restype = ctypes.c_void_p
lib.ctcbd_decode.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32, ctypes.c_double,
    ctypes.c_char_p, ctypes.c_int32,
]
res = lib.ctcbd_decode(ctypes.c_void_p(dec), data, frames, cols, 8, -10.0, -5.0, 0,
                       None, 0, 10.0, err, 512)
if not res:
    raise SystemExit(f"ctcbd_decode failed: {err.value!r}")
lib.ctcbd_result_free.argtypes = [ctypes.c_void_p]
lib.ctcbd_result_free(ctypes.c_void_p(res))
lib.ctcbd_free.argtypes = [ctypes.c_void_p]
lib.ctcbd_free(ctypes.c_void_p(dec))
print("    decoded from the installed library")
PY

echo "install looks good"
