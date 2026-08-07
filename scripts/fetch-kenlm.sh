#!/usr/bin/env bash
# Fetch the KenLM sources this library builds against, at a pinned commit.
#
# Pinned rather than tracking master, because KenLM decides the scores the
# language model contributes. A silent change upstream would move transcripts,
# and the failure would look like a bad model rather than a changed
# dependency. (classroom-captions' scripts/setup.sh currently downloads
# kenlm master unpinned for the Python bindings — worth aligning.)
#
# Fetched over git rather than as a release tarball so the download verifies
# itself: a git commit id is a hash of its content, so checking out the pinned
# id and confirming HEAD matches proves the tree is the intended one. Pinning
# a tarball checksum would be weaker — GitHub's generated archives are not
# guaranteed byte-stable over time, so a legitimate change in their
# compression would be indistinguishable from tampering.
#
# KenLM is LGPL. It is built as its own shared library and linked
# dynamically, so it stays a replaceable component; see NOTICE.
set -euo pipefail

# kpu/kenlm master as of 2025-03-30.
KENLM_COMMIT="4cb443e60b7bf2c0ddf3c745378f76cb59e254e5"
KENLM_REPO="https://github.com/kpu/kenlm.git"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/third_party/kenlm}"

if [[ -f "$DEST/.commit" ]] && [[ "$(cat "$DEST/.commit")" == "$KENLM_COMMIT" ]]; then
  echo "kenlm already at $KENLM_COMMIT"
  exit 0
fi

echo "==> Fetching kenlm $KENLM_COMMIT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -c protocol.version=2 init -q "$TMP/kenlm"
git -C "$TMP/kenlm" remote add origin "$KENLM_REPO"
git -C "$TMP/kenlm" fetch -q --depth 1 origin "$KENLM_COMMIT"
git -C "$TMP/kenlm" checkout -q FETCH_HEAD

GOT="$(git -C "$TMP/kenlm" rev-parse HEAD)"
if [[ "$GOT" != "$KENLM_COMMIT" ]]; then
  echo "error: fetched $GOT, expected $KENLM_COMMIT" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP/kenlm" "$DEST"
rm -rf "$DEST/.git"
echo "$KENLM_COMMIT" > "$DEST/.commit"

echo "kenlm sources in $DEST (verified at $KENLM_COMMIT)"
