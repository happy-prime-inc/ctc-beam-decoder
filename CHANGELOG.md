# Changelog

Release notes are taken from this file. `release.yml` extracts the section
matching the tag and uses it as the release body, so a tag without a section
here fails the release rather than publishing an empty one.

## v0.2.0

**The Linux binary in v0.1.0 does not load on Ubuntu 22.04.** If you are on
22.04, Debian 12, RHEL 9 or anything of that vintage, v0.1.0 fails with:

```
libstdc++.so.6: version `GLIBCXX_3.4.32' not found
  (required by libctc_beam_decoder.so)
```

It was built on `ubuntu-latest`, which is now 24.04. This release builds the
Linux artifacts on Ubuntu 22.04, so they need glibc 2.35 or later and run on
everything newer.

Nothing else about the binaries changed — no decoding behaviour is affected, and
the parity guarantee against pyctcdecode is unchanged. macOS and Windows users
have no reason to upgrade beyond the wheels below.

**`pip install ctc-beam-decoder` now works.** Wheels are published for macOS
(arm64), Linux (x64) and Windows (x64), carrying the shared libraries inside the
package, so there is nothing to unpack and no paths to set:

```bash
pip install ctc-beam-decoder
```

Python 3.9 and up. The binding is pure `ctypes` with no C extension, so one
wheel serves every version — and the release workflow installs and imports that
wheel on 3.9, 3.11, 3.13 and 3.14 rather than assuming it.

There is deliberately no source distribution. An sdist would build from source,
and a from-source dependency chain breaking a platform is why this project
exists.

The release bundles are still published unchanged, for consumers that pin
binaries by checksum rather than installing from an index.

### Not in this release

`-static-libstdc++` was tried as a second layer of portability and reverted. It
segfaults when the library is loaded alongside another C++ extension — two
copies of the C++ runtime's global state in one process — which is the normal
case, both in this project's own parity tests and in any application that loads
the decoder next to another native library. Building against an old enough
toolchain is sufficient on its own. The reasoning is kept in `CMakeLists.txt`,
because it is a plausible thing to try again.

## v0.1.0

Prebuilt decoder and KenLM libraries for macOS (arm64), Linux (x64) and Windows
(x64). Each bundle carries both, along with the Python binding: the decoder
resolves KenLM beside itself by relative path, so they must travel together.

A C++ CTC prefix beam search reproducing pyctcdecode's behaviour, with KenLM
shallow fusion and hotword biasing, verified against a frozen oracle built from
pyctcdecode itself.
