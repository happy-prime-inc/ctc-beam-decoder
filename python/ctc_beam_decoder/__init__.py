"""ctypes binding for libctc_beam_decoder.

The call signature mirrors pyctcdecode's, because the point of the library is
to be droppable in where pyctcdecode is:

    dec = BeamSearchDecoder(vocab)
    text = dec.decode(log_probs, hotwords=["ravi"], hotword_weight=10.0)
    beams = dec.decode_beams(log_probs, beam_width=100)

`decode_beams` returns tuples of
`(text, lm_state, word_frames, logit_score, combined_score)`, with `lm_state`
always None until language model fusion exists, so callers that unpack the
reference implementation's five-tuple keep working.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
from pathlib import Path

import numpy as np

# Bumped by the library whenever these signatures or their meaning change. The
# check happens before anything else, so a stale library fails with a sentence
# rather than a bad read.
_REQUIRED_ABI_VERSION = 2

DEFAULT_BEAM_WIDTH = 100
DEFAULT_PRUNE_LOGP = -10.0
DEFAULT_MIN_TOKEN_LOGP = -5.0
DEFAULT_HOTWORD_WEIGHT = 10.0
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 1.5
# Charged against words the language model has never seen. pyctcdecode's
# default, kept because it was measured rather than assumed: sweeping it
# -10/-5/-2/0 on a live recording moved WER 0.240/0.234/0.251/0.251 and left
# name similarity flat, because most name failures are not close calls — the
# model never produces the name at all.
DEFAULT_UNK_SCORE_OFFSET = -10.0

# Floor applied to log-probabilities, guarding against a -inf column.
MIN_TOKEN_CLIP_P = 1e-15


def _prepare(log_probs: np.ndarray) -> np.ndarray:
    """Normalise input the way pyctcdecode does, in numpy.

    This is done here rather than in C++ deliberately. The library has its own
    implementation for callers who are not using Python, but it does not agree
    with this one to the last bit: numpy's float32 `exp` and `log` are its own
    vectorised routines, and its `sum` uses pairwise accumulation. Reproducing
    all three exactly is a much harder problem than it looks, and getting it
    wrong by one unit in the last place shifts every beam score downstream —
    measured at 4 ulps of drift over a 1250-frame fixture, enough to change
    which beams merge and so which word timings come back.

    Doing it in numpy costs one vectorised pass and removes the whole problem.
    numpy is already a dependency of anything feeding this a logit matrix.
    """
    x = np.ascontiguousarray(log_probs, dtype=np.float32)
    if math.isclose(x.sum(axis=1).mean(), 1):
        # Probabilities rather than log-probabilities.
        return np.log(np.clip(x, MIN_TOKEN_CLIP_P, 1))
    x_max = np.amax(x, axis=1, keepdims=True)
    x_max[~np.isfinite(x_max)] = 0
    tmp = x - x_max
    with np.errstate(divide="ignore"):
        total = np.log(np.sum(np.exp(tmp), axis=1, keepdims=True))
    # float32 explicitly, rather than by inheritance. Under numpy 1.x this
    # clip stays float32 by value-based casting and under numpy 2.x the
    # float64 bound would widen it, so leaving it implicit would make the
    # decoder's output depend on the numpy version — the exact coupling this
    # library exists to remove.
    return np.clip(tmp - total, np.log(MIN_TOKEN_CLIP_P), 0).astype(np.float32)


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libctc_beam_decoder.dylib"
    if system == "Linux":
        return "libctc_beam_decoder.so"
    if system == "Windows":
        return "ctc_beam_decoder.dll"
    raise RuntimeError(f"ctc_beam_decoder: unsupported platform {system!r}")


def _find_library(explicit: str | os.PathLike[str] | None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"no decoder library at {p}")
        return p
    name = _library_name()
    env = os.environ.get("CTC_BEAM_DECODER_LIB")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve().parent
    build = here.parent.parent / "build" / "cmake"
    candidates += [here / name, build / name]
    # Multi-config generators nest by configuration. CMakeLists pins the
    # output directory so this should not trigger, but a build configured by
    # someone else's tooling is not ours to assume about.
    candidates += [build / config / name
                   for config in ("Release", "RelWithDebInfo", "Debug")]
    candidates.append(Path(name))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"could not find {name}; build it with cmake or set CTC_BEAM_DECODER_LIB"
    )


class _Capi:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path))
        c = self.lib

        c.ctcbd_abi_version.argtypes = []
        c.ctcbd_abi_version.restype = ctypes.c_int
        found = c.ctcbd_abi_version()
        if found != _REQUIRED_ABI_VERSION:
            raise RuntimeError(
                f"{path} has ABI version {found}, this binding needs "
                f"{_REQUIRED_ABI_VERSION}; rebuild the library"
            )

        c.ctcbd_create.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32,
            ctypes.c_char_p, ctypes.c_int32,
        ]
        c.ctcbd_create.restype = ctypes.c_void_p

        c.ctcbd_has_kenlm.argtypes = []
        c.ctcbd_has_kenlm.restype = ctypes.c_int

        c.ctcbd_create_with_lm.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32,
            ctypes.c_char_p,                                   # kenlm path
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32,   # unigrams
            ctypes.c_double, ctypes.c_double,                  # alpha, beta
            ctypes.c_double, ctypes.c_int32,                   # unk offset, boundary
            ctypes.c_char_p, ctypes.c_int32,
        ]
        c.ctcbd_create_with_lm.restype = ctypes.c_void_p

        c.ctcbd_free.argtypes = [ctypes.c_void_p]
        c.ctcbd_free.restype = None

        c.ctcbd_n_columns.argtypes = [ctypes.c_void_p]
        c.ctcbd_n_columns.restype = ctypes.c_int32

        c.ctcbd_blank_id.argtypes = [ctypes.c_void_p]
        c.ctcbd_blank_id.restype = ctypes.c_int32

        # Declared but not called from here — `_prepare` below does this in
        # numpy for parity. Kept so that a caller reaching for the C entry
        # point gets a correctly typed one rather than ctypes' int-shaped
        # guess, which would silently corrupt a pointer.
        c.ctcbd_prepare.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
        ]
        c.ctcbd_prepare.restype = None

        c.ctcbd_decode.argtypes = [
            ctypes.c_void_p,                    # decoder
            ctypes.POINTER(ctypes.c_float),     # log_probs
            ctypes.c_int32, ctypes.c_int32,     # n_frames, n_columns
            ctypes.c_int32,                     # beam_width
            ctypes.c_double, ctypes.c_double,   # beam_prune_logp, token_min_logp
            ctypes.c_int32,                     # prune_history
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_int32,  # hotwords
            ctypes.c_double,                    # hotword_weight
            ctypes.c_char_p, ctypes.c_int32,    # err
        ]
        c.ctcbd_decode.restype = ctypes.c_void_p

        c.ctcbd_sizes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        c.ctcbd_sizes.restype = None

        c.ctcbd_pack.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        c.ctcbd_pack.restype = None

        c.ctcbd_result_free.argtypes = [ctypes.c_void_p]
        c.ctcbd_result_free.restype = None


_ERR_LEN = 512


def _string_array(items: list[str]):
    arr = (ctypes.c_char_p * max(len(items), 1))()
    for i, s in enumerate(items):
        arr[i] = s.encode("utf-8")
    return arr


class BeamSearchDecoder:
    """CTC prefix beam search over a fixed vocabulary."""

    def __init__(
        self,
        labels: list[str],
        library: str | os.PathLike[str] | None = None,
        kenlm_model_path: str | os.PathLike[str] | None = None,
        unigrams: list[str] | None = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        unk_score_offset: float = DEFAULT_UNK_SCORE_OFFSET,
        score_boundary: bool = True,
    ) -> None:
        self._capi = _Capi(_find_library(library))
        err = ctypes.create_string_buffer(_ERR_LEN)
        labels = list(labels)
        if kenlm_model_path is None:
            handle = self._capi.lib.ctcbd_create(
                _string_array(labels), len(labels), err, _ERR_LEN
            )
        else:
            if not self._capi.lib.ctcbd_has_kenlm():
                raise RuntimeError(
                    "this build of the decoder has no language model support; "
                    "rebuild with -DCTCBD_WITH_KENLM=ON"
                )
            path = Path(kenlm_model_path)
            if not path.exists():
                raise FileNotFoundError(f"no language model at {path}")
            # None and [] mean different things: no vocabulary at all makes
            # every word prefix count as unknown, an empty list would claim a
            # vocabulary that happens to be empty.
            words = None if unigrams is None else list(unigrams)
            if words is None and path.suffix == ".arpa":
                # A plain ARPA carries its own vocabulary, so use it rather
                # than decoding as though nothing were known. A compressed or
                # binary model does not, which is why this is not
                # unconditional — and why the app, whose model is `.arpa.gz`,
                # runs with no vocabulary today.
                words = _unigrams_from_arpa(path)
            handle = self._capi.lib.ctcbd_create_with_lm(
                _string_array(labels), len(labels),
                str(path).encode("utf-8"),
                _string_array(words) if words is not None else None,
                0 if words is None else len(words),
                ctypes.c_double(alpha), ctypes.c_double(beta),
                ctypes.c_double(unk_score_offset), 1 if score_boundary else 0,
                err, _ERR_LEN,
            )
        if not handle:
            raise ValueError(err.value.decode("utf-8", "replace") or "could not build decoder")
        self._handle = ctypes.c_void_p(handle)
        self._n_columns = self._capi.lib.ctcbd_n_columns(self._handle)
        self._blank_id = self._capi.lib.ctcbd_blank_id(self._handle)

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            try:
                self._capi.lib.ctcbd_free(handle)
            except Exception:  # interpreter teardown can pull the library first
                pass
            self._handle = None

    @property
    def n_columns(self) -> int:
        """Logit columns expected per frame — the vocabulary plus a blank."""
        return self._n_columns

    @property
    def blank_id(self) -> int:
        return self._blank_id

    def decode_beams(
        self,
        log_probs: np.ndarray,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        prune_history: bool = False,
        hotwords: list[str] | None = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> list[tuple[str, None, list[tuple[str, tuple[int, int]]], float, float]]:
        if log_probs.ndim != 2:
            raise ValueError(f"log_probs must be 2-D (time, vocabulary), got {log_probs.shape}")
        if log_probs.shape[1] != self._n_columns:
            raise ValueError(
                f"log_probs have {log_probs.shape[1]} columns, "
                f"vocabulary needs {self._n_columns}"
            )
        x = np.ascontiguousarray(_prepare(log_probs), dtype=np.float32)
        n_frames, n_cols = x.shape
        ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        hw = list(hotwords or [])
        err = ctypes.create_string_buffer(_ERR_LEN)
        res = self._capi.lib.ctcbd_decode(
            self._handle, ptr, n_frames, n_cols, beam_width,
            ctypes.c_double(beam_prune_logp), ctypes.c_double(token_min_logp),
            1 if prune_history else 0,
            _string_array(hw), len(hw), ctypes.c_double(hotword_weight),
            err, _ERR_LEN,
        )
        if not res:
            raise RuntimeError(err.value.decode("utf-8", "replace") or "decode failed")
        res = ctypes.c_void_p(res)
        try:
            return self._unpack(res)
        finally:
            self._capi.lib.ctcbd_result_free(res)

    def _unpack(self, res):
        lib = self._capi.lib
        n_beams = ctypes.c_int32()
        total_words = ctypes.c_int32()
        text_bytes = ctypes.c_int32()
        lib.ctcbd_sizes(res, ctypes.byref(n_beams), ctypes.byref(total_words),
                        ctypes.byref(text_bytes))
        n, nw, nb = n_beams.value, total_words.value, text_bytes.value
        if n == 0:
            return []

        text_buf = ctypes.create_string_buffer(max(nb, 1))
        words_per_beam = np.empty(n, dtype=np.int32)
        frames = np.empty(max(nw * 2, 1), dtype=np.int32)
        logit_scores = np.empty(n, dtype=np.float64)
        combined_scores = np.empty(n, dtype=np.float64)
        lib.ctcbd_pack(
            res,
            text_buf,
            words_per_beam.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            frames.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            logit_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            combined_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        # The library hands back its texts NUL-separated; splitting on the
        # trailing NUL leaves one empty tail entry.
        texts = text_buf.raw[:nb].split(b"\x00")[:n]
        out = []
        at = 0
        for i in range(n):
            text = texts[i].decode("utf-8")
            count = int(words_per_beam[i])
            pairs = frames[at * 2:(at + count) * 2].reshape(count, 2)
            at += count
            word_frames = [
                (w, (int(s), int(e)))
                for w, (s, e) in zip(text.split(), pairs)
            ]
            out.append((text, None, word_frames, float(logit_scores[i]),
                        float(combined_scores[i])))
        return out

    def decode(
        self,
        log_probs: np.ndarray,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        beam_prune_logp: float = DEFAULT_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_MIN_TOKEN_LOGP,
        hotwords: list[str] | None = None,
        hotword_weight: float = DEFAULT_HOTWORD_WEIGHT,
    ) -> str:
        beams = self.decode_beams(
            log_probs,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            # History pruning, because the reference turns it on here and only
            # here. It drops beams that agree over all the history a language
            # model can still see, which costs n-best diversity — irrelevant
            # when only the top beam is wanted, and not a free choice: leaving
            # it off changes the winning transcript on 34 of 262 recorded
            # cases.
            prune_history=True,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )
        return beams[0][0] if beams else ""


def _unigrams_from_arpa(path: Path) -> list[str]:
    """Read the unigram section of an ARPA file.

    pyctcdecode does this whenever it is handed a plain `.arpa` and no
    vocabulary, and it is not cosmetic: a decoder with a known vocabulary
    penalises only word prefixes that begin no known word, while one without
    penalises every prefix. Skipping it would decode differently from the
    reference on exactly the inputs where a language model earns its keep.

    Only entries carrying a backoff weight count, which is the reference's
    rule — it reads three tab-separated fields and ignores anything else, so
    `</s>` and other backoff-less entries are left out.
    """
    unigrams = []
    started = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "\\1-grams:":
                started = True
            elif line == "\\2-grams:":
                break
            if started and line:
                parts = line.split("\t")
                if len(parts) == 3:
                    unigrams.append(parts[1])
    if not unigrams:
        raise ValueError(f"no unigrams found in {path}; is it a valid ARPA file?")
    return unigrams


def has_kenlm(library: str | os.PathLike[str] | None = None) -> bool:
    """Whether the built library can load a language model.

    A build without KenLM still does beam search, hotwords and the allowlist,
    which is the configuration the app ships; asking it for a language model
    raises rather than quietly decoding without one.
    """
    try:
        capi = _Capi(_find_library(library))
    except FileNotFoundError:
        # No library at all — nothing is built, so nothing has KenLM.
        return False
    # A library that exists but will not load, or whose ABI does not match, is
    # a broken build and says so. Swallowing those here reported a decoder
    # that could not be loaded as a valid one compiled without KenLM, which
    # sent a packaging failure looking like a configuration choice.
    return bool(capi.lib.ctcbd_has_kenlm())


def build_ctcdecoder(
    labels: list[str],
    kenlm_model_path: str | os.PathLike[str] | None = None,
    unigrams: list[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    unk_score_offset: float = DEFAULT_UNK_SCORE_OFFSET,
    lm_score_boundary: bool = True,
    library: str | os.PathLike[str] | None = None,
) -> BeamSearchDecoder:
    """pyctcdecode-shaped constructor, so a swap is a change of import."""
    return BeamSearchDecoder(
        labels,
        library=library,
        kenlm_model_path=kenlm_model_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
        unk_score_offset=unk_score_offset,
        score_boundary=lm_score_boundary,
    )
