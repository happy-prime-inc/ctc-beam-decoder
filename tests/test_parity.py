"""Hold the decoder to pyctcdecode's behaviour, without needing a real model.

The 131-fixture comparison in `tools/` is the real acceptance test, but it
needs a gigabyte of model and a separate fixture repository. These run
anywhere: they decode random logits over small vocabularies with both
implementations and require the answers to be the same.

Random rather than fixed inputs on purpose. A handful of committed cases only
covers the paths whoever wrote them thought of; a few hundred random ones
reach the combinations nobody would think to write down — which is how the
two frame-handling bugs in this decoder were found.

    pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import ctc_beam_decoder  # noqa: E402
from token_order import ascending  # noqa: E402

# pyctcdecode is optional here, and that is the point of the exercise.
#
# It is needed only by the tests that compare against it. The rest — that the
# library builds, is named what the binding looks for, lands where the binding
# searches, loads, decodes, and boosts a hotword — need nothing but this
# library, and those are the ones worth running on the platform this project
# exists to unblock.
#
# Installing it there is actively harmful: it pins numpy<2, numpy 1.26 has no
# Windows wheel for Python 3.13, so pip builds it from source for ten minutes
# and produces a numpy that crashes on import. That is the original defect
# this decoder was written to escape, and CI reproduced it faithfully.
try:
    import pyctcdecode  # noqa: E402
    import pyctcdecode.decoder as pd  # noqa: E402
except ImportError:  # pragma: no cover
    pyctcdecode = None
    pd = None

needs_reference = pytest.mark.skipif(
    pyctcdecode is None,
    reason="pyctcdecode is not installed; comparison tests need something to compare with",
)

TINY_ARPA = str(Path(__file__).resolve().parent / "tiny.arpa")

# Word boundaries marked on a word's first token, and word boundaries as a
# token of their own. pyctcdecode handles both and so must this.
VOCABS = {
    "subword": ["▁a", "▁b", "c", "d", "▁", "<unk>"],
    "space": [" ", "a", "b", "c", "d", "<unk>"],
}
BEAM_WIDTH = 8
TRIALS = 60
FRAMES = 12


def logits_for(seed: int, n_cols: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 3, size=(FRAMES, n_cols)).astype(np.float32)


def decode_both(py, cpp, logits, **kwargs):
    """Decode with both, with pyctcdecode visiting tokens in ascending order.

    Its natural order is the iteration order of a Python set, which is
    arbitrary but fixed, and on a small minority of inputs that decides how
    near-ties break. Pinning it to ascending removes a source of disagreement
    that is not about either implementation being wrong. See
    tools/check_order_sensitivity.py for the measurement on real audio.
    """
    pd.set = ascending()
    try:
        expected = py.decode_beams(logits, beam_width=BEAM_WIDTH, **kwargs)
    finally:
        pd.set = set
    return expected, cpp.decode_beams(logits, beam_width=BEAM_WIDTH, **kwargs)


# Scores are compared to this rather than exactly. They agree bit for bit on
# one machine, but `exp` and `log` are allowed to differ by an ulp between
# platforms and compilers, and beam scores accumulate over frames. This is two
# orders tighter than the acceptance bar the corpus is measured against, so it
# still catches any real change; transcripts and frame indices are compared
# exactly, which is where a defect would actually show.
SCORE_TOLERANCE = 1e-6


def assert_same(expected, got, context=""):
    assert [t for t, *_ in got] == [t for t, *_ in expected], f"transcripts differ {context}"
    for i, (e, g) in enumerate(zip(expected, got)):
        assert [(w, tuple(f)) for w, f in g[2]] == [(w, tuple(f)) for w, f in e[2]], \
            f"word frames differ on beam {i} {context}"
        assert g[3] == pytest.approx(e[3], abs=SCORE_TOLERANCE), \
            f"logit score differs on beam {i} {context}"
        assert g[4] == pytest.approx(e[4], abs=SCORE_TOLERANCE), \
            f"combined score differs on beam {i} {context}"


@pytest.fixture(scope="module", params=sorted(VOCABS))
def vocab(request):
    return VOCABS[request.param]


def assert_decode_same(py, cpp, seeds=range(TRIALS), **kwargs):
    """`decode()` is checked separately from `decode_beams()` on purpose.

    The reference does not implement one in terms of the other on equal
    settings — `decode()` turns history pruning on, since only the top beam is
    wanted and the lost n-best diversity costs nothing there. So a decoder can
    reproduce every beam and still return a different transcript, which is
    exactly the defect this test exists for: it shipped once, because the
    parity harness recorded `decode()` and only ever called `decode_beams()`.
    """
    for seed in seeds:
        logits = logits_for(seed, cpp.n_columns)
        pd.set = ascending()
        try:
            expected = py.decode(logits, beam_width=BEAM_WIDTH, **kwargs)
        finally:
            pd.set = set
        assert cpp.decode(logits, beam_width=BEAM_WIDTH, **kwargs) == expected, \
            f"decode() differs (seed {seed})"


class TestWithoutLanguageModel:
    @needs_reference
    def test_matches_over_random_inputs(self, vocab):
        py = pyctcdecode.build_ctcdecoder(vocab)
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            assert_same(*decode_both(py, cpp, logits), context=f"(seed {seed})")

    @needs_reference
    def test_matches_with_hotwords(self, vocab):
        py = pyctcdecode.build_ctcdecoder(vocab)
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            assert_same(
                *decode_both(py, cpp, logits, hotwords=["ac", "b", "abcd"], hotword_weight=10.0),
                context=f"(seed {seed})",
            )

    @needs_reference
    def test_decode_matches(self, vocab):
        py = pyctcdecode.build_ctcdecoder(vocab)
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        assert_decode_same(py, cpp)

    @needs_reference
    def test_decode_matches_with_hotwords(self, vocab):
        py = pyctcdecode.build_ctcdecoder(vocab)
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        assert_decode_same(py, cpp, hotwords=["ac", "abcd"], hotword_weight=10.0)

    def test_decode_prunes_history(self, vocab):
        """`decode()` must not be `decode_beams()[0]` with the same settings.

        Asserted rather than left implicit, because the two agree on most
        inputs — which is what made the original defect survive a 131-fixture
        comparison of everything except this.
        """
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        differing = 0
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            unpruned = cpp.decode_beams(logits, beam_width=BEAM_WIDTH, prune_history=False)
            differing += cpp.decode(logits, beam_width=BEAM_WIDTH) != unpruned[0][0]
        assert differing > 0, "history pruning made no difference; is decode() passing it?"

    @pytest.mark.parametrize("separator, name", [
        (" ", "space"),
        ("\u00a0", "no-break space"),
        ("\u3000", "ideographic space"),
        ("\u202f", "narrow no-break space"),
    ])
    @needs_reference
    def test_hotwords_split_on_unicode_whitespace(self, vocab, separator, name):
        """A hotword joined by any whitespace splits into words, as it does in
        Python.

        Hotwords come from a list a person edits — a class roster, a glossary
        pasted out of a document — so a pasted non-breaking space is ordinary.
        Splitting on ASCII whitespace only made such an entry one unigram that
        no token sequence could match, so the name silently never boosted and
        the only symptom was "the allowlist does not work for this name".

        Checked two ways: it agrees with pyctcdecode, and it actually fires.
        Agreement alone would pass if both sides ignored the entry.
        """
        py = pyctcdecode.build_ctcdecoder(vocab)
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        hotwords = [f"ac{separator}bd"]
        fired = 0
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            pd.set = ascending()
            try:
                expected = py.decode(logits, beam_width=BEAM_WIDTH,
                                     hotwords=hotwords, hotword_weight=15.0)
            finally:
                pd.set = set
            got = cpp.decode(logits, beam_width=BEAM_WIDTH,
                             hotwords=hotwords, hotword_weight=15.0)
            assert got == expected, f"differs from pyctcdecode on {name} (seed {seed})"
            fired += got != cpp.decode(logits, beam_width=BEAM_WIDTH)
        assert fired > 0, f"a hotword joined by {name} never boosted anything"

    def test_hotwords_change_the_output(self, vocab):
        """The feature that beat Apple on names, asserted rather than assumed.

        Boosting has to reach the transcript, not merely be accepted as an
        argument — a decoder that ignored hotwords entirely would pass every
        parity test above, because it would match a reference that was also
        given none.
        """
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        changed = 0
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            plain = cpp.decode(logits, beam_width=BEAM_WIDTH)
            boosted = cpp.decode(logits, beam_width=BEAM_WIDTH,
                                 hotwords=["abcd", "ac"], hotword_weight=25.0)
            changed += plain != boosted
        assert changed > 0, "hotwords never altered any transcript"


class TestAskingForALanguageModel:
    """What happens when one is requested — in either kind of build.

    Everything in TestWithLanguageModel is skipped when the library is built
    without KenLM, which is correct but leaves the claim that matters most in
    that configuration untested: that asking for a language model *fails
    loudly* rather than quietly decoding without one. A silent downgrade would
    show up as a quality regression with no obvious cause, and it is exactly
    the kind of thing a build flag invites.

    These run in both builds, and assert opposite things depending on which.
    """

    def test_a_missing_model_is_an_error(self, vocab):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            ctc_beam_decoder.build_ctcdecoder(vocab, kenlm_model_path="/nonexistent/model.arpa")

    def test_never_silently_decodes_without_the_model_asked_for(self, vocab):
        """The important one: no path from "give me a language model" to a
        decoder that hasn't got one."""
        if ctc_beam_decoder.has_kenlm():
            dec = ctc_beam_decoder.build_ctcdecoder(vocab, kenlm_model_path=TINY_ARPA)
            plain = ctc_beam_decoder.build_ctcdecoder(vocab)
            logits = logits_for(0, dec.n_columns)
            differ = sum(
                dec.decode(logits_for(seed, dec.n_columns), beam_width=BEAM_WIDTH)
                != plain.decode(logits_for(seed, dec.n_columns), beam_width=BEAM_WIDTH)
                for seed in range(TRIALS)
            )
            assert differ > 0, "asked for a language model and got plain decoding"
        else:
            with pytest.raises(RuntimeError, match="CTCBD_WITH_KENLM"):
                ctc_beam_decoder.build_ctcdecoder(vocab, kenlm_model_path=TINY_ARPA)

    def test_a_broken_library_is_not_reported_as_one_without_kenlm(self, tmp_path):
        """has_kenlm() must distinguish "not built with it" from "will not load".

        It used to answer False for both, which made a packaging failure
        indistinguishable from a configuration choice — and that is precisely
        how the installed-library rpath bug would have been misread, since it
        broke loading and nothing else.
        """
        missing = tmp_path / "libctc_beam_decoder.dylib"
        assert ctc_beam_decoder.has_kenlm(library=missing) is False

        # A file that exists and is not a shared library.
        broken = tmp_path / "broken.dylib"
        broken.write_bytes(b"this is not a shared library")
        with pytest.raises(OSError):
            ctc_beam_decoder.has_kenlm(library=broken)


@pytest.mark.skipif(not ctc_beam_decoder.has_kenlm(),
                    reason="built without KenLM")
class TestWithLanguageModel:
    ALPHA, BETA = 0.2, 1.0  # config.toml's values

    def build(self, vocab, unigrams=None):
        py = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path=TINY_ARPA,
                                          unigrams=unigrams, alpha=self.ALPHA, beta=self.BETA)
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab, kenlm_model_path=TINY_ARPA,
                                                unigrams=unigrams, alpha=self.ALPHA,
                                                beta=self.BETA)
        return py, cpp

    @needs_reference
    def test_matches_over_random_inputs(self, vocab):
        py, cpp = self.build(vocab)
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            assert_same(*decode_both(py, cpp, logits), context=f"(seed {seed})")

    @needs_reference
    def test_matches_with_unigrams(self, vocab):
        """A known vocabulary changes how unseen word prefixes are scored.

        With no unigram list every prefix counts as unknown and is penalised;
        with one, only prefixes that begin no known word are. Both paths are
        reachable from config.toml, so both are checked.
        """
        py, cpp = self.build(vocab, unigrams=["a", "b", "ac", "bd", "cd"])
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            assert_same(*decode_both(py, cpp, logits), context=f"(seed {seed})")

    @needs_reference
    def test_matches_with_hotwords(self, vocab):
        """Hotwords and the language model at once.

        These interact: with a language model, a word prefix that is in the
        hotword list is credited as a hotword, and anything else falls to the
        language model's out-of-vocabulary penalty instead of scoring zero.
        """
        py, cpp = self.build(vocab, unigrams=["a", "b", "ac", "bd", "cd"])
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp.n_columns)
            assert_same(
                *decode_both(py, cpp, logits, hotwords=["ac", "abcd"], hotword_weight=10.0),
                context=f"(seed {seed})",
            )

    def test_fusion_pulls_output_toward_what_the_model_knows(self):
        """Does shallow fusion actually fuse?

        Every other language-model test here compares against pyctcdecode, and
        that cannot answer this question: a model that loaded and contributed
        nothing would make both implementations agree perfectly and both be
        wrong. Matching a reference only shows you match the reference.

        So this asks the model itself. `tiny.arpa` knows five words and four
        bigrams. If its probabilities are reaching the beam, scaled by alpha,
        then raising alpha must pull decoded output toward those words and
        those bigrams — and the effect must grow with alpha, because that is
        what alpha means. Measured over 200 random inputs:

            alpha   words it knows   bigrams it knows
              off            44.6%              11.7%
              0.0            76.4%              27.8%
              0.5            99.8%              59.1%
              1.0            99.8%              70.0%
              2.0            99.8%              81.0%

        No plumbing defect produces that curve. A wrong sign would push the
        other way, a dropped alpha would flatten it, a score that never
        reached the beam would leave it at the "off" row.

        alpha=0 still beats "off" because the penalty for a word prefix that
        starts no known word is not scaled by alpha — in the reference either.
        That penalty is doing much of the steering, which is worth knowing.
        """
        vocab = VOCABS["subword"]
        known_words = {"a", "b", "ac", "bd", "cd"}
        known_bigrams = {("a", "b"), ("b", "a"), ("a", "ac"), ("b", "bd")}

        def rates(dec, trials=200):
            in_vocab = words = in_bigram = pairs = 0
            for seed in range(trials):
                out = dec.decode(logits_for(seed, dec.n_columns), beam_width=BEAM_WIDTH).split()
                in_vocab += sum(1 for w in out if w in known_words)
                words += len(out)
                got = list(zip(out, out[1:]))
                in_bigram += sum(1 for g in got if g in known_bigrams)
                pairs += len(got)
            return in_vocab / max(words, 1), in_bigram / max(pairs, 1)

        off_words, off_bigrams = rates(ctc_beam_decoder.build_ctcdecoder(vocab))
        measured = []
        for alpha in (0.5, 1.0, 2.0):
            measured.append(rates(ctc_beam_decoder.build_ctcdecoder(
                vocab, kenlm_model_path=TINY_ARPA, alpha=alpha, beta=0.0)))

        for alpha_words, alpha_bigrams in measured:
            assert alpha_words > off_words, "fusion did not favour words the model knows"
            assert alpha_bigrams > off_bigrams, "fusion did not favour bigrams the model knows"

        bigram_rates = [b for _w, b in measured]
        assert bigram_rates == sorted(bigram_rates), (
            f"more alpha did not mean more of the model's own bigrams: {bigram_rates}"
        )

    @needs_reference
    def test_language_model_changes_the_output(self, vocab):
        cpp_plain = ctc_beam_decoder.build_ctcdecoder(vocab)
        _, cpp_lm = self.build(vocab)
        changed = 0
        for seed in range(TRIALS):
            logits = logits_for(seed, cpp_lm.n_columns)
            changed += cpp_plain.decode(logits, beam_width=BEAM_WIDTH) != \
                cpp_lm.decode(logits, beam_width=BEAM_WIDTH)
        assert changed > 0, "the language model never altered any transcript"

    @needs_reference
    def test_decode_matches(self, vocab):
        py, cpp = self.build(vocab, unigrams=["a", "b", "ac", "bd", "cd"])
        assert_decode_same(py, cpp)

    def test_missing_model_file_is_an_error(self, vocab):
        with pytest.raises(FileNotFoundError):
            ctc_beam_decoder.build_ctcdecoder(vocab, kenlm_model_path="/nonexistent/model.arpa")


class TestInputHandling:
    def test_rejects_wrong_column_count(self, vocab):
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        with pytest.raises(ValueError, match="columns"):
            cpp.decode_beams(np.zeros((4, cpp.n_columns + 1), dtype=np.float32))

    def test_rejects_non_2d_input(self, vocab):
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        with pytest.raises(ValueError, match="2-D"):
            cpp.decode_beams(np.zeros(cpp.n_columns, dtype=np.float32))

    def test_does_not_modify_the_caller_s_array(self, vocab):
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        logits = logits_for(0, cpp.n_columns)
        before = logits.copy()
        cpp.decode_beams(logits, beam_width=BEAM_WIDTH)
        assert np.array_equal(logits, before)

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="duplicate"):
            ctc_beam_decoder.build_ctcdecoder(["▁a", "▁a", "b"])

    def test_accepts_float64_input(self, vocab):
        """Callers should not have to know the library wants float32."""
        cpp = ctc_beam_decoder.build_ctcdecoder(vocab)
        logits = logits_for(0, cpp.n_columns)
        assert (cpp.decode(logits.astype(np.float64), beam_width=BEAM_WIDTH)
                == cpp.decode(logits, beam_width=BEAM_WIDTH))
