# ctc-beam-decoder

CTC prefix beam search with hotword biasing, in C++.

Takes a `[T, vocab+1]` array of log-probabilities and returns n-best
hypotheses with per-word frame indices and scores. Nothing model-specific:
where the logits come from is the caller's business.

## Why this exists

The reference implementation for this job in Python is
[pyctcdecode](https://github.com/kensho-technologies/pyctcdecode), which has
had no release since April 2023 and pins `numpy<2.0.0`. That pin is
precautionary rather than measured — decoding is byte-identical under numpy
2.5.1 — but it is still enforced by every resolver, and numpy 1.26 has no
Windows wheel for Python 3.13. The result is that a stale cap on an abandoned
package decides which platforms a downstream application can ship on.

The alternatives are worse rather than better: `flashlight-text` has no wheels
past Python 3.12 and no stable release since 0.0.7; `ctcdecode` is source-only
with no wheels at all; `k2` requires PyTorch. Everything else that is
maintained lives inside a training framework.

So: a small, dependency-light implementation that can be built rather than
resolved.

## Design constraints

**Behaviour identical to pyctcdecode.** Not "equivalent quality" — the same
strings. Two correct beam searches with different pruning produce different
output on the same logits, so parity means matching the pruning too
(`token_min_logp`, `beam_prune_logp`, history pruning), not only the search.

**n-best with word timings.** Downstream consumers use beam agreement across
hypotheses, and word-level frame indices, not just the top string.

**Hotword partial credit.** Boosting has to apply as a prefix is built, which
is what lets a near-miss be rescued. A whole-word bonus applied at the end
does not do the same job.

**KenLM optional.** It is LGPL, so it stays a separately linked component that
a downstream build can omit or replace.

## Status

Beam search, word timings, hotword biasing and KenLM fusion are implemented,
and match the reference on the evaluation corpus.

## Using it

```
./scripts/fetch-kenlm.sh          # skip with -DCTCBD_WITH_KENLM=OFF
cmake -S . -B build/cmake && cmake --build build/cmake
./scripts/check-install.sh        # installs to a clean prefix and loads it
```

`check-install.sh` is worth running before shipping anything. The build tree
cannot show the failure it checks for: a CMake-built library keeps an rpath
pointing at its own build directory, so it finds its neighbours however the
install is configured, and `cmake --install` strips that rpath. An installed
decoder that cannot find KenLM beside it passes every other test here.

```python
from ctc_beam_decoder import BeamSearchDecoder

dec = BeamSearchDecoder(vocab)
beams = dec.decode_beams(log_probs, beam_width=100, hotwords=["ravi", "kwame"])
text, _lm_state, word_frames, logit_score, combined_score = beams[0]
```

With a language model:

```python
dec = build_ctcdecoder(vocab, kenlm_model_path="lm/3-gram.arpa.gz",
                       alpha=0.2, beta=1.0)
```

The signature is pyctcdecode's, so swapping is a change of import.

KenLM is optional at build time. Without it the decoder still does beam
search, hotwords and the allowlist — the configuration the app ships — and
asking for a language model raises rather than quietly decoding without one.
`ctc_beam_decoder.has_kenlm()` reports which build you have.

## Verification

Parity is measured, not asserted, against 72 fixtures — 82 minutes, 67,839
frames — of spontaneous meeting speech, read speech from 40 speakers, one
paragraph across seven accents, and a name-dense passage.

The recorded reference and the logits it was made from are kept outside this
repository — generated data has a different lifecycle from the source it holds
to account, and left here it was 99.6% of the diff. That corpus is not public:
some of the audio it derives from carries a data-use agreement that has not
been settled, so nothing derived from it is published either. The tests below
need none of it.

With a checkout of that data, the comparison runs as:

```
python tools/check_parity.py \
    --logits ../ctc-decoder-parity-data/build/logits \
    --reference ../ctc-decoder-parity-data/reference
```

`tests/` needs none of that. Those decode random inputs against pyctcdecode
directly, with no model and no corpus, and are what CI runs on all three
platforms — which is where the two Windows packaging faults found in review
would have failed on the first push, and where neither could fail on a
developer's macOS machine.

Measured 2026-08-06, worst case of hotwords on and off:

| | no LM | with LM | either, allowing for token order |
|---|---|---|---|
| `decode()` identical | 69/72 | 69/72 | **72/72** |
| top-1 of n-best identical | 69/72 | 68/72 | **72/72** |
| every beam's text identical | 68/72 | 68/72 | **72/72** |
| word frame indices identical | 65/72 | 65/72 | **72/72** |
| beam scores within 1e-4 | 67/72 | 68/72 | **72/72** |

The language model is the LibriSpeech 3-gram the app ships, at its configured
`alpha=0.2, beta=1.0`.

`decode()` is measured separately from the n-best list rather than assumed to
be its first entry. The reference does not implement one in terms of the other
on equal settings — `decode()` turns history pruning on, since only the top
beam is wanted there — so a decoder can reproduce every beam and still return
a different transcript. It did, once: that defect survived a 131-fixture
comparison of everything except the call the application actually makes.

The second column needs explaining, and it is not a rounding-error excuse.

pyctcdecode collects each frame's candidate tokens in a Python set and
iterates it, so the arbitrary-but-fixed order of a CPython hash table decides
how near-ties break. Re-running pyctcdecode against *itself* with tokens
visited in ascending order instead, **9 of these 72 fixtures decode
differently, and 8 of the 9 differ in the top-1 transcript**. For those there
is no single right answer to match: both outputs are pyctcdecode. The second
column counts a fixture as matching if it agrees with either.

A decoder that reproduced the first column exactly would have to reimplement
CPython's set internals and stay pinned to them. That is a worse artifact than
this one. `check_order_sensitivity.py` in the data repository is what measures
this, and is worth re-running before trusting any parity number.

Scores are compared to 1e-4 rather than bit for bit, deliberately: they order
beams and feed the caption stability comparison, and nobody reads them. In
practice they agree far more closely than that.

### Does the language model actually do anything?

Parity cannot answer that. A model that loaded and contributed nothing would
make both implementations agree perfectly and both be wrong — matching a
reference only shows you match the reference. So it is checked separately, two
ways.

**Without any corpus**, in the test suite:
`test_fusion_pulls_output_toward_what_the_model_knows` decodes random inputs
against `tests/tiny.arpa`, which knows five words and four bigrams, and
measures how much of the output the model recognises as alpha rises:

| alpha | words it knows | bigrams it knows |
|---|---|---|
| off | 44.6% | 11.7% |
| 0.0 | 76.4% | 27.8% |
| 0.5 | 99.8% | 59.1% |
| 1.0 | 99.8% | 70.0% |
| 2.0 | 99.8% | 81.0% |

No plumbing defect produces that curve: a wrong sign pushes the other way, a
dropped alpha flattens it, a score that never reaches the beam leaves it at
the "off" row. (alpha=0 still beats "off" because the penalty for a prefix
that begins no known word is not scaled by alpha — in the reference either.)

**With the real model and corpus**, via `tools/check_language_model.py`. The
n-gram the app ships is trained on LibriSpeech text, so it is in domain on
LibriSpeech audio and out of domain everywhere else, and behaves accordingly:

| set | fixtures | changed | WER off | WER on | delta |
|---|---|---|---|---|---|
| `librispeech` | 40 | 30 | 0.0326 | **0.0299** | **−0.0027** |
| `gmu_accent` | 7 | 4 | 0.1366 | 0.1387 | +0.0021 |
| `ami_spontaneous` | 23 | 23 | 0.1974 | 0.2174 | +0.0200 |
| `live_reading` | 2 | 2 | 0.2743 | 0.3029 | +0.0286 |

Helping in domain and hurting out of it is what a working n-gram does, and it
is why `classroom-captions` ships with `enabled = false`. Sweeping alpha on the
in-domain set gives the expected inverted U — 0.0339 at alpha=0, best 0.0294 at
0.1, 0.0352 by 0.8 — which is a curve only a correctly scaled fusion produces.

### What the tests would catch

A suite tells you what passes, not what it would notice. Those are different
questions, and three defects here answered the second one badly: `decode()`
was recorded by the parity harness and never called, a hotword that split on
ASCII whitespace only agreed perfectly with a reference that was also ignoring
it, and an empty beam list made the frame comparison iterate zero times and
score as matching. Each was a check that could not fail.

`tools/mutation_audit.py` asks the other question. It applies a small,
realistic defect, rebuilds, runs the suite, and reverts. **19 mutations, 19
caught, none survived** — dropped hotword credit, merge frames taken from the
wrong duplicate, scores accumulated in float32, tokens visited in the wrong
order, pruning ignored, and for the language model: no alpha, no beta, no
unknown-word penalty, no end-of-sentence, no base-10 conversion.

Three of them exist specifically for the blind spot the rest cannot reach.
Every other mutation is caught by comparing against pyctcdecode — which is
precisely the check that missed all three real defects, because a feature
inert on *both* sides agrees perfectly. So `hotword-ascii-whitespace-only`
reintroduces a bug that shipped and agreed with the reference,
`language-model-inert` loads a model that contributes nothing, and
`decoder-returns-nothing` returns no beams to see whether any comparison is
vacuous.

Run it before trusting a green suite:

```
python tools/mutation_audit.py --list
python tools/mutation_audit.py
```

It fails when it proves nothing, which took a review comment to get right. A
harness that asks "would this fail?" can itself pass having done nothing: an
unknown `--only` selection, or an anchor that a refactor moved, used to print
`0 caught, 0 survived` and exit successfully. Since this runs in CI, the claim
above could have gone quietly hollow while the badge stayed green.

Now an unapplied mutation is a failure, and the tally says so —
`19 caught, 0 survived, 0 not exercised, 0 skipped by build`. Mutations that
only mean something in a particular build declare it, so a language-model
mutation on a build without KenLM is a *declared* skip rather than a silent
one. That distinction is what keeps the strictness honest: without it, the
first false alarm would have been fixed by relaxing the exit code.

Two behaviours differ from the reference on purpose:

- **Input normalisation happens in numpy, in the Python binding.** The library
  has `ctcbd_prepare` for callers who are not using Python, but numpy's float32
  `exp`, `log` and pairwise `sum` cannot be reproduced to the last bit, and
  being one unit out in the last place drifts beam scores enough to change
  which beams merge — measured, on a 1250-frame fixture.
- **Degenerate input returns no beams instead of raising.** pyctcdecode throws
  from `max()` on an empty sequence when a frame prunes everything away, which
  callers currently work around by decoding greedily first to check.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.
