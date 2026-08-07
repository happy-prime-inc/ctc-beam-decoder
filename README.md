# ctc-beam-decoder

CTC prefix beam search with hotword biasing and optional KenLM fusion, in C++.

Takes a `[T, vocab]` array of log-probabilities and returns n-best hypotheses
with per-word frame indices and scores. Nothing model-specific: where the
logits come from is your business.

It is a drop-in replacement for
[pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) — same
constructor, same method signatures, same five-tuple results — that can be
built rather than resolved.

```python
from ctc_beam_decoder import build_ctcdecoder

decoder = build_ctcdecoder(vocab)
text = decoder.decode(log_probs, hotwords=["ravi", "kwame"], hotword_weight=10.0)

beams = decoder.decode_beams(log_probs, beam_width=100)
text, _lm_state, word_frames, logit_score, combined_score = beams[0]
```

## Why this exists

pyctcdecode is the reference implementation for this job in Python. It has had
no release since April 2023 and pins `numpy<2.0.0`.

That pin is precautionary rather than measured — decoding is byte-identical
under numpy 1.26.4 and 2.5.1 across the whole corpus below — but every resolver
enforces it, and numpy 1.26 publishes no Windows wheel for Python 3.13. So pip
compiles numpy from source there, slowly, and the result fails its own
floating-point checks on import. A stale cap on an unmaintained package decides
which platforms a downstream project can ship on.

Working around it needs a resolver override *and* a `--no-deps` install for one
package — two mechanisms telling two resolvers to disregard an upstream
constraint. That is fragile, and when it misbehaves it looks like a bad model
rather than a bad install.

The alternatives are worse rather than better: `flashlight-text` has no wheels
past Python 3.12 and no stable release since 0.0.7; `ctcdecode` is source-only
with no wheels at all; `k2` requires PyTorch. Everything else that is
maintained lives inside a training framework.

## Scope

This reproduces pyctcdecode's behaviour. It is deliberately **not** a platform
for new decoding features — the value on offer is that output does not change,
and every addition erodes it. Bug reports and portability fixes are welcome;
proposals for new search strategies, scoring schemes or output formats will be
declined, kindly.

## Using it

Prebuilt bundles for macOS (arm64), Linux (x64) and Windows (x64) are attached
to each release. Each contains the shared library, the KenLM library it loads,
the Python binding and the C header. Put that directory on `PYTHONPATH` and
import it — the binding finds its libraries beside itself.

To build from source you need CMake 3.16+, a C++17 compiler, and Python with
numpy:

```
./scripts/fetch-kenlm.sh          # skip with -DCTCBD_WITH_KENLM=OFF
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
./scripts/check-install.sh build/cmake   # installs to a clean prefix and loads it
```

`check-install.sh` is worth running before shipping anything built here. A
CMake-built library keeps an rpath pointing at its own build directory, so it
finds its neighbours however the install is configured, and `cmake --install`
strips that rpath. An installed library that cannot find KenLM beside it
passes every other test in this repository.

With a language model:

```python
decoder = build_ctcdecoder(vocab, kenlm_model_path="3-gram.arpa.gz",
                           alpha=0.2, beta=1.0)
```

KenLM is optional at build time. Without it the decoder still does beam search
and hotword biasing, and asking for a language model raises rather than quietly
decoding without one. `ctc_beam_decoder.has_kenlm()` reports which build you
have. It is LGPL and linked dynamically; see `NOTICE`.

## Design constraints

**Behaviour identical to pyctcdecode.** Not "equivalent quality" — the same
strings. Two correct beam searches with different pruning produce different
output on the same logits, so parity means matching the pruning too
(`token_min_logp`, `beam_prune_logp`, history pruning), not only the search.

**n-best with word timings.** Consumers use beam agreement across hypotheses,
and word-level frame indices, not only the top string.

**Hotword partial credit.** Boosting applies as a prefix is built, which is
what lets a near-miss be rescued. A whole-word bonus paid at the end cannot do
the same job: by the time the word is complete, the beam that would have
spelled it has already been pruned.

## Verification

Parity is measured, not asserted, against 72 fixtures — 82 minutes, 67,839
frames — of spontaneous meeting speech, read speech from 40 speakers, one
paragraph across seven accents, and a name-dense passage.

Worst case of hotwords on and off, with and without a LibriSpeech 3-gram at
`alpha=0.2, beta=1.0`:

| | no LM | with LM | either, allowing for token order |
|---|---|---|---|
| `decode()` identical | 69/72 | 69/72 | **72/72** |
| top-1 of n-best identical | 69/72 | 68/72 | **72/72** |
| every beam's text identical | 68/72 | 68/72 | **72/72** |
| word frame indices identical | 65/72 | 65/72 | **72/72** |
| beam scores within 1e-4 | 67/72 | 68/72 | **72/72** |

`decode()` is measured separately from the n-best list rather than assumed to
be its first entry. The reference does not implement one in terms of the other
on equal settings — `decode()` turns history pruning on, since only the top
beam is wanted there — so a decoder can reproduce every beam and still return a
different transcript. This one did, once.

### That third column

pyctcdecode collects each frame's candidate tokens in a Python set and iterates
it, so the arbitrary-but-fixed order of a CPython hash table decides how
near-ties break. Re-running pyctcdecode against *itself* with tokens visited in
ascending order instead, **9 of these 72 fixtures decode differently, and 8 of
the 9 differ in the top-1 transcript.** For those there is no single right
answer to match: both outputs are pyctcdecode. The third column counts a
fixture as matching if it agrees with either.

A decoder that reproduced the first column exactly would have to reimplement
CPython's set internals and stay pinned to them. That is a worse artifact than
this one.

Scores are compared to 1e-4 rather than bit for bit, deliberately: they order
beams and feed downstream stability comparisons, and nobody reads them. In
practice they agree far more closely.

The corpus itself is not published. Some of the audio it derives from carries a
data-use agreement that has not been settled, so nothing derived from it is
released either. **The test suite needs none of it.**

### Does the language model actually do anything?

Parity cannot answer that. A model that loaded and contributed nothing would
make both implementations agree perfectly and both be wrong — matching a
reference only shows you match the reference.

So it is checked directly, with no corpus, in
`test_fusion_pulls_output_toward_what_the_model_knows`. A tiny committed ARPA
knows five words and four bigrams; the test decodes random inputs and measures
how much of the output the model recognises as alpha rises:

| alpha | words it knows | bigrams it knows |
|---|---|---|
| off | 44.6% | 11.7% |
| 0.0 | 76.4% | 27.8% |
| 0.5 | 99.8% | 59.1% |
| 1.0 | 99.8% | 70.0% |
| 2.0 | 99.8% | 81.0% |

No plumbing defect produces that curve: a wrong sign pushes the other way, a
dropped alpha flattens it, a score that never reaches the beam leaves it on the
"off" row. (alpha=0 still beats "off" because the penalty for a word prefix
that begins no known word is not scaled by alpha — in the reference either.
That penalty does much of the steering.)

### What the tests would catch

A suite tells you what passes, not what it would notice.
`tools/mutation_audit.py` asks the second question: it applies a small,
realistic defect, rebuilds, runs the suite, and reverts. **19 mutations, 19
caught** — dropped hotword credit, merge frames taken from the wrong duplicate,
scores accumulated in float32, tokens visited in the wrong order, pruning
ignored, and for the language model: no alpha, no beta, no unknown-word
penalty, no end-of-sentence, no base-10 conversion.

Three of them exist for the blind spot the rest cannot reach. Every other
mutation is caught by comparing against pyctcdecode — which is exactly the
check that missed the real defects found in review, because a feature inert on
*both* sides agrees perfectly.

It fails when it proves nothing: an unknown selection, an anchor a refactor
moved, or a mutant that no longer compiles are all failures, because an audit
that ran nothing is not an audit that found nothing.

```
python tools/mutation_audit.py --list
python tools/mutation_audit.py
```

## Two deliberate differences from the reference

- **Input normalisation happens in numpy, in the Python binding.** The library
  has `ctcbd_prepare` for callers who are not using Python, but numpy's float32
  `exp`, `log` and pairwise `sum` cannot be reproduced to the last bit, and
  being one unit out in the last place drifts beam scores enough to change
  which beams merge — measured, on a 1250-frame input.
- **Degenerate input returns no beams instead of raising.** pyctcdecode throws
  from `max()` on an empty sequence when a frame prunes everything away.

## Tests

```
pip install -r requirements-dev.txt
pip install -r requirements-reference.txt   # the comparison tests need it
pytest tests/ -q
```

47 tests, no model and no corpus required. They decode random inputs against
pyctcdecode and require the same answers, across both vocabulary styles, with
and without hotwords, with and without a known vocabulary.

Without `requirements-reference.txt` the comparison tests skip and the rest
still run — which is how CI covers Windows, where installing pyctcdecode means
building `numpy<2` from source for ten minutes and getting one that crashes on
import.

## Licence

Apache-2.0, matching pyctcdecode, whose behaviour this reproduces and whose
source was used as reference. See `LICENSE` and `NOTICE`.
