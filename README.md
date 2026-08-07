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

Other implementations have their own constraints: `flashlight-text` has no
wheels past Python 3.12 and no stable release since 0.0.7; `ctcdecode` is
source-only with no wheels; `k2` requires PyTorch.

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

`check-install.sh` checks something the build tree cannot. A CMake-built
library keeps a search path pointing at its own build directory, so it finds
KenLM wherever the install is configured to put it; `cmake --install` removes
that path. An installed library that cannot find KenLM beside it therefore
passes every other test here, and fails on the first machine that is not the
one that built it.

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

**n-best with word timings.** The full hypothesis list and per-word frame
indices are part of the contract, not just the top string.

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

### What "allowing for token order" means

pyctcdecode gives slightly different answers depending on something it never
meant to depend on.

At each frame it gathers the candidate tokens into a Python set and loops over
them. A set has no order you can rely on. Python picks one, it stays the same
for the same input, and it is otherwise arbitrary. When two candidate
transcripts score almost exactly the same, that arbitrary order is what decides
which of them wins.

You can watch it happen. Run pyctcdecode twice on the same audio, changing
nothing except the order it walks those tokens: **9 of the 72 fixtures come out
differently, and on 8 of those the final transcript changes.**

So on those 9 there is no single answer to match — both results are
pyctcdecode's own. The first two columns require matching the run recorded
here. The last column accepts either.

Matching the first two everywhere would mean copying how a particular version
of Python happens to order a set, and staying tied to it.

Scores are compared to within 1e-4 rather than bit for bit. Floating-point
accumulation order differs between numpy and C++, and matching it exactly is a
much larger problem than matching the output. In practice they agree far more
closely than that bound.

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

## Two differences from pyctcdecode

**Log-probabilities are normalised in Python, not C++.**

Before searching, the input is log-softmaxed and clipped. pyctcdecode does this
with numpy, and so does this library — in the Python binding, before the data
reaches C++.

Doing it in C++ would be the obvious choice and gives slightly different
answers. numpy's `exp`, `log` and `sum` are its own routines, and a C++
reimplementation lands a fraction out in the final decimal place. That sounds
harmless and is not: those values are added up across every frame, and by the
end the difference is large enough to change which hypotheses the search treats
as identical, which changes the transcript. Measured on a 1250-frame input.

C callers get `ctcbd_prepare()`, which does the same normalisation in C++. It
is close but not identical, so a C caller and a Python caller can differ in the
last digit.

**Silence returns an empty list rather than raising.**

If the input has nothing decodable in it, every candidate can be discarded and
the search ends with no hypotheses at all. pyctcdecode raises an exception in
that case. This returns an empty list, so callers can check for it rather than
guarding with a `try`.

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
