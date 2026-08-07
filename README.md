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

### pip

```bash
pip install ctc-beam-decoder
```

Wheels are published for macOS (arm64), Linux (x64) and Windows (x64). There is
no source distribution, deliberately: an sdist would build from source, and a
from-source dependency chain breaking a platform is the reason this project
exists. `pip install` either finds a wheel for your platform or fails clearly.

The wheel carries the shared libraries inside the package, so nothing else needs
installing and no paths need setting.

**Python 3.9 and up.** The binding is pure `ctypes` with no C extension, so one
wheel serves every version — and the release workflow installs and imports that
wheel on 3.9, 3.11, 3.13 and 3.14 rather than assuming it. A version bound
asserted instead of measured is what made this project necessary; it would be a
poor joke to repeat it here.

**Linux compatibility.** The Linux wheel is built on Ubuntu 22.04 and links the
C++ runtime statically, so it runs on glibc 2.35 and later — 22.04, Debian 12,
RHEL 9 and newer. v0.1.0 was built on a newer toolchain and would not load on
any of those.

### From a release

The bundles are still published for consumers that pin binaries by checksum
rather than installing from an index, which is what `classroom-captions` does.
`ctc_beam_decoder` is a pure-Python module that loads a shared library through
`ctypes`, so using a bundle means putting one directory on `PYTHONPATH` — the
module and the library it loads travel together, and it finds the library
itself.

Bundles for macOS (arm64), Linux (x64) and Windows (x64) are attached to each
release. Unpack one and point at it:

```
tar xzf ctc-beam-decoder-v0.1.0-macos-arm64.tar.gz
export PYTHONPATH=$PWD/ctc-beam-decoder-v0.1.0-macos-arm64
python -c "import ctc_beam_decoder; print(ctc_beam_decoder.has_kenlm())"
```

The directory holds the Python module, the decoder library, the KenLM library
it loads, the C header and the licences. Nothing else is needed and nothing is
installed.

### From source

Needs CMake 3.16+, a C++17 compiler, and Python with numpy:

```
./scripts/fetch-kenlm.sh          # skip with -DCTCBD_WITH_KENLM=OFF
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
```

That leaves the libraries in `build/cmake/` and the module in `python/`. From
the repository root:

```
PYTHONPATH=python python -c "import ctc_beam_decoder; print(ctc_beam_decoder.has_kenlm())"
```

The module looks for its library next to itself, then one directory up, then in
`build/cmake` — which is why a source tree works without copying anything
about. To use it from elsewhere, either copy the two libraries next to
`python/ctc_beam_decoder/`, or set `CTC_BEAM_DECODER_LIB` to the library's full
path.

### Before shipping a build

```
./scripts/check-install.sh build/cmake
```

It installs to a clean prefix and loads the result from there, which is a check
the build tree cannot perform. A CMake-built library keeps a search path
pointing at its own build directory, so it finds KenLM wherever the install
puts it; `cmake --install` removes that path. An installed library that cannot
find KenLM beside it therefore passes every other test here, and fails on the
first machine that is not the one that built it.

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

The corpus is not distributed here — it is 44 MB of intermediate arrays and a
few hundred megabytes of audio — but the method is, and most of the material
is public. 63 of the 72 fixtures are built from LibriSpeech and the AMI Meeting
Corpus, both CC BY 4.0, by scripts that stream them from Hugging Face. The
other 9 are a set whose licence has not been checked and two private
recordings, so those are not redistributable.

**What you can reproduce, and what you cannot.** Be clear about this before
starting.

The comparison itself is fully here, and it works on any CTC model. Run your
model over your audio, save the output, and describe it:

```
build/logits/index.json        {"vocab": [...], "blank_id": N,
                                "items": [{"key": ..., "set": ...,
                                           "reference": "transcript"}, ...]}
build/logits/logits/<key>.npy  float32 [frames, columns]
```

Three things to get right, because getting them wrong fails in confusing ways:

- **`columns` is not always `len(vocab)`.** pyctcdecode appends a CTC blank to
  a vocabulary that has none, so the array needs a column for it. If your
  labels already include the blank, `columns == len(vocab)`. Recognised as a
  blank: the empty string, a `<pad>`/`[PAD]`-style token, and — for
  vocabularies that are not sub-word — `_`. Otherwise it is `len(vocab) + 1`,
  blank last.
  The corpus above is the second case: 1024 labels, 1025 columns.
- **`blank_id` is the blank's index in that column space**, so 1024 in the
  example.
- **`reference` is the ground-truth transcript**, used to pick per-fixture
  hotwords. Leave it empty and the hotword half of the comparison has nothing
  to boost.

Raw model outputs are fine — both this library and pyctcdecode log-softmax
their input, so there is no need to normalise first.

Then:

```
python tools/make_oracle.py  --logits build/logits --out reference
python tools/check_parity.py --logits build/logits --reference reference
```

`make_oracle.py` records what pyctcdecode does; `check_parity.py` compares this
decoder against that recording. Neither knows or cares what produced the
arrays. If you already use pyctcdecode you already have logits in this shape,
give or take an `index.json`.

### Which model the numbers came from

One, and it is worth being specific rather than leaving "a CTC model" to do the
work: **Parakeet CTC 1.1b**, the `q4_k` GGUF from
[`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf),
run through [parakeet.cpp](https://github.com/mudler/parakeet.cpp) v0.5.0. A
1024-token sub-word vocabulary with the blank appended, so every figure above
describes that vocabulary and that model's probability distributions.

Nothing has been measured on another acoustic model at this scale. The tests
cover both vocabulary styles pyctcdecode supports, on random inputs, so the
code paths are exercised — but "72/72 on a different model" is not a claim
anyone has earned yet.

**So the exact table is not reproducible from this repository alone.** The
audio is: `tools/fixtures/build_librispeech.py` and `tools/fixtures/build_ami.py`
rebuild the 63 CC BY 4.0 fixtures. The model is public. But the code that runs it and
writes out log-probabilities belongs to the application this was extracted
from, and is not here — you would be writing that part.

What *is* reproducible here, in two commands, is the same comparison against
your own model. That is the more useful thing anyway: it tells you whether this
decoder matches pyctcdecode on the distributions you actually decode, rather
than on someone else's.

### Where the disagreements are

If you want to go straight to the interesting cases rather than decode 82
minutes, these are the fixtures where pyctcdecode disagrees with itself
depending on token order. `check_order_sensitivity.py` finds them; these are
the ones it found here:

| fixture | source | top-1 changes |
|---|---|---|
| `EN2002c_MEE073` | AMI, meeting EN2002c, speaker MEE073 | yes |
| `EN2002d_FEO072` | AMI, meeting EN2002d, speaker FEO072 | yes |
| `EN2002d_MEE071` | AMI, meeting EN2002d, speaker MEE071 | yes |
| `ES2004a_FEE016` | AMI, meeting ES2004a, speaker FEE016 | yes |
| `test_clean_260` | LibriSpeech test-clean, speaker 260 | yes |
| `test_clean_7176` | LibriSpeech test-clean, speaker 7176 | yes |
| `test_other_4852` | LibriSpeech test-other, speaker 4852 | yes |
| `test_other_6432` | LibriSpeech test-other, speaker 6432 | yes |

A ninth fixture, one of the private recordings, differs too but only below the
top hypothesis, so it is not listed above.

`show_order_divergence.py` prints how far apart any of them are.

**The test suite needs none of this.**

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
