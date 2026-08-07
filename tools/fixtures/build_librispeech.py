#!/usr/bin/env python3
"""Build the librispeech fixture from LibriSpeech test-clean and test-other.

Why this fixture exists
-----------------------
Breadth, for decoder work rather than for accuracy claims. Comparing two
decoders means comparing them on many different shapes of probability
distribution — confident and flat, short and long, clean and noisy — and the
cheapest way to get that spread is many speakers reading many things.

`gmu_accent` has seven speakers reading one identical paragraph, which is a
narrow slice by construction. This is 40 passages from 40 different speakers
reading unrelated material.

Both splits, deliberately. test-clean is well-recorded read speech, where the
acoustic model is confident and the beam search has little to do; test-other is
harder audio, where probabilities are flatter and more hypotheses stay alive.
The second is where two decoders that differ will differ, so it is the more
useful half.

What this is not
----------------
A quality benchmark for the language model. The n-gram this project ships is
trained on LibriSpeech text, so measuring word error rate here with that model
enabled compares a model against its own training domain and flatters it. For
comparing two decoders on identical logits that does not matter at all — it is
the same input to both — but do not read a WER from it and believe it.

Source
------
`openslr/librispeech_asr` on Hugging Face. CC BY 4.0 — attribution required,
commercial use permitted, no share-alike.

    python build_fixture.py     # needs `datasets` and `soundfile`
"""

from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
SAMPLE_RATE = 16_000

# Split between the two conditions. test-other is weighted higher because
# flatter probabilities exercise the search harder — a confident distribution
# barely branches.
PASSAGES_PER_SPLIT = {"test.clean": 15, "test.other": 25}

SCAN_RECORDS = 3000  # per split, before grouping
TARGET_S = 60.0      # passage length to aim for
MIN_S = 25.0         # below this a passage is not worth keeping
MIN_WORDS = 60
GAP_S = 0.25         # silence inserted between utterances of a passage


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop anything not a letter or apostrophe.

    LibriSpeech transcripts are already uppercase words with apostrophes and
    nothing else, so this is mostly a lowercase — but it is applied anyway so
    that a reference from here is directly comparable with one from another
    set built by a different script.
    """
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    return " ".join(text.split())


def build_split(split: str, want: int) -> list[dict]:
    print(f"streaming up to {SCAN_RECORDS} utterances from {split}...")
    ds = datasets.load_dataset("openslr/librispeech_asr", split=split, streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for i, rec in enumerate(ds):
        if i >= SCAN_RECORDS:
            break
        raw = (rec.get("audio") or {}).get("bytes")
        if not raw:
            continue
        grouped[str(rec["speaker_id"])].append({"text": rec["text"], "bytes": raw})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} utterances, {len(grouped)} speakers")

    built = []
    # Longest-first, so the speakers with the most material come first and the
    # selection is stable: raising `want` later appends rather than reshuffles.
    for speaker in sorted(grouped, key=lambda s: (-len(grouped[s]), s)):
        if len(built) >= want:
            break
        chunks: list[np.ndarray] = []
        words: list[str] = []
        total_s = 0.0
        for u in grouped[speaker]:
            audio, sr = sf.read(io.BytesIO(u["bytes"]), dtype="float32")
            if sr != SAMPLE_RATE:
                continue
            if chunks:
                chunks.append(np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32))
                total_s += GAP_S
            chunks.append(audio)
            total_s += len(audio) / SAMPLE_RATE
            text = normalize(u["text"])
            if text:
                words.extend(text.split())
            if total_s >= TARGET_S:
                break
        if total_s < MIN_S or len(words) < MIN_WORDS:
            continue

        name = f"{split.replace('.', '_')}_{speaker}"
        sf.write(HERE / f"{name}.wav", np.concatenate(chunks), SAMPLE_RATE)
        built.append({
            "name": f"{name}.wav",
            "split": split,
            "speaker_id": speaker,
            "duration_s": round(total_s, 2),
            "words": len(words),
            "reference": " ".join(words),
        })
        print(f"  {name}.wav  {total_s:.1f}s  {len(words)} words")
    return built


def main() -> None:
    manifest = []
    for split, want in PASSAGES_PER_SPLIT.items():
        manifest.extend(build_split(split, want))

    references = {m["name"]: m["reference"] for m in manifest}
    (HERE / "references.json").write_text(json.dumps(references, indent=1, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps({
        "source": "openslr/librispeech_asr",
        "licence": "CC BY 4.0",
        "sample_rate": SAMPLE_RATE,
        "target_s": TARGET_S,
        "gap_s": GAP_S,
        "passages": [{k: v for k, v in m.items() if k != "reference"} for m in manifest],
    }, indent=1))

    total = sum(m["duration_s"] for m in manifest)
    print(f"\n{len(manifest)} passages, {total / 60:.1f} minutes")


if __name__ == "__main__":
    main()
