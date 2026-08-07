#!/usr/bin/env python3
"""Build the ami_spontaneous fixture from the AMI Meeting Corpus.

Why this fixture exists
-----------------------
Every other scoreable fixture here is read speech. gmu_accent is a rehearsed
passage, live_reading is a script read aloud, sap_severity is clinical prompted
speech. The only genuinely spontaneous audio we had was the ai_in_bc meeting,
which has no reference transcript and so can only be scored by agreement rather
than accuracy.

That gap has real consequences. The language model was added to the pipeline to
clean up caption stutters and half-words — artifacts of disfluent speech — and on
2026-08-04 it measured as making no difference on gmu_accent (0.062 vs 0.064) or
on read5 (0.240 vs 0.234). Both are read speech, where there are no stutters to
clean up. The LM's claimed benefit was invisible on every fixture we could score,
by construction.

Source
------
edinburghcstr/ami on Hugging Face, `ihm` configuration (individual headset
microphone), test split. CC BY 4.0 — attribution required, commercial use
permitted, no share-alike.

IHM rather than SDM deliberately: headset audio is clean, which isolates
spontaneity as the variable under test instead of confounding it with far-field
noise. An sdm variant would be the right way to test room mics later, and is a
different question.

What this script constructs, and how it differs from raw AMI
-----------------------------------------------------------
The HF release is segmented into individual utterances of roughly 0.4-1.4
seconds. Fed to the pipeline directly, those are far too short to exercise
anything we care about — no VAD decisions, no window management, no commitment
behaviour. So this script groups utterances by (meeting, speaker), sorts them
into their original time order, and concatenates them into passages.

**The inter-utterance gaps are capped.** On a headset channel, the silence
between one speaker's utterances is mostly other people talking, which would make
a passage largely silence. Capping the gap at MAX_GAP_S yields a continuous
single-speaker stream with natural short pauses — much closer to one person
talking at length, which is what the product actually faces. This makes the audio
a *construction*, not raw AMI: it is spontaneous speech from a real meeting with
the waiting removed. Turn-taking behaviour cannot be studied on it.

Transcripts in this release are uppercase and unpunctuated. They are lowercased
here; punctuation is absent from the source rather than stripped, which happens
to match what our CTC vocabulary emits.
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

# Pulled from the stream before grouping. Records arrive unordered, so a speaker's
# utterances are scattered; this needs to be generous enough that several speakers
# accumulate enough material to reach TARGET_S.
SCAN_RECORDS = 6000

MAX_GAP_S = 0.6      # inter-utterance silence kept, in seconds -- see module docstring
TARGET_S = 100.0     # passage length to aim for
MIN_S = 60.0         # below this a passage is not worth keeping
# Every qualifying (meeting, speaker) group, rather than a handful. The old
# cap of 4 was chosen to keep the fixture small, which costs nothing to
# relax: the audio is gitignored and rebuilt on demand, so more passages are
# build time rather than repository size.
#
# Selection is longest-first, so raising this only appends — the passages an
# earlier build produced keep their names and their references, and per-file
# measurements stay comparable. A directory-level average over the set does
# change, because it is now an average over more material.
MAX_PASSAGES = 100
MIN_WORDS = 120      # a passage of nothing but "YEAH" and "MM-HMM" measures nothing


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace. AMI marks unclear speech and
    vocal sounds in a way that survives naive processing as stray tokens, so
    strip the bracketed markup rather than let it into a reference transcript.
    """
    text = re.sub(r"[\[<][^\]>]*[\]>]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return " ".join(text.split())


def main() -> None:
    print(f"streaming up to {SCAN_RECORDS} utterances from edinburghcstr/ami (ihm, test)...")
    ds = datasets.load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for i, rec in enumerate(ds):
        if i >= SCAN_RECORDS:
            break
        raw = (rec.get("audio") or {}).get("bytes")
        if not raw:
            continue
        grouped[(rec["meeting_id"], rec["speaker_id"])].append(
            {
                "begin": rec["begin_time"],
                "end": rec["end_time"],
                "text": rec["text"],
                "bytes": raw,
            }
        )
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} utterances, {len(grouped)} (meeting, speaker) groups")

    # Longest-first: the groups with most material give the most contiguous passages.
    order = sorted(grouped, key=lambda k: -len(grouped[k]))

    manifest, references = [], {}
    for meeting_id, speaker_id in order:
        if len(manifest) >= MAX_PASSAGES:
            break
        utts = sorted(grouped[(meeting_id, speaker_id)], key=lambda u: u["begin"])

        chunks: list[np.ndarray] = []
        words: list[str] = []
        total_s = 0.0
        prev_end: float | None = None
        for u in utts:
            audio, sr = sf.read(io.BytesIO(u["bytes"]), dtype="float32")
            if sr != SAMPLE_RATE:
                continue
            if prev_end is not None:
                gap = min(max(u["begin"] - prev_end, 0.0), MAX_GAP_S)
                if gap > 0:
                    chunks.append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))
                    total_s += gap
            chunks.append(audio)
            total_s += len(audio) / SAMPLE_RATE
            text = normalize(u["text"])
            if text:
                words.extend(text.split())
            prev_end = u["end"]
            if total_s >= TARGET_S:
                break

        if total_s < MIN_S or len(words) < MIN_WORDS:
            continue

        name = f"{meeting_id}_{speaker_id}"
        sf.write(HERE / f"{name}.wav", np.concatenate(chunks), SAMPLE_RATE)
        references[f"{name}.wav"] = " ".join(words)
        manifest.append(
            {
                "file": f"{name}.wav",
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "duration_s": round(total_s, 2),
                "words": len(words),
                "utterances": len([u for u in utts]),
            }
        )
        print(f"  wrote {name}.wav  {total_s:.1f}s  {len(words)} words")

    (HERE / "manifest.json").write_text(
        json.dumps(
            {
                "source": "edinburghcstr/ami",
                "config": "ihm",
                "split": "test",
                "license": "CC BY 4.0",
                "sample_rate": SAMPLE_RATE,
                "constructed": {
                    "max_gap_s": MAX_GAP_S,
                    "target_s": TARGET_S,
                    "note": "consecutive same-speaker utterances concatenated with gaps capped",
                },
                "passages": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "references.json").write_text(json.dumps(references, indent=2) + "\n")
    print(f"\n{len(manifest)} passages, {sum(p['duration_s'] for p in manifest):.0f}s total")


if __name__ == "__main__":
    main()
