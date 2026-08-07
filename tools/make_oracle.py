"""Record what pyctcdecode does, so a C++ implementation can be held to it.

This is the acceptance test for the whole project. It runs the reference
implementation over stored logits and writes down exactly what came out; any
replacement has to reproduce it.

Per fixture and per configuration:

  - the full detail of the top beams — text, per-word frame ranges, scores —
    which is what a failing comparison needs in order to be diagnosable
  - a hash over *every* beam's text, so full n-best parity is checked without
    storing 100 beams for 131 fixtures

The n-best list matters as much as the winner: downstream, beam agreement
across hypotheses is what decides when a caption word stops changing, so a
decoder that gets the top string right and the rest wrong is not a
replacement.

Each configuration is recorded twice, once with pyctcdecode visiting candidate
tokens in Python-set order (what ships today) and once in ascending index
order. On four of the 131 fixtures those disagree — see
check_order_sensitivity.py — so having both is what distinguishes a real
defect in an implementation from pyctcdecode tie-breaking on the layout of a
CPython hash table.

    python make_oracle.py --logits build/logits --out reference

It reads a directory produced by whatever runs your acoustic model:

    build/logits/index.json          {"vocab": [...], "blank_id": N,
                                      "items": [{"key": ..., "set": ...,
                                                 "reference": "..."}, ...]}
    build/logits/logits/<key>.npy    float32 [frames, columns]

`columns` is not always `len(vocab)`: a vocabulary with no blank in it gets one
appended, and the array needs a column for it. See the README for which labels
count as a blank.

Nothing here cares where the log-probabilities came from — one array per
utterance and the vocabulary they were produced with is the whole input.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyctcdecode.decoder as pd
from pyctcdecode import build_ctcdecoder

from token_order import ascending

# What the reference is recorded at. check_parity.py reads these back out of
# the file rather than assuming them, so a reference made at one beam width
# cannot be compared against a decode run at another.
BEAM_WIDTH = 100
HOTWORD_WEIGHT = 10.0

TOP_BEAMS_RECORDED = 10


def hotwords_for(reference: str, n: int = 5) -> list[str]:
    """A deterministic hotword list per fixture: its longest distinct words.

    Derived rather than fixed so every fixture exercises boosting on words that
    actually occur in it — a hotword the audio never contains tests nothing.
    Lowercase because pyctcdecode matches case-sensitively against an
    all-lowercase vocabulary, and a capitalised entry silently never fires.
    """
    words = {w for w in reference.split() if len(w) >= 4}
    return sorted(words, key=lambda w: (-len(w), w))[:n]


def beams_digest(beams) -> str:
    """Hash of every beam's text, in order. Catches n-best divergence."""
    h = hashlib.sha256()
    for text, _lm_state, _frames, _logit, _combined in beams:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def write_reference(path, reference):
    """One line per fixture, rather than one line per number.

    `json.dumps(indent=...)` puts every frame index on its own line, which
    turned a 7 MB file into 567,000 of them — enough that the recorded
    reference is 99.6% of this repository's diff and a reviewer cannot see the
    code past it. Compact within a fixture and newline-separated between them
    keeps the useful granularity, which is "this fixture changed", and loses
    the useless one, which is "this integer changed".
    """
    fixtures = reference.pop("fixtures")
    head = json.dumps(reference, indent=1)[:-2]  # drop the closing "\n}"
    body = ",\n".join(
        f"  {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}"
        for k, v in fixtures.items()
    )
    path.write_text(f'{head},\n "fixtures": {{\n{body}\n }}\n}}\n', encoding="utf-8")


def record(dec, logits, hotwords, top_n=TOP_BEAMS_RECORDED):
    # Passed exactly as decoder.py passes them: an empty allowlist arrives as
    # None, and hotword_weight is supplied either way.
    kw = {"hotwords": hotwords or None, "hotword_weight": HOTWORD_WEIGHT}
    beams = dec.decode_beams(logits, beam_width=BEAM_WIDTH, **kw)
    top = []
    for text, _lm_state, word_frames, logit_score, combined in beams[:top_n]:
        top.append({
            # No separate word list: a beam's words are its text split on
            # single spaces, which is how the reference pairs them with
            # frames. Recording both stores the transcript twice.
            "text": text,
            "frames": [[int(s), int(e)] for _w, (s, e) in word_frames],
            "logit_score": round(float(logit_score), 6),
            "combined_score": round(float(combined), 6),
        })
    return {
        "decode": dec.decode(logits, **kw),
        "n_beams": len(beams),
        "beams_digest": beams_digest(beams),
        "top_beams": top,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", required=True, help="directory written by dump_logits.py")
    ap.add_argument("--out", required=True, help="directory to write reference JSON into")
    ap.add_argument("--kenlm", default=None,
                    help="ARPA or binary language model; records the fusion configurations "
                         "into pyctcdecode_reference_lm.json instead")
    ap.add_argument("--unigrams", default=None,
                    help="known-vocabulary file, one word per line. Omitting it is not the "
                         "same as an empty one — with no vocabulary every word prefix counts "
                         "as unknown")
    ap.add_argument("--alpha", type=float, default=0.2, help="config.toml's value")
    ap.add_argument("--beta", type=float, default=1.0, help="config.toml's value")
    args = ap.parse_args()
    if not args.kenlm and (args.unigrams or ap.get_default("alpha") != args.alpha
                           or ap.get_default("beta") != args.beta):
        ap.error("--unigrams/--alpha/--beta need --kenlm; without one they do nothing "
                 "but would be recorded into the reference as though they had")

    src = Path(args.logits).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src / "index.json").read_text(encoding="utf-8"))
    unigrams = None
    if args.unigrams:
        unigrams = [w.strip() for w in Path(args.unigrams).read_text(encoding="utf-8").split() if w.strip()]
    if args.kenlm:
        dec = build_ctcdecoder(meta["vocab"], kenlm_model_path=args.kenlm,
                               unigrams=unigrams, alpha=args.alpha, beta=args.beta)
    else:
        dec = build_ctcdecoder(meta["vocab"])

    import pyctcdecode as _p
    reference = {
        "generated_with": {
            "pyctcdecode": getattr(_p, "__version__", "0.5.0"),
            "numpy": np.__version__,
            "beam_width": BEAM_WIDTH,
            "hotword_weight": HOTWORD_WEIGHT,
            **({"kenlm": args.kenlm, "unigrams": args.unigrams,
                "alpha": args.alpha, "beta": args.beta} if args.kenlm else {}),
        },
        "blank_id": meta["blank_id"],
        "fixtures": {},
    }
    ascending_set = ascending()

    for item in meta["items"]:
        logits = np.load(src / "logits" / f"{item['key']}.npy")
        hot = hotwords_for(item["reference"])
        entry = {
            "set": item["set"],
            "frames": int(logits.shape[0]),
            "hotwords": hot,
            "plain": record(dec, logits, None),
            "hotworded": record(dec, logits, hot),
        }
        # Only the top beam in full for the ascending variant: it exists to
        # identify order noise, not to be diffed line by line.
        try:
            pd.set = ascending_set
            entry["plain_ascending"] = record(dec, logits, None, top_n=1)
            entry["hotworded_ascending"] = record(dec, logits, hot, top_n=1)
        finally:
            pd.set = set
        reference["fixtures"][item["key"]] = entry
        print(f"  {item['key']}", flush=True)

    n_fixtures = len(reference["fixtures"])
    path = out / ("pyctcdecode_reference_lm.json" if args.kenlm
                  else "pyctcdecode_reference.json")
    # write_reference consumes reference["fixtures"], so the count is taken
    # before the call rather than after it.
    write_reference(path, reference)
    kb = path.stat().st_size / 1024
    print(f"\n{n_fixtures} fixtures -> {path} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
