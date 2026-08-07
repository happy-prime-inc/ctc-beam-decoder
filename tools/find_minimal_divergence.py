"""Shrink a disagreement with pyctcdecode down to something readable.

Divergences show up on 1250-frame fixtures with 1025-token vocabularies, where
nothing is inspectable. This generates small random problems instead — a
handful of tokens, a handful of frames — finds one where the two
implementations disagree, and then shrinks it by dropping frames for as long
as the disagreement survives.

What comes out is usually a few frames over five tokens, which can be worked
through by hand.

    python find_minimal_divergence.py --trials 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyctcdecode.decoder as pd
from pyctcdecode import build_ctcdecoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ctc_beam_decoder import BeamSearchDecoder  # noqa: E402

from token_order import ascending  # noqa: E402

# Small, but with every structural case the search distinguishes: word-start
# tokens, mid-word tokens, a token that is only a word marker, and the unknown
# token that is marked on both sides.
VOCABS = {
    # Word boundaries marked on the first token of a word. Includes a token
    # that is only a marker, and the unknown token that is marked both sides.
    "bpe": ["▁a", "▁b", "c", "d", "▁", "<unk>"],
    # Word boundaries as a space token of its own, the other style pyctcdecode
    # supports.
    "regular": [" ", "a", "b", "c", "d", "<unk>"],
}
BEAM_WIDTH = 5


def decode_both(dec_py, dec_cpp, logits, hotwords=None):
    pd.set = ascending()
    try:
        py = dec_py.decode_beams(logits, beam_width=BEAM_WIDTH, hotwords=hotwords,
                                 hotword_weight=10.0)
    finally:
        pd.set = set
    cpp = dec_cpp.decode_beams(logits, beam_width=BEAM_WIDTH, hotwords=hotwords,
                               hotword_weight=10.0)
    return py, cpp


def summarise(beams):
    return [(t, [(w, tuple(f)) for w, f in wf], round(float(ls), 5))
            for t, _s, wf, ls, _c in beams]


def disagree(py, cpp):
    return summarise(py) != summarise(cpp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--hotwords", nargs="*", default=None)
    ap.add_argument("--vocab", choices=sorted(VOCABS), default="bpe")
    args = ap.parse_args()

    vocab = VOCABS[args.vocab]
    dec_py = build_ctcdecoder(vocab)
    dec_cpp = BeamSearchDecoder(vocab)
    n_cols = dec_cpp.n_columns
    print(f"vocab {vocab} -> {n_cols} columns, blank at {dec_cpp.blank_id}")

    for trial in range(args.trials):
        rng = np.random.default_rng(trial)
        logits = rng.normal(0, 3, size=(args.frames, n_cols)).astype(np.float32)
        py, cpp = decode_both(dec_py, dec_cpp, logits, args.hotwords)
        if not disagree(py, cpp):
            continue

        print(f"\ntrial {trial} disagrees; shrinking")
        best = logits
        changed = True
        while changed:
            changed = False
            for drop in range(best.shape[0]):
                trimmed = np.delete(best, drop, axis=0)
                if trimmed.shape[0] == 0:
                    continue
                py2, cpp2 = decode_both(dec_py, dec_cpp, trimmed, args.hotwords)
                if disagree(py2, cpp2):
                    best = trimmed
                    changed = True
                    break

        py, cpp = decode_both(dec_py, dec_cpp, best, args.hotwords)
        print(f"\nminimal case: {best.shape[0]} frames\n")
        labels = vocab + [""] if len(vocab) < n_cols else vocab
        print("       " + "".join(f"{(l or 'blank'):>9}" for l in labels))
        for t, row in enumerate(best):
            print(f"  f{t:<3}  " + "".join(f"{v:>9.3f}" for v in row))
        print("\n  pyctcdecode:")
        for row in summarise(py):
            print(f"    {row}")
        print("\n  ctc_beam_decoder:")
        for row in summarise(cpp):
            print(f"    {row}")
        out = Path(__file__).resolve().parent.parent / "build" / "minimal_divergence.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, best)
        print(f"\n  saved to {out}")
        return

    print(f"{args.trials} trials, no disagreement")


if __name__ == "__main__":
    main()
