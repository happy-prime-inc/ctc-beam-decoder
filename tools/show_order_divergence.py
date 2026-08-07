"""Show exactly how the order-sensitive fixtures diverge.

`check_order_sensitivity.py` answers yes/no. This answers "by how much", which
is the number that decides whether byte-identical parity is a sensible bar or
an arbitrary one. If changing an implementation detail that has no business
affecting the answer changes a whole sentence, the bar is meaningful; if it
moves one word between two near-tied hypotheses, the output at that point is
noise and no implementation is more correct than another.

    python show_order_divergence.py --logits build/logits --keys a b c
"""

import argparse
import difflib
import json
from pathlib import Path

import numpy as np
import pyctcdecode.decoder as pd
from pyctcdecode import build_ctcdecoder

from token_order import ascending, descending

BEAM_WIDTH = 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    args = ap.parse_args()

    src = Path(args.logits).resolve()
    meta = json.loads((src / "index.json").read_text())
    dec = build_ctcdecoder(meta["vocab"])
    orders = {"native": set, "ascending": ascending(), "descending": descending()}

    for key in args.keys:
        logits = np.load(src / "logits" / f"{key}.npy")
        runs = {}
        try:
            for name, cls in orders.items():
                pd.set = cls
                runs[name] = dec.decode_beams(logits, beam_width=BEAM_WIDTH)
        finally:
            pd.set = set

        print(f"\n=== {key}  ({logits.shape[0]} frames) ===")
        base = runs["native"]
        for name, beams in runs.items():
            print(f"  {name:<11} {len(beams):>3} beams  "
                  f"top1 score {beams[0][3]:.6f} / {beams[0][4]:.6f}")
        for name, beams in runs.items():
            if name == "native":
                continue
            if beams[0][0] != base[0][0]:
                a, b = base[0][0].split(), beams[0][0].split()
                sm = difflib.SequenceMatcher(a=a, b=b)
                changed = sum(max(i2 - i1, j2 - j1)
                              for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
                print(f"\n  top-1 differs under {name}: "
                      f"{changed} of {len(a)} words, score gap "
                      f"{base[0][4] - beams[0][4]:+.6f}")
                for line in difflib.unified_diff(a, b, "native", name, lineterm="", n=2):
                    print("   " + line)
            else:
                first = next((i for i, (t, *_) in enumerate(beams)
                              if i >= len(base) or t != base[i][0]), None)
                print(f"\n  top-1 identical under {name}; "
                      f"n-best first differs at rank {first}")
                if first is not None and first < len(base):
                    print(f"    native    #{first}: {base[first][0][-90:]!r}"
                          f"  ({base[first][4]:.6f})")
                    print(f"    {name} #{first}: {beams[first][0][-90:]!r}"
                          f"  ({beams[first][4]:.6f})")


if __name__ == "__main__":
    main()
