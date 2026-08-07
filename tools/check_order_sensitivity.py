"""Does pyctcdecode's output depend on the iteration order of a Python set?

This decides how hard the port is.

At each frame, pyctcdecode picks the candidate tokens with
`set(np.where(...)[0]) | {max_idx}` and iterates that set. A Python set of
integers has a deterministic but arbitrary order, and that order decides the
order beams are appended, which decides dict insertion order in `_merge_beams`,
which decides how `heapq.nlargest` breaks ties. If any of that reaches the
output, a C++ implementation would have to reproduce CPython's set internals to
be identical — a genuinely unpleasant requirement, and one pinned to a private
detail of the interpreter.

If it does not reach the output, the port is free to iterate however it likes.

    python check_order_sensitivity.py --logits build/logits
"""

import argparse
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
    ap.add_argument("--limit", type=int, default=0, help="stop after N fixtures")
    args = ap.parse_args()

    src = Path(args.logits).resolve()
    meta = json.loads((src / "index.json").read_text(encoding="utf-8"))
    dec = build_ctcdecoder(meta["vocab"])
    orders = {"native": set, "ascending": ascending(), "descending": descending()}

    items = meta["items"][: args.limit] if args.limit else meta["items"]
    differing = []
    for item in items:
        logits = np.load(src / "logits" / f"{item['key']}.npy")
        out = {}
        try:
            for name, cls in orders.items():
                pd.set = cls
                out[name] = [text for text, *_ in dec.decode_beams(logits, beam_width=BEAM_WIDTH)]
        finally:
            pd.set = set

        top_same = len({v[0] for v in out.values()}) == 1
        nbest_same = len({tuple(v) for v in out.values()}) == 1
        if top_same and nbest_same:
            print(f"  same    {item['key']}", flush=True)
        else:
            differing.append((item["key"], top_same, nbest_same))
            print(f"  DIFFERS {item['key']}  top1={top_same} nbest={nbest_same}", flush=True)

    print(f"\n{len(items)} fixtures, {len(differing)} order-sensitive")
    if differing:
        print("Token iteration order reaches the output.")
        for key, top, nbest in differing:
            print(f"  {key}: top1 identical={top}, n-best identical={nbest}")
        print("Run show_order_divergence.py on those to see how far apart they are.")
    else:
        print("Token iteration order does not reach the output.")


if __name__ == "__main__":
    main()
