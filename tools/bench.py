"""How long each implementation takes on the same logits.

Decoding runs on every caption tick, against a forward pass of a few hundred
milliseconds, so what matters is whether the decoder is a rounding error in
that budget or a visible part of it.

    python bench.py --logits build/logits
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyctcdecode import build_ctcdecoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ctc_beam_decoder import BeamSearchDecoder  # noqa: E402

BEAM_WIDTH = 100      # the app's configured beam width
HOTWORD_WEIGHT = 10.0  # the app's configured allowlist weight


def timed(fn, logits, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(logits)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--count", type=int, default=12, help="fixtures to time")
    args = ap.parse_args()

    src = Path(args.logits).resolve()
    meta = json.loads((src / "index.json").read_text())
    py = build_ctcdecoder(meta["vocab"])
    cpp = BeamSearchDecoder(meta["vocab"])

    # A spread of lengths rather than the first N, which are all one corpus.
    items = sorted(meta["items"], key=lambda i: i["key"])
    step = max(1, len(items) // args.count)
    chosen = items[::step][: args.count]

    print(f"{'frames':>7} {'pyctcdecode':>13} {'this':>10} {'speedup':>9}")
    total_py = total_cpp = total_frames = 0.0
    for item in chosen:
        logits = np.load(src / "logits" / f"{item['key']}.npy")
        hot = None
        t_py = timed(lambda x: py.decode_beams(x, beam_width=BEAM_WIDTH, hotwords=hot,
                                               hotword_weight=HOTWORD_WEIGHT),
                     logits, args.repeats)
        t_cpp = timed(lambda x: cpp.decode_beams(x, beam_width=BEAM_WIDTH, hotwords=hot,
                                                 hotword_weight=HOTWORD_WEIGHT),
                      logits, args.repeats)
        total_py += t_py
        total_cpp += t_cpp
        total_frames += logits.shape[0]
        print(f"{logits.shape[0]:>7} {t_py*1000:>11.1f}ms {t_cpp*1000:>8.1f}ms "
              f"{t_py/t_cpp:>8.1f}x")

    print(f"\n{int(total_frames)} frames total: "
          f"pyctcdecode {total_py*1000:.0f}ms, this {total_cpp*1000:.0f}ms, "
          f"{total_py/total_cpp:.1f}x")
    print(f"per frame: pyctcdecode {total_py/total_frames*1000:.3f}ms, "
          f"this {total_cpp/total_frames*1000:.3f}ms")


if __name__ == "__main__":
    main()
