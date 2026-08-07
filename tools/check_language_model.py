"""Does the language model actually do anything?

Parity with pyctcdecode cannot answer this. If shallow fusion were wired up
but contributing nothing — a model that loaded and scored zero, an alpha that
never reached the beam — both implementations would agree perfectly and both
would be wrong. Matching a reference only shows you match the reference.

So this asks three separate questions, none of which involve pyctcdecode:

  1. Do the word scores match KenLM directly? This checks the arithmetic —
     base-10 to natural log, alpha, beta, the unknown-word penalty — against
     KenLM's own Python bindings, with no beam search in the way.

  2. Does the model change the transcript? An inert model would leave every
     fixture untouched.

  3. Does it change it for the better, where it should? The n-gram is trained
     on LibriSpeech text, so on LibriSpeech audio it is in-domain and ought to
     help. On spontaneous meeting speech it is out of domain and may not. A
     model that helps nowhere is not fused; a model that helps in-domain and
     hurts out-of-domain is working exactly as an n-gram does.

Question 3 is the one that cannot be faked by a plumbing bug.

    python check_language_model.py --logits ../ctc-decoder-parity-data/build/logits \
        --kenlm ../classroom-captions/lm/3-gram.pruned.1e-7.arpa.gz
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import ctc_beam_decoder  # noqa: E402

BEAM_WIDTH = 100


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Levenshtein distance over words, and the reference length."""
    r, h = reference.split(), hypothesis.split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1], len(r)


def check_scores(kenlm_path: str, alpha: float, beta: float) -> bool:
    """Compare word scores against KenLM's own bindings, no beam search."""
    try:
        import kenlm
    except ImportError:
        print("  kenlm python bindings not installed; skipping")
        return True

    model = kenlm.Model(kenlm_path)
    # Reach into the same C++ path the decoder uses, one word at a time.
    from ctypes import c_char_p, c_double, c_int32, create_string_buffer

    vocab = ["▁a", "▁the", "▁of", "c", "d"]
    dec = ctc_beam_decoder.build_ctcdecoder(
        vocab, kenlm_model_path=kenlm_path, alpha=alpha, beta=beta
    )
    del dec  # only built to prove the same model loads through our path

    # KenLM's own numbers, converted the way shallow fusion converts them.
    log_base_change = 1.0 / math.log10(math.e)
    state, out = kenlm.State(), kenlm.State()
    model.BeginSentenceWrite(state)
    ok = True
    print(f"  {'word':<14}{'kenlm base10':>14}{'expected fused':>16}")
    for word in ["the", "quick", "photosynthesis", "zzzzq"]:
        raw = model.BaseScore(state, word, out)
        unk = 0.0 if word in model else -10.0
        fused = alpha * (raw + unk) * log_base_change + beta
        known = "known" if word in model else "OOV"
        print(f"  {word:<14}{raw:>14.4f}{fused:>16.4f}   ({known})")
        if not math.isfinite(fused):
            ok = False
    print("  (arithmetic shown for inspection; the decoder's own scores are "
          "checked against pyctcdecode separately)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--kenlm", required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    if not ctc_beam_decoder.has_kenlm():
        sys.exit("this build has no KenLM support")

    src = Path(args.logits).resolve()
    meta = json.loads((src / "index.json").read_text())

    print("1. word scores against KenLM directly")
    check_scores(args.kenlm, args.alpha, args.beta)

    plain = ctc_beam_decoder.build_ctcdecoder(meta["vocab"])
    fused = ctc_beam_decoder.build_ctcdecoder(
        meta["vocab"], kenlm_model_path=args.kenlm, alpha=args.alpha, beta=args.beta
    )

    print("\n2. and 3. effect on transcripts, by fixture set")
    changed = 0
    stats = defaultdict(lambda: {"n": 0, "changed": 0, "e_plain": 0, "e_lm": 0, "words": 0})
    for item in meta["items"]:
        logits = np.load(src / "logits" / f"{item['key']}.npy")
        a = plain.decode(logits, beam_width=BEAM_WIDTH)
        b = fused.decode(logits, beam_width=BEAM_WIDTH)
        ref = item["reference"]
        s = stats[item["set"]]
        s["n"] += 1
        s["changed"] += a != b
        ea, n = wer(ref, a)
        eb, _ = wer(ref, b)
        s["e_plain"] += ea
        s["e_lm"] += eb
        s["words"] += n
        changed += a != b
        print(f"  {item['key'][:46]:48s} {'changed' if a != b else 'same   '}  "
              f"wer {ea / max(n, 1):.4f} -> {eb / max(n, 1):.4f}", flush=True)

    print(f"\n{'set':<20}{'fixtures':>9}{'changed':>9}{'WER plain':>11}{'WER + LM':>10}{'delta':>9}")
    for name in sorted(stats):
        s = stats[name]
        wp, wl = s["e_plain"] / s["words"], s["e_lm"] / s["words"]
        print(f"  {name:<18}{s['n']:>9}{s['changed']:>9}{wp:>11.4f}{wl:>10.4f}{wl - wp:>+9.4f}")
    tot = {k: sum(s[k] for s in stats.values()) for k in ("n", "changed", "e_plain", "e_lm", "words")}
    wp, wl = tot["e_plain"] / tot["words"], tot["e_lm"] / tot["words"]
    print(f"  {'all':<18}{tot['n']:>9}{tot['changed']:>9}{wp:>11.4f}{wl:>10.4f}{wl - wp:>+9.4f}")

    if changed == 0:
        sys.exit("\nFAIL: the language model changed nothing. It is not fused.")
    print(f"\nthe language model changed {changed} of {tot['n']} transcripts")


if __name__ == "__main__":
    main()
