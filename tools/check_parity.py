"""Compare this decoder against the recorded pyctcdecode reference.

Reports, per configuration:

  top-1 identical      the transcript a caption viewer would actually see
  n-best identical     every beam's text, in order, via the recorded hash
  frames identical     word-level frame ranges on the top beams
  scores within 1e-4   ordering and stability comparisons use these; nobody
                       reads them

Where a fixture fails against the recorded Python-set ordering, it is checked
again against the ascending-order recording. A fixture that matches that one
differs because pyctcdecode broke a tie on the layout of a CPython hash table,
which is not a defect to fix.

The references and the logits live in ctc-decoder-parity-data, not here:

    python check_parity.py \
        --logits ../ctc-decoder-parity-data/build/logits \
        --reference ../ctc-decoder-parity-data/reference
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ctc_beam_decoder import BeamSearchDecoder  # noqa: E402

SCORE_TOLERANCE = 1e-4


def beams_digest(beams) -> str:
    h = hashlib.sha256()
    for text, _lm_state, _frames, _logit, _combined in beams:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def compare(beams, expected, decoded):
    """Compare a decode against one recorded configuration.

    `decoded` is the separate single-result call. It is checked separately
    from the n-best list because the reference does not implement one in terms
    of the other on equal settings — decode() turns history pruning on — so a
    decoder can match every beam and still return a different transcript.
    """
    top = expected["top_beams"]
    # frames and scores are compared beam by beam below, over zip(beams, top)
    # — which quietly compares nothing at all when `beams` is empty or shorter
    # than the recorded top beams. Starting them at True in that case would
    # score a decoder that returned nothing as matching on both.
    enough = len(beams) >= len(top)
    result = {
        "decode": decoded == expected["decode"],
        "top1": bool(beams) and beams[0][0] == top[0]["text"],
        "nbest": beams_digest(beams) == expected["beams_digest"],
        "n_beams": len(beams) == expected["n_beams"],
        "frames": enough,
        "scores": enough,
    }
    for got, want in zip(beams, top):
        if got[0] != want["text"]:
            result["frames"] = False
            result["scores"] = False
            break
        if [list(f) for _w, f in got[2]] != want["frames"]:
            result["frames"] = False
        # np.isclose rather than abs(a - b): every comparison against a NaN
        # is False, so a NaN score would have read as "within tolerance" and a
        # numerical collapse would have passed the parity report silently.
        if not (np.isclose(got[3], want["logit_score"], rtol=0, atol=SCORE_TOLERANCE)
                and np.isclose(got[4], want["combined_score"], rtol=0,
                               atol=SCORE_TOLERANCE)):
            result["scores"] = False
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--library", default=None, help="path to libctc_beam_decoder")
    ap.add_argument("--kenlm", default=None, help="compare the language model configurations")
    ap.add_argument("--unigrams", default=None, help="known-vocabulary file")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print the first differing transcript")
    args = ap.parse_args()

    src = Path(args.logits).resolve()
    meta = json.loads((src / "index.json").read_text(encoding="utf-8"))
    name = "pyctcdecode_reference_lm.json" if args.kenlm else "pyctcdecode_reference.json"
    ref = json.loads((Path(args.reference).resolve() / name).read_text(encoding="utf-8"))

    unigrams = None
    if args.unigrams:
        unigrams = [w.strip() for w in Path(args.unigrams).read_text(encoding="utf-8").split() if w.strip()]
    dec = BeamSearchDecoder(meta["vocab"], library=args.library,
                            kenlm_model_path=args.kenlm, unigrams=unigrams,
                            alpha=args.alpha, beta=args.beta)
    if dec.blank_id != ref["blank_id"]:
        sys.exit(f"blank id {dec.blank_id} but reference recorded {ref['blank_id']}")

    # Read the settings out of the reference rather than assume them. They used
    # to be a shared constant, which stopped working the moment the reference
    # moved to its own repository — and a reference recorded at one beam width
    # compared against a decode run at another would have looked like a
    # decoder defect.
    settings = ref["generated_with"]
    beam_width = settings["beam_width"]
    hotword_weight = settings["hotword_weight"]
    print(f"reference: pyctcdecode {settings.get('pyctcdecode')}, "
          f"numpy {settings.get('numpy')}, beam_width {beam_width}, "
          f"hotword_weight {hotword_weight}"
          + (f", kenlm alpha {settings['alpha']} beta {settings['beta']}"
             if settings.get("kenlm") else ""))

    configs = ["plain", "hotworded"]
    measures = ("decode", "top1", "nbest", "n_beams", "frames", "scores")
    tally = {c: {k: 0 for k in measures} for c in configs}
    tally_relaxed = {c: {k: 0 for k in measures} for c in configs}
    total = 0
    first_failure = None

    items = meta["items"][: args.limit] if args.limit else meta["items"]
    for item in items:
        key = item["key"]
        entry = ref["fixtures"][key]
        logits = np.load(src / "logits" / f"{key}.npy")
        total += 1
        line = [f"  {key}"]
        for config in configs:
            hot = entry["hotwords"] if config == "hotworded" else None
            beams = dec.decode_beams(
                logits, beam_width=beam_width, hotwords=hot, hotword_weight=hotword_weight
            )
            decoded = dec.decode(
                logits, beam_width=beam_width, hotwords=hot, hotword_weight=hotword_weight
            )
            got = compare(beams, entry[config], decoded)
            for k, ok in got.items():
                tally[config][k] += int(ok)
            # Second chance against the ascending-order recording, for every
            # measure: a fixture that fails one and passes the other differs
            # because pyctcdecode broke a tie on the layout of a CPython hash
            # table, which is not something to reproduce.
            alt = compare(beams, entry[config + "_ascending"], decoded)
            for k in tally_relaxed[config]:
                tally_relaxed[config][k] += int(got[k] or alt[k])
            flags = "".join(k[0].upper() if v else k[0] for k, v in got.items())
            line.append(f"{config}:{flags}")
            if first_failure is None and not (got["top1"] and got["decode"]):
                first_failure = (key, config, beams[0][0] if beams else "",
                                 entry[config]["top_beams"][0]["text"],
                                 entry[config + "_ascending"]["top_beams"][0]["text"])
        print(" ".join(line), flush=True)

    print(f"\n{total} fixtures")
    for config in configs:
        t, r = tally[config], tally_relaxed[config]
        print(f"\n  {config}                        as recorded    allowing for token order")
        for label, key in (("decode() identical", "decode"),
                           ("top-1 of n-best identical", "top1"), ("n-best identical", "nbest"),
                           ("beam count identical", "n_beams"), ("word frames identical", "frames"),
                           (f"scores within {SCORE_TOLERANCE:g}", "scores")):
            print(f"    {label:<24} {t[key]:>4}/{total}       {r[key]:>4}/{total}")

    if args.verbose and first_failure:
        key, config, got, want, alt = first_failure
        print(f"\nfirst differing top-1 — {key} [{config}]")
        print(f"  got       {got!r}")
        print(f"  reference {want!r}")
        print(f"  ascending {alt!r}")

    ok = all(tally_relaxed[c][m] == total for c in configs for m in ("decode", "top1"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
