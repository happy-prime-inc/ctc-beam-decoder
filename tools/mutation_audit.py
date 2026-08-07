"""Break the decoder on purpose and see whether the tests notice.

A test suite tells you what passes. It does not tell you what it would catch,
and those are different questions — three defects have already survived this
one: `decode()` was recorded but never called, an inert hotword agreed with an
inert reference, and an empty beam list scored as matching. Each was a check
that could not fail.

So this asks the other question directly. It applies a small, realistic defect
to the source, rebuilds, runs the suite, and reverts. A mutation that survives
is a behaviour nothing is checking.

    python mutation_audit.py            # all of them
    python mutation_audit.py --list

Run from a clean tree: it edits files in place and restores them with git.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name, file, find, replace, what it breaks[, requires])
#
# `requires` marks a mutation that only means something in a particular build.
# A declared skip is fine; an undeclared one is a broken audit, and the
# difference is the whole point of the exit status below.
MUTATIONS = [
    ("hotword-partial-credit", "src/hotwords.cpp",
     "  if (sorted_.empty() || token.empty()) return 0.0;",
     "  return 0.0;  // MUTANT",
     "partial credit for a prefix of a hotword"),

    ("hotword-whole-word", "src/hotwords.cpp",
     "  if (unigrams_.empty()) return 0.0;\n  // Equivalent to re.findall",
     "  return 0.0;  // MUTANT\n  // Equivalent to re.findall",
     "credit for a completed hotword"),

    ("merge-keeps-first-frames", "src/beam_search.cpp",
     "      target.text_frames = std::move(b.text_frames);",
     "      // MUTANT: keep the first duplicate's frames",
     "which duplicate's word timings survive a merge"),

    ("scores-in-float32", "src/beam_search.cpp",
     "          nb.score = b.score + static_cast<double>(p_char);",
     "          nb.score = static_cast<float>(b.score) + p_char;  // MUTANT",
     "accumulating beam scores in double"),

    ("tokens-descending", "src/beam_search.cpp",
     "    for (int32_t i = 0; i < n_cols; i++) {\n      if (col[i] >= opts.token_min_logp || i == max_idx) candidates.push_back(i);\n    }",
     "    for (int32_t i = n_cols - 1; i >= 0; i--) {  // MUTANT\n      if (col[i] >= opts.token_min_logp || i == max_idx) candidates.push_back(i);\n    }",
     "the order candidate tokens are visited in"),

    ("ignore-token-min-logp", "src/beam_search.cpp",
     "      if (col[i] >= opts.token_min_logp || i == max_idx) candidates.push_back(i);",
     "      candidates.push_back(i);  // MUTANT",
     "skipping tokens below the floor"),

    ("ignore-beam-prune", "src/beam_search.cpp",
     "    const double cutoff = max_score + opts.beam_prune_logp;",
     "    const double cutoff = -std::numeric_limits<double>::infinity();  // MUTANT",
     "dropping beams far behind the best"),

    ("word-end-off-by-one", "src/beam_search.cpp",
     "          if (!is_blank) nb.part_end = frame + 1;",
     "          if (!is_blank) nb.part_end = frame;  // MUTANT",
     "the end frame of a repeated token"),

    ("decode-without-prune-history", "python/ctc_beam_decoder/__init__.py",
     "            prune_history=True,",
     "            prune_history=False,  # MUTANT",
     "history pruning inside decode()"),

    ("no-input-normalisation", "python/ctc_beam_decoder/__init__.py",
     "        x = np.ascontiguousarray(_prepare(log_probs), dtype=np.float32)",
     "        x = np.ascontiguousarray(log_probs, dtype=np.float32)  # MUTANT",
     "normalising the input the way the reference does"),

    ("lm-drop-beta", "src/language_model.cpp",
     "  return impl_->opts.alpha * lm_score * kLogBaseChangeFactor + impl_->opts.beta;",
     "  return impl_->opts.alpha * lm_score * kLogBaseChangeFactor;  // MUTANT",
     "the word-insertion bonus", "kenlm"),

    ("lm-drop-alpha", "src/language_model.cpp",
     "  return impl_->opts.alpha * lm_score * kLogBaseChangeFactor + impl_->opts.beta;",
     "  return lm_score * kLogBaseChangeFactor + impl_->opts.beta;  // MUTANT",
     "scaling the language model by alpha", "kenlm"),

    ("lm-drop-unknown-penalty", "src/language_model.cpp",
     "  if (unknown) lm_score += impl_->opts.unk_score_offset;",
     "  // MUTANT: no penalty for unknown words",
     "the penalty for a word the model has never seen", "kenlm"),

    ("lm-drop-end-of-sentence", "src/language_model.cpp",
     "  if (is_last_word && impl_->opts.score_boundary) {",
     "  if (false) {  // MUTANT",
     "charging for ending the sentence", "kenlm"),

    ("lm-drop-prefix-penalty", "src/language_model.cpp",
     "  double unk_score = impl_->opts.unk_score_offset * is_oov;",
     "  double unk_score = 0.0;  // MUTANT",
     "the penalty for a prefix that starts no known word", "kenlm"),

    # These three target the blind spot the others cannot reach. Every
    # mutation above is caught by comparing against pyctcdecode — which is
    # exactly the check that missed the three defects that actually shipped,
    # because a feature inert on both sides agrees perfectly. So: break things
    # in ways that leave the two implementations agreeing, or that make a
    # comparison vacuous, and see whether anything still notices.
    ("hotword-ascii-whitespace-only", "src/hotwords.cpp",
     "  if (cp < 0x80) {\n    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20);\n  }",
     "  if (cp < 0x80) {\n    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20);\n  }\n  return false;  // MUTANT: ASCII whitespace only",
     "splitting hotwords on non-ASCII whitespace (shipped broken; agreed with the reference)"),

    ("language-model-inert", "src/language_model.cpp",
     "  return impl_->opts.alpha * lm_score * kLogBaseChangeFactor + impl_->opts.beta;",
     "  return 0.0;  // MUTANT: loaded, wired up, contributing nothing",
     "the language model contributing anything at all", "kenlm"),

    ("decoder-returns-nothing", "src/beam_search.cpp",
     "  std::vector<OutputBeam> out;\n  out.reserve(beams.size());",
     "  std::vector<OutputBeam> out;\n  return out;  // MUTANT: no beams at all",
     "returning any beams (a vacuous comparison would score this as matching)"),

    ("no-log-base-conversion", "src/language_model.cpp",
     "const double kLogBaseChangeFactor = 1.0 / std::log10(std::exp(1.0));",
     "const double kLogBaseChangeFactor = 1.0;  // MUTANT",
     "converting KenLM's base-10 scores to natural log", "kenlm"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="run one mutation by name")
    ap.add_argument("--pytest", default="pytest")
    args = ap.parse_args()

    if args.list:
        for m in MUTATIONS:
            name, path, _f, _r, what = m[:5]
            needs = f"  [needs {m[5]}]" if len(m) > 5 else ""
            print(f"  {name:<28} {path:<34} {what}{needs}")
        return

    dirty = run("git status --porcelain").stdout.strip()
    if dirty:
        sys.exit(f"working tree is not clean; refusing to edit in place:\n{dirty}")

    known = {m[0] for m in MUTATIONS}
    if args.only and args.only not in known:
        sys.exit(f"no mutation called {args.only!r}; --list shows the names")

    # Which builds are available decides which mutations mean anything.
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import ctc_beam_decoder
        have = {"kenlm"} if ctc_beam_decoder.has_kenlm() else set()
    except Exception as e:  # the library has to be built before auditing it
        sys.exit(f"could not load the decoder to check its build: {e}")

    caught, survived, not_exercised, declined = [], [], [], []
    chosen = [m for m in MUTATIONS if not args.only or m[0] == args.only]
    for m in chosen:
        name, rel, find, replace, what = m[:5]
        needs = m[5] if len(m) > 5 else None
        if needs and needs not in have:
            declined.append((name, needs))
            print(f"  {name:<28} skipped   needs a {needs} build")
            continue
        path = ROOT / rel
        original = path.read_text()
        if find not in original:
            # A stale anchor mutates nothing and proves nothing. Reported as a
            # failure because the alternative is an audit that goes quietly
            # hollow as the source moves under it.
            not_exercised.append((name, f"anchor no longer in {rel}"))
            print(f"  {name:<28} NOT RUN   anchor no longer in {rel}")
            continue
        path.write_text(original.replace(find, replace, 1))
        try:
            build = run("cmake --build build/cmake -j8")
            if build.returncode != 0:
                not_exercised.append((name, "did not compile"))
                print(f"  {name:<28} NOT RUN   did not compile")
                continue
            result = run(f"{args.pytest} tests/ -q -x")
            failed = result.returncode != 0
            (caught if failed else survived).append((name, what))
            print(f"  {name:<28} {'caught' if failed else 'SURVIVED':<9} {what}")
        finally:
            path.write_text(original)
    run("cmake --build build/cmake -j8")

    print(f"\n{len(caught)} caught, {len(survived)} survived, "
          f"{len(not_exercised)} not exercised, {len(declined)} skipped by build")

    if not chosen or not (caught or survived):
        # An audit that ran nothing is not an audit that found nothing.
        sys.exit("\nno mutation was applied — this proves nothing")

    problems = False
    if survived:
        print("\nnothing checks these:")
        for name, what in survived:
            print(f"  {name:<28} {what}")
        problems = True
    if not_exercised:
        print("\nthese never ran, so the numbers above are incomplete:")
        for name, why in not_exercised:
            print(f"  {name:<28} {why}")
        problems = True
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
