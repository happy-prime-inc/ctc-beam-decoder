// Hotword biasing, following pyctcdecode's HotwordScorer.
//
// Two scores, and the second is the one that does the work:
//
//   score(text)                counts whole-word hotword occurrences
//   score_partial(word_part)   partial credit for a prefix of a hotword
//
// The partial score is what rescues a near-miss. It applies while a word is
// still being built, so a beam spelling out the first few characters of a
// hotword is kept alive long enough to finish it. A bonus paid only on
// completed words cannot do that — by the time the word is complete the beam
// that would have spelled it has already been pruned.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ctcbd {

class HotwordScorer {
 public:
  HotwordScorer() = default;
  HotwordScorer(const std::vector<std::string> &hotwords, double weight);

  bool empty() const { return unigrams_.empty(); }

  // weight * (number of whole-word matches in text)
  double score(const std::string &text) const;

  // weight * len(token) / len(shortest hotword that starts with token), or 0
  // if no hotword starts with token. Lengths are in code points, matching
  // Python's len() on a str.
  double score_partial(const std::string &token) const;

  // Whether any hotword starts with `token` (pyctcdecode's `in` operator).
  bool has_prefix(const std::string &token) const;

 private:
  std::vector<std::string>::const_iterator prefix_begin(const std::string &token) const;

  // Sorted longest-first, so the longest alternative wins at a given position
  // — the order the reference builds its regex alternation in.
  std::vector<std::string> by_length_desc_;
  // Sorted lexically and deduplicated, so "does any hotword start with this?"
  // is a binary search rather than a scan.
  std::vector<std::string> sorted_;
  std::vector<std::string> unigrams_;
  double weight_ = 0.0;
};

// Number of Unicode code points in a UTF-8 string.
size_t utf8_length(const std::string &s);

}  // namespace ctcbd
