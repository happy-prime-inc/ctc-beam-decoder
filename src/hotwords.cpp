#include "hotwords.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace ctcbd {

size_t utf8_length(const std::string &s) {
  size_t n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0) != 0x80) n++;  // count everything but continuation bytes
  }
  return n;
}

namespace {

// Decode the code point starting at `i`, returning it and its byte length.
// Malformed input yields the raw byte, which keeps this total.
std::pair<uint32_t, size_t> decode_utf8(const std::string &s, size_t i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  size_t len = 1;
  uint32_t cp = c;
  if ((c & 0x80) == 0) {
    return {cp, 1};
  } else if ((c & 0xE0) == 0xC0) {
    len = 2;
    cp = c & 0x1Fu;
  } else if ((c & 0xF0) == 0xE0) {
    len = 3;
    cp = c & 0x0Fu;
  } else if ((c & 0xF8) == 0xF0) {
    len = 4;
    cp = c & 0x07u;
  } else {
    return {cp, 1};
  }
  if (i + len > s.size()) return {c, 1};
  for (size_t k = 1; k < len; k++) {
    const unsigned char cc = static_cast<unsigned char>(s[i + k]);
    if ((cc & 0xC0) != 0x80) return {c, 1};
    cp = (cp << 6) | (cc & 0x3Fu);
  }
  return {cp, len};
}

// Whitespace as Python's str.isspace() defines it, which is what str.split()
// and the \S in the reference's word-boundary pattern both use.
//
// This is not pedantry about exotic input. Hotwords come from a list a person
// edits — a class roster, a glossary pasted out of a document — and a pasted
// non-breaking space is ordinary. Treating one as part of a word would make
// the name it appears in silently never match, which presents to a user as
// "the allowlist does not work for this name" with nothing to see.
bool is_python_space(uint32_t cp) {
  if (cp < 0x80) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20);
  }
  switch (cp) {
    case 0x85:                          // NEL
    case 0xA0:                          // no-break space
    case 0x1680:                        // ogham space mark
    case 0x2028: case 0x2029:           // line and paragraph separators
    case 0x202F:                        // narrow no-break space
    case 0x205F:                        // medium mathematical space
    case 0x3000:                        // ideographic space
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;  // en quad through hair space
  }
}

// Whether a whitespace code point starts at `i`; its byte length in `len`.
bool space_starts_at(const std::string &s, size_t i, size_t *len) {
  auto [cp, n] = decode_utf8(s, i);
  *len = n;
  return is_python_space(cp);
}

// Whether the code point *ending* at `i` is whitespace — the lookbehind.
bool space_ends_at(const std::string &s, size_t i) {
  if (i == 0) return false;
  size_t start = i - 1;
  while (start > 0 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) start--;
  return is_python_space(decode_utf8(s, start).first);
}

// Python's str.split() with no argument: split on runs of whitespace,
// dropping empties.
std::vector<std::string> split_ws(const std::string &s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t n = 1;
    while (i < s.size() && space_starts_at(s, i, &n)) i += n;
    const size_t start = i;
    while (i < s.size() && !space_starts_at(s, i, &n)) i += n;
    if (i > start) out.push_back(s.substr(start, i - start));
  }
  return out;
}

}  // namespace

HotwordScorer::HotwordScorer(const std::vector<std::string> &hotwords, double weight)
    : weight_(weight) {
  for (const std::string &entry : hotwords) {
    // Multi-word entries are scored word by word, as the reference does.
    for (const std::string &unigram : split_ws(entry)) unigrams_.push_back(unigram);
  }
  by_length_desc_ = unigrams_;
  // By code points, not bytes. The reference sorts by Python's len(), and for
  // anything non-ASCII a byte count orders differently — which changes the
  // alternation order, which decides which of two hotwords matches first.
  // Stable, so equal-length entries keep input order, as Python's sorted()
  // does.
  std::stable_sort(by_length_desc_.begin(), by_length_desc_.end(),
                   [](const std::string &a, const std::string &b) {
                     return utf8_length(a) > utf8_length(b);
                   });
  // Sorted once so prefix questions are a binary search rather than a scan of
  // every hotword. Both are asked per beam per frame; the decoder caches them
  // per distinct word part, but the first ask of each still lands here, and an
  // allowlist is a roster that can run to hundreds of names.
  sorted_ = unigrams_;
  std::sort(sorted_.begin(), sorted_.end());
  sorted_.erase(std::unique(sorted_.begin(), sorted_.end()), sorted_.end());
}

double HotwordScorer::score(const std::string &text) const {
  if (unigrams_.empty()) return 0.0;
  // Equivalent to re.findall over an alternation of the unigrams, each
  // bounded by (?<!\S) and (?!\S): scan left to right, take the first
  // alternative that matches at a whitespace-bounded position, then resume
  // after it. Matches do not overlap.
  size_t count = 0;
  size_t pos = 0;
  while (pos <= text.size()) {
    if (pos == 0 || space_ends_at(text, pos)) {
      for (const std::string &u : by_length_desc_) {
        if (u.empty() || pos + u.size() > text.size()) continue;
        if (text.compare(pos, u.size(), u) != 0) continue;
        const size_t after = pos + u.size();
        size_t n = 1;
        if (after != text.size() && !space_starts_at(text, after, &n)) continue;
        count++;
        pos = after;
        goto matched;
      }
    }
    pos++;
  matched:;
  }
  return weight_ * static_cast<double>(count);
}

// First hotword at or after `token` in sorted order, if it has `token` as a
// prefix.
std::vector<std::string>::const_iterator HotwordScorer::prefix_begin(
    const std::string &token) const {
  return std::lower_bound(sorted_.begin(), sorted_.end(), token);
}

bool HotwordScorer::has_prefix(const std::string &token) const {
  auto it = prefix_begin(token);
  return it != sorted_.end() && it->compare(0, token.size(), token) == 0;
}

double HotwordScorer::score_partial(const std::string &token) const {
  if (sorted_.empty() || token.empty()) return 0.0;
  // The reference walks a trie built from length-sorted keys and takes the
  // first key found, which is the shortest hotword with this prefix. Sorted
  // order puts every such hotword in one contiguous run, so this reads only
  // that run rather than the whole list.
  size_t shortest = std::numeric_limits<size_t>::max();
  for (auto it = prefix_begin(token);
       it != sorted_.end() && it->compare(0, token.size(), token) == 0; ++it) {
    shortest = std::min(shortest, utf8_length(*it));
  }
  if (shortest == std::numeric_limits<size_t>::max()) return 0.0;
  return weight_ * static_cast<double>(utf8_length(token)) / static_cast<double>(shortest);
}

}  // namespace ctcbd
