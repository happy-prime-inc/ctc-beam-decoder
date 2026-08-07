// CTC prefix beam search, following pyctcdecode's BeamSearchDecoderCTC.
//
// The aim here is not "a correct beam search" but "the same beam search".
// Two correct implementations that prune differently produce different
// transcripts on the same audio, so the details below that look like
// pedantry — the order candidates are visited in, which beam's frame indices
// survive a merge, what precision a score is accumulated in — are the parts
// that decide whether output matches.
//
// Three of those are worth knowing about before changing anything:
//
//   Precision.  Log-probabilities arrive as float32 and beam scores
//   accumulate in double, which is not a choice but what the reference does:
//   it adds a float32 array element to a Python float, and numpy promotes
//   two scalars to float64. Accumulating in float32 instead drifts a few
//   units in the last place over a thousand frames, which is enough to change
//   which beams merge and so which word timings come back.
//
//   Order.  The reference collects each frame's candidate tokens in a Python
//   set, whose iteration order is arbitrary but deterministic, and that order
//   reaches the output on four of 131 evaluation fixtures. We visit tokens in
//   ascending index order, which agrees with the reference on the top beam
//   for every fixture measured and differs only in low-ranked n-best entries.
//   See tools/check_order_sensitivity.py.
//
//   force_next_break.  Only reachable in the sub-word style. It is a single
//   variable shared across every beam and
//   every token in a frame, and it persists across frames. That is a quirk of
//   the reference rather than a design, but it is observable, so it is
//   reproduced rather than tidied.

#include "beam_search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace ctcbd {

namespace {

constexpr const char *kBpeToken = "\xE2\x96\x81";      // U+2581 LOWER ONE EIGHTH BLOCK
constexpr const char *kUnkBpeToken = "\xE2\x96\x81\xE2\x81\x87\xE2\x96\x81";  // ▁⁇▁
constexpr const char *kUnkToken = "\xE2\x81\x87";                              // ⁇
constexpr double kMinTokenClipP = 1e-15;

bool starts_with(const std::string &s, const char *p) {
  size_t n = std::strlen(p);
  return s.size() >= n && s.compare(0, n, p) == 0;
}

bool ends_with(const std::string &s, const char *p) {
  size_t n = std::strlen(p);
  return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

// Python's s[:-1] — drop one code point, not one byte, and tolerate "".
std::string drop_last_codepoint(const std::string &s) {
  size_t i = s.size();
  while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) i--;
  return i > 0 ? s.substr(0, i - 1) : std::string();
}

// <unk>/[UNK] and friends.
bool is_unk_label(const std::string &s) {
  if (s.size() < 5) return false;
  char open = s.front(), close = s.back();
  if (!((open == '<' && close == '>') || (open == '[' && close == ']'))) return false;
  std::string inner = s.substr(1, s.size() - 2);
  if (inner.size() != 3) return false;
  for (char &c : inner) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return inner == "unk";
}

// <pad>/[PAD], which the reference reads as the CTC blank.
bool is_pad_label(const std::string &s) {
  if (s.size() != 5) return false;
  char open = s.front(), close = s.back();
  if (!((open == '<' && close == '>') || (open == '[' && close == ']'))) return false;
  std::string inner = s.substr(1, 3);
  for (char &c : inner) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return inner == "pad";
}

// pyctcdecode's _sum_log_scores: log-sum-exp of two log scores.
double sum_log_scores(double a, double b) {
  return (a >= b) ? a + std::log(1 + std::exp(b - a)) : b + std::log(1 + std::exp(a - b));
}

// Word frame ranges, appended to but never mutated, so beams share their
// history instead of copying a vector each frame. Stored newest-first.
struct FrameNode {
  int32_t start, end;
  std::shared_ptr<const FrameNode> prev;
};
using FrameList = std::shared_ptr<const FrameNode>;

FrameList frames_push(const FrameList &prev, int32_t s, int32_t e) {
  return std::make_shared<const FrameNode>(FrameNode{s, e, prev});
}

std::vector<std::pair<int32_t, int32_t>> frames_to_vector(const FrameList &list) {
  std::vector<std::pair<int32_t, int32_t>> out;
  for (const FrameNode *n = list.get(); n != nullptr; n = n->prev.get()) {
    out.emplace_back(n->start, n->end);
  }
  std::reverse(out.begin(), out.end());
  return out;
}

// Interned transcripts. Beam texts are long, repeat heavily between beams,
// and change only when a word completes; interning turns the merge key into
// three integers and a short string.
class TextPool {
 public:
  TextPool() { intern(""); }

  int32_t intern(const std::string &s) {
    auto it = index_.find(s);
    if (it != index_.end()) return it->second;
    int32_t id = static_cast<int32_t>(texts_.size());
    texts_.push_back(s);
    index_.emplace(s, id);
    return id;
  }

  const std::string &get(int32_t id) const { return texts_[static_cast<size_t>(id)]; }

  // pyctcdecode's _merge_tokens: whitespace-safe join of a transcript and a
  // completed word.
  int32_t append_word(int32_t text_id, const std::string &word) {
    if (word.empty()) return text_id;
    const std::string &text = get(text_id);
    if (text.empty()) return intern(word);
    return intern(text + " " + word);
  }

 private:
  std::vector<std::string> texts_;
  std::unordered_map<std::string, int32_t> index_;
};

struct Beam {
  int32_t text_id = 0;
  // The transcript before this beam's most recent word was added, and that
  // word. Only the language model needs them: it scores a word given the
  // context it follows, so it has to see the two separately even though the
  // search only ever uses them joined.
  int32_t parent_text_id = 0;
  std::string next_word;
  std::string word_part;
  int32_t last_id = -1;  // -1 is the reference's None
  FrameList text_frames;
  int32_t part_start = -1, part_end = -1;  // NULL_FRAMES
  double score = 0.0;
  double lm_score = 0.0;  // score + hotwords; recomputed each frame
};

struct MergeKey {
  int32_t text_id;
  int32_t last_id;
  std::string word_part;

  bool operator==(const MergeKey &o) const {
    return text_id == o.text_id && last_id == o.last_id && word_part == o.word_part;
  }
};

struct MergeKeyHash {
  size_t operator()(const MergeKey &k) const {
    size_t h = std::hash<std::string>()(k.word_part);
    h ^= static_cast<size_t>(k.text_id) * 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h ^= static_cast<size_t>(k.last_id + 1) * 0xC2B2AE3D27D4EB4FULL + (h << 6) + (h >> 2);
    return h;
  }
};

// Merge beams that have become identical, summing their scores.
//
// Two details here are load-bearing and neither is obvious. A merged beam
// keeps the *last* duplicate's frame history and language-model fields, not
// the first — the reference overwrites its dictionary entry wholesale and
// carries only the score forward — while its position in the list stays where
// the *first* duplicate put it, because that is where the dictionary key was
// inserted. Frames come from one beam and ordering from another.
//
// Position matters because it decides how ties break downstream; frames
// matter because they become word timings.
//
// Taking the last duplicate's parent_text_id/next_word looks wrong and is
// not. Beams merge on the transcript they have *become*, so duplicates can
// have reached it by different splits — "a b" + nothing, or "a" + "b" — and
// the language model scores those differently. Keeping the first beam's split
// instead would be defensible in isolation and would diverge from the
// reference, which is the thing being reproduced. The parent is always
// already in the cache: it is some beam's transcript from the previous frame,
// and every beam is scored before the frame ends.
void merge_beams(std::vector<Beam> &beams) {
  std::unordered_map<MergeKey, size_t, MergeKeyHash> seen;
  seen.reserve(beams.size() * 2);
  std::vector<Beam> merged;
  merged.reserve(beams.size());
  for (Beam &b : beams) {
    MergeKey key{b.text_id, b.last_id, b.word_part};
    auto it = seen.find(key);
    if (it == seen.end()) {
      seen.emplace(std::move(key), merged.size());
      merged.push_back(std::move(b));
    } else {
      Beam &target = merged[it->second];
      const double combined = sum_log_scores(target.score, b.score);
      target.text_frames = std::move(b.text_frames);
      target.parent_text_id = b.parent_text_id;
      target.next_word = std::move(b.next_word);
      target.part_start = b.part_start;
      target.part_end = b.part_end;
      target.score = combined;
    }
  }
  beams = std::move(merged);
}

// heapq.nlargest(n, beams, key) — the top n by score, ties resolved toward
// whichever came first.
void sort_and_trim(std::vector<Beam> &beams, int32_t beam_width) {
  std::stable_sort(beams.begin(), beams.end(),
                   [](const Beam &a, const Beam &b) { return a.lm_score > b.lm_score; });
  if (static_cast<int32_t>(beams.size()) > beam_width) {
    beams.resize(static_cast<size_t>(beam_width));
  }
}

std::vector<std::string> split_spaces(const std::string &s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && s[i] == ' ') i++;
    size_t start = i;
    while (i < s.size() && s[i] != ' ') i++;
    if (i > start) out.push_back(s.substr(start, i - start));
  }
  return out;
}

// Drop beams that agree over all the history a language model can still see.
// Faster, at the cost of n-best diversity — which is why it stays off
// wherever the n-best list is used for anything.
void prune_history_beams(std::vector<Beam> &beams, const TextPool &pool, int32_t lm_order) {
  const size_t keep = static_cast<size_t>(std::max(1, lm_order - 1));
  std::vector<Beam> kept;
  kept.reserve(beams.size());
  std::unordered_set<std::string> seen;
  seen.reserve(beams.size() * 2);
  for (Beam &b : beams) {
    std::vector<std::string> words = split_spaces(pool.get(b.text_id));
    std::string key;
    for (size_t i = words.size() > keep ? words.size() - keep : 0; i < words.size(); i++) {
      key += words[i];
      key += '\x1f';
    }
    key += b.word_part;
    key += '\x1f';
    key += std::to_string(b.last_id);
    if (seen.insert(std::move(key)).second) {
      kept.push_back(std::move(b));
    }
  }
  beams = std::move(kept);
}

}  // namespace

Decoder::Decoder(const std::vector<std::string> &labels) {
  labels_ = labels;
  is_bpe_ = false;
  for (const std::string &s : labels_) {
    if (starts_with(s, "##") || starts_with(s, kBpeToken)) {
      is_bpe_ = true;
      break;
    }
  }
  for (const std::string &s : labels_) {
    if (starts_with(s, "##")) {
      throw std::invalid_argument("'##'-style sub-word vocabularies are not supported");
    }
  }
  // Normalisation, following pyctcdecode's Alphabet. The two styles differ in
  // what a word boundary is: a marker attached to the first token of a word,
  // or a space token of its own.
  if (!is_bpe_) {
    // A vocabulary that writes its space as '|' or its blank as '_'.
    const bool has_space = std::find(labels_.begin(), labels_.end(), " ") != labels_.end();
    auto bar = std::find(labels_.begin(), labels_.end(), "|");
    if (bar != labels_.end() && !has_space) *bar = " ";
  }
  for (std::string &s : labels_) {
    if (is_pad_label(s)) s = "";
  }
  bool has_blank = std::find(labels_.begin(), labels_.end(), "") != labels_.end();
  if (!is_bpe_ && !has_blank) {
    auto underscore = std::find(labels_.begin(), labels_.end(), "_");
    if (underscore != labels_.end()) {
      *underscore = "";
      has_blank = true;
    }
  }
  if (!has_blank) labels_.push_back("");
  for (std::string &s : labels_) {
    if (is_unk_label(s)) s = is_bpe_ ? kUnkBpeToken : kUnkToken;
  }
  for (size_t i = 0; i < labels_.size(); i++) {
    if (labels_[i].empty()) {
      blank_id_ = static_cast<int32_t>(i);
      break;
    }
  }
  // Labels are compared by identity during the search, so duplicates would
  // silently merge two tokens into one.
  std::vector<std::string> sorted = labels_;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    throw std::invalid_argument("vocabulary contains duplicate labels");
  }
}

void prepare_log_probs(float *x, int32_t n_frames, int32_t n_columns) {
  const float lo = static_cast<float>(std::log(kMinTokenClipP));
  for (int32_t t = 0; t < n_frames; t++) {
    float *row = x + static_cast<size_t>(t) * n_columns;
    float max_v = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_columns; i++) max_v = std::max(max_v, row[i]);
    if (!std::isfinite(max_v)) max_v = 0.0f;
    double sum = 0.0;
    for (int32_t i = 0; i < n_columns; i++) sum += std::exp(static_cast<double>(row[i] - max_v));
    const float log_sum = static_cast<float>(std::log(sum));
    for (int32_t i = 0; i < n_columns; i++) {
      float v = (row[i] - max_v) - log_sum;
      row[i] = std::min(0.0f, std::max(lo, v));
    }
  }
}

std::vector<OutputBeam> Decoder::decode(const float *log_probs, int32_t n_frames,
                                        const std::vector<std::string> &hotwords,
                                        const DecodeOptions &opts) const {
  const int32_t n_cols = n_columns();
  const HotwordScorer scorer(hotwords, opts.hotword_weight);
  TextPool pool;

  std::vector<Beam> beams(1);  // the empty beam
  std::vector<Beam> next;
  std::vector<int32_t> candidates;
  bool force_next_break = false;

  // Language model state, carried for the whole decode.
  //
  // Cached by transcript rather than by beam, which is the reference's design
  // and not an optimisation bolted on: many beams share a transcript, and a
  // word is scored once for all of them. Each entry also holds the KenLM
  // context that transcript ends in, so the next word can be scored against
  // it without replaying the sentence.
  struct LmEntry {
    double lm_hw_score;  // language model plus whole-word hotword credit
    double raw_score;    // language model alone, so extensions can accumulate
    LmState state;
  };
  struct LmKeyHash {
    size_t operator()(const std::pair<int32_t, bool> &k) const {
      return (static_cast<size_t>(k.first) << 1) | static_cast<size_t>(k.second);
    }
  };
  std::unordered_map<std::pair<int32_t, bool>, LmEntry, LmKeyHash> lm_cache;
  std::unordered_map<std::string, double> partial_cache;

  // Whole-word hotword credit, cached by transcript.
  //
  // The score counts hotword occurrences in a beam's whole transcript, so it
  // only changes when a word completes — but it was being recomputed for
  // every beam on every frame, scanning a string that grows all decode. That
  // is quadratic in the length of the audio, and it went unnoticed because
  // the corpus it was measured on was mostly clips of a few seconds. On a
  // 1300-frame passage hotwords cost 9.5x plain decoding; at 569 frames,
  // 4.5x. The shape of that ratio is the bug.
  //
  // Beams sharing a transcript share the answer, and most beams do.
  std::unordered_map<int32_t, double> hotword_cache;
  auto hotword_score = [&](int32_t text_id) {
    auto it = hotword_cache.find(text_id);
    if (it == hotword_cache.end()) {
      it = hotword_cache.emplace(text_id, scorer.score(pool.get(text_id))).first;
    }
    return it->second;
  };
  // Partial-word credit is cached by the partial itself, which is short and
  // repeats heavily across beams.
  std::unordered_map<std::string, double> hotword_partial_cache;
  auto hotword_partial = [&](const std::string &word_part) {
    if (word_part.empty()) return 0.0;
    auto it = hotword_partial_cache.find(word_part);
    if (it == hotword_partial_cache.end()) {
      it = hotword_partial_cache.emplace(word_part, scorer.score_partial(word_part)).first;
    }
    return it->second;
  };

  if (lm_) {
    lm_cache[{0, false}] = LmEntry{0.0, 0.0, lm_->start_state()};
  }

  auto apply_scores = [&](std::vector<Beam> &bs, bool is_eos) {
    if (!lm_) {
      for (Beam &b : bs) {
        b.lm_score = scorer.empty()
                         ? b.score
                         : b.score + hotword_score(b.text_id) + hotword_partial(b.word_part);
      }
      return;
    }
    for (Beam &b : bs) {
      const std::pair<int32_t, bool> key{b.text_id, is_eos};
      auto it = lm_cache.find(key);
      if (it == lm_cache.end()) {
        // The parent transcript was scored on an earlier frame, or is the
        // empty one seeded above.
        const LmEntry &parent = lm_cache.at({b.parent_text_id, false});
        LmEntry entry{0.0, 0.0, parent.state};
        const double word_score = lm_->score(parent.state, b.next_word, is_eos, &entry.state);
        entry.raw_score = parent.raw_score + word_score;
        entry.lm_hw_score = entry.raw_score + hotword_score(b.text_id);
        it = lm_cache.emplace(key, std::move(entry)).first;
      }
      double lm_score = it->second.lm_hw_score;
      if (!b.word_part.empty()) {
        auto pit = partial_cache.find(b.word_part);
        if (pit == partial_cache.end()) {
          // A hotword prefix is credited as a hotword; anything else is
          // judged by the language model, which charges for prefixes that
          // start no word it knows.
          const double v = scorer.has_prefix(b.word_part)
                               ? scorer.score_partial(b.word_part)
                               : lm_->score_partial_token(b.word_part);
          pit = partial_cache.emplace(b.word_part, v).first;
        }
        lm_score += pit->second;
      }
      b.lm_score = b.score + lm_score;
    }
  };

  // Drop beams far behind the best, then keep the best `beam_width`.
  auto prune = [&](std::vector<Beam> &bs) {
    double max_score = -std::numeric_limits<double>::infinity();
    for (const Beam &b : bs) max_score = std::max(max_score, b.lm_score);
    const double cutoff = max_score + opts.beam_prune_logp;
    bs.erase(std::remove_if(bs.begin(), bs.end(),
                            [&](const Beam &b) { return !(b.lm_score >= cutoff); }),
             bs.end());
    sort_and_trim(bs, opts.beam_width);
  };

  for (int32_t frame = 0; frame < n_frames; frame++) {
    const float *col = log_probs + static_cast<size_t>(frame) * n_cols;

    // Tokens worth extending: everything above the floor, plus the argmax
    // whether or not it clears it.
    int32_t max_idx = 0;
    for (int32_t i = 1; i < n_cols; i++) {
      if (col[i] > col[max_idx]) max_idx = i;
    }
    candidates.clear();
    for (int32_t i = 0; i < n_cols; i++) {
      if (col[i] >= opts.token_min_logp || i == max_idx) candidates.push_back(i);
    }

    next.clear();
    for (int32_t idx : candidates) {
      const float p_char = col[idx];
      const std::string &token = labels_[static_cast<size_t>(idx)];
      const bool is_blank = (idx == blank_id_);
      for (const Beam &b : beams) {
        if (is_blank || b.last_id == idx) {
          // Blank, or the same token repeated: the transcript does not grow.
          Beam nb = b;
          nb.parent_text_id = b.text_id;
          nb.next_word.clear();
          nb.last_id = idx;
          if (!is_blank) nb.part_end = frame + 1;
          nb.score += static_cast<double>(p_char);
          next.push_back(std::move(nb));
        } else if (is_bpe_ && (starts_with(token, kBpeToken) || force_next_break)) {
          // A word boundary. Whatever was being built becomes a word.
          force_next_break = false;
          // Both tests are against the original token, not the stripped one.
          // It matters for the bare word marker: stripping its leading marker
          // leaves nothing to strip from the end, but the reference still
          // sees a token that ends in one and still sets force_next_break.
          std::string clean = token;
          if (starts_with(token, kBpeToken)) clean = clean.substr(std::strlen(kBpeToken));
          if (ends_with(token, kBpeToken)) {
            clean = drop_last_codepoint(clean);
            force_next_break = true;
          }
          Beam nb;
          nb.parent_text_id = b.text_id;
          nb.next_word = b.word_part;
          nb.text_id = pool.append_word(b.text_id, b.word_part);
          nb.word_part = clean;
          nb.last_id = idx;
          nb.text_frames = b.word_part.empty()
                               ? b.text_frames
                               : frames_push(b.text_frames, b.part_start, b.part_end);
          nb.part_start = frame;
          nb.part_end = frame + 1;
          nb.score = b.score + static_cast<double>(p_char);
          next.push_back(std::move(nb));
        } else if (!is_bpe_ && token == " ") {
          // A word boundary written as a token of its own. The word that was
          // being built is finished and nothing is started in its place.
          Beam nb;
          nb.parent_text_id = b.text_id;
          nb.next_word = b.word_part;
          nb.text_id = pool.append_word(b.text_id, b.word_part);
          nb.last_id = idx;
          nb.text_frames = b.word_part.empty()
                               ? b.text_frames
                               : frames_push(b.text_frames, b.part_start, b.part_end);
          nb.score = b.score + static_cast<double>(p_char);
          next.push_back(std::move(nb));
        } else {
          // Mid-word: the token extends the word being built.
          Beam nb = b;
          nb.parent_text_id = b.text_id;
          nb.next_word.clear();
          nb.word_part += token;
          nb.last_id = idx;
          if (nb.part_start < 0) nb.part_start = frame;
          nb.part_end = frame + 1;
          nb.score += static_cast<double>(p_char);
          next.push_back(std::move(nb));
        }
      }
    }

    beams.swap(next);
    merge_beams(beams);
    apply_scores(beams, /*is_eos=*/false);
    prune(beams);
    if (opts.prune_history) {
      prune_history_beams(beams, pool, lm_ ? lm_->order() : 1);
    }
  }

  // Close off whatever word each beam was still building.
  for (Beam &b : beams) {
    b.parent_text_id = b.text_id;
    b.next_word = b.word_part;
    if (!b.word_part.empty()) {
      b.text_frames = frames_push(b.text_frames, b.part_start, b.part_end);
      b.text_id = pool.append_word(b.text_id, b.word_part);
      b.word_part.clear();
    }
    b.last_id = -1;
    b.part_start = b.part_end = -1;
  }
  merge_beams(beams);
  apply_scores(beams, /*is_eos=*/true);
  prune(beams);

  std::vector<OutputBeam> out;
  out.reserve(beams.size());
  for (const Beam &b : beams) {
    const std::string &text = pool.get(b.text_id);
    OutputBeam ob;
    // _normalize_whitespace: words are joined with single spaces already, so
    // this only trims, but it is what the reference returns.
    std::vector<std::string> words = split_spaces(text);
    for (size_t i = 0; i < words.size(); i++) {
      if (i) ob.text += ' ';
      ob.text += words[i];
    }
    ob.frames = frames_to_vector(b.text_frames);
    ob.logit_score = b.score;
    ob.combined_score = b.lm_score;
    out.push_back(std::move(ob));
  }
  return out;
}

}  // namespace ctcbd
