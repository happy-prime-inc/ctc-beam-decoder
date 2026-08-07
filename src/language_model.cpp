#include "language_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#ifdef CTCBD_WITH_KENLM
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/exception.hh"
#endif

#include "hotwords.h"  // utf8_length

namespace ctcbd {

namespace {

// KenLM reports base-10 log probabilities; everything else here is natural
// log, so scores are converted before being combined.
const double kLogBaseChangeFactor = 1.0 / std::log10(std::exp(1.0));

// Expected characters per word. A prefix longer than this that still begins no
// known word is charged proportionally more — the longer a dead end runs, the
// less likely it is to become a word.
constexpr double kAvgTokenLen = 6.0;

}  // namespace

#ifdef CTCBD_WITH_KENLM

struct LmState::Impl {
  lm::ngram::State state;
};

LmState::LmState() : data_(new Impl()) {}
LmState::LmState(const LmState &other) : data_(new Impl(*other.data_)) {}
LmState &LmState::operator=(const LmState &other) {
  if (this != &other) *data_ = *other.data_;
  return *this;
}
LmState::~LmState() = default;

struct LanguageModel::Impl {
  std::unique_ptr<lm::base::Model> model;
  LanguageModelOptions opts;
  std::unordered_set<std::string> unigram_set;
  // Sorted, for "does any known word start with this?" by binary search.
  // pyctcdecode walks a character trie for the same question.
  std::vector<std::string> sorted_unigrams;
  bool have_unigrams = false;

  bool knows_prefix(const std::string &prefix) const {
    auto it = std::lower_bound(sorted_unigrams.begin(), sorted_unigrams.end(), prefix);
    return it != sorted_unigrams.end() && it->compare(0, prefix.size(), prefix) == 0;
  }

  bool in_vocabulary(const std::string &word) const {
    // KenLM maps every unknown word to index 0.
    return model->BaseVocabulary().Index(word) != 0;
  }
};

bool LanguageModel::available() { return true; }

LanguageModel::LanguageModel(const std::string &path, const std::vector<std::string> &unigrams,
                             bool have_unigrams, const LanguageModelOptions &opts)
    : impl_(new Impl()) {
  impl_->opts = opts;
  try {
    // Default configuration, which is what the reference implementation's
    // KenLM bindings use.
    impl_->model.reset(lm::ngram::LoadVirtual(path.c_str()));
  } catch (const util::Exception &e) {
    throw std::runtime_error("could not load language model " + path + ": " + e.what());
  }
  impl_->have_unigrams = have_unigrams;
  if (have_unigrams) {
    // Only words the model actually contains: a vocabulary listing words the
    // model has never seen would claim coverage it does not have.
    for (const std::string &w : unigrams) {
      if (impl_->in_vocabulary(w)) impl_->unigram_set.insert(w);
    }
    impl_->sorted_unigrams.assign(impl_->unigram_set.begin(), impl_->unigram_set.end());
    std::sort(impl_->sorted_unigrams.begin(), impl_->sorted_unigrams.end());
  }
}

LanguageModel::~LanguageModel() = default;

int LanguageModel::order() const { return static_cast<int>(impl_->model->Order()); }

LmState LanguageModel::start_state() const {
  LmState s;
  auto *state = static_cast<lm::ngram::State *>(s.data());
  if (impl_->opts.score_boundary) {
    impl_->model->BeginSentenceWrite(state);
  } else {
    impl_->model->NullContextWrite(state);
  }
  return s;
}

double LanguageModel::score(const LmState &prev, const std::string &word, bool is_last_word,
                            LmState *next) const {
  auto *out = static_cast<lm::ngram::State *>(next->data());
  const auto *in = static_cast<const lm::ngram::State *>(prev.data());
  const lm::WordIndex index = impl_->model->BaseVocabulary().Index(word);
  double lm_score = impl_->model->BaseScore(in, index, out);

  // Charge for a word the model does not know. The unigram list is consulted
  // first only because it is the cheaper lookup.
  const bool unknown = (impl_->have_unigrams && !impl_->unigram_set.count(word)) || index == 0;
  if (unknown) lm_score += impl_->opts.unk_score_offset;

  if (is_last_word && impl_->opts.score_boundary) {
    // Cost of ending the sentence here. The state this produces is thrown
    // away: `next` stays extendable, so a beam that turns out not to be final
    // can carry on.
    lm::ngram::State end_state;
    lm_score += impl_->model->BaseScore(out, impl_->model->BaseVocabulary().EndSentence(),
                                        &end_state);
  }
  return impl_->opts.alpha * lm_score * kLogBaseChangeFactor + impl_->opts.beta;
}

double LanguageModel::score_partial_token(const std::string &partial) const {
  // With no vocabulary list there is nothing to check a prefix against, so
  // every prefix is treated as unknown.
  const double is_oov =
      (!impl_->have_unigrams || !impl_->knows_prefix(partial)) ? 1.0 : 0.0;
  double unk_score = impl_->opts.unk_score_offset * is_oov;
  const double length = static_cast<double>(utf8_length(partial));
  if (length > kAvgTokenLen) unk_score = unk_score * length / kAvgTokenLen;
  return unk_score;
}

#else  // built without KenLM

struct LmState::Impl {};
LmState::LmState() : data_(new Impl()) {}
LmState::LmState(const LmState &) : data_(new Impl()) {}
LmState &LmState::operator=(const LmState &) { return *this; }
LmState::~LmState() = default;

struct LanguageModel::Impl {};

bool LanguageModel::available() { return false; }

LanguageModel::LanguageModel(const std::string &, const std::vector<std::string> &, bool,
                             const LanguageModelOptions &) {
  throw std::runtime_error(
      "this build has no language model support; rebuild with -DCTCBD_WITH_KENLM=ON");
}

LanguageModel::~LanguageModel() = default;
int LanguageModel::order() const { return 1; }
LmState LanguageModel::start_state() const { return LmState(); }
double LanguageModel::score(const LmState &, const std::string &, bool, LmState *) const {
  return 0.0;
}
double LanguageModel::score_partial_token(const std::string &) const { return 0.0; }

#endif

}  // namespace ctcbd
