// N-gram language model scoring, following pyctcdecode's LanguageModel.
//
// Shallow fusion: the acoustic score of a beam gets the language model's
// opinion of the words added to it. What the model contributes is
//
//     alpha * (log-probability, converted from base 10) + beta
//
// per completed word, plus a penalty for words it has never seen — and, while
// a word is still being spelled out, a penalty for prefixes that begin no
// known word at all. That last part is what keeps beam search from wandering
// into letter sequences no word starts with.
//
// Compiled out entirely when built without KenLM; `available()` says which.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ctcbd {

// Opaque KenLM context. Sized and copied by the implementation, so nothing
// outside language_model.cpp needs KenLM's headers.
class LmState {
 public:
  LmState();
  LmState(const LmState &other);
  LmState &operator=(const LmState &other);
  ~LmState();

  void *data() { return data_.get(); }
  const void *data() const { return data_.get(); }

 private:
  struct Impl;
  std::unique_ptr<Impl> data_;
};

struct LanguageModelOptions {
  double alpha = 0.5;
  double beta = 1.5;
  // Charged against words the model has not seen. pyctcdecode's default of
  // -10.0 is deliberate and measured: sweeping it -10/-5/-2/0 on a live
  // recording moved word error rate 0.240/0.234/0.251/0.251 and left name
  // similarity flat, because most name failures are not close calls.
  double unk_score_offset = -10.0;
  // Whether the model sees sentence start and end.
  bool score_boundary = true;
};

class LanguageModel {
 public:
  // Loads an ARPA (optionally gzipped) or binary KenLM model. `unigrams` is
  // the known vocabulary; pass empty with have_unigrams=false to say there is
  // none, which makes every word prefix count as unknown.
  LanguageModel(const std::string &path, const std::vector<std::string> &unigrams,
                bool have_unigrams, const LanguageModelOptions &opts);
  ~LanguageModel();

  static bool available();  // false if built without KenLM

  int order() const;
  LmState start_state() const;

  // Score `word` following `prev`, writing the resulting context to `next`.
  // `is_last_word` additionally charges for ending the sentence here.
  double score(const LmState &prev, const std::string &word, bool is_last_word,
               LmState *next) const;

  // Penalty for a word prefix that starts no word the model knows.
  double score_partial_token(const std::string &partial) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ctcbd
