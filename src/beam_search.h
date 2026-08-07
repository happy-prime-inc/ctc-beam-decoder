#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hotwords.h"
#include "language_model.h"

namespace ctcbd {

struct OutputBeam {
  std::string text;
  std::vector<std::pair<int32_t, int32_t>> frames;  // one per word in `text`
  double logit_score = 0.0;
  double combined_score = 0.0;
};

struct DecodeOptions {
  int32_t beam_width = 100;
  double beam_prune_logp = -10.0;
  double token_min_logp = -5.0;
  bool prune_history = false;
  double hotword_weight = 10.0;
};

class Decoder {
 public:
  // Normalises the vocabulary the way pyctcdecode's Alphabet does; may append
  // a blank, so n_columns() can exceed labels.size().
  explicit Decoder(const std::vector<std::string> &labels);

  // Takes ownership. Passing one switches on shallow fusion; without it the
  // decoder scores on acoustics and hotwords alone.
  void set_language_model(std::unique_ptr<LanguageModel> lm) { lm_ = std::move(lm); }
  bool has_language_model() const { return lm_ != nullptr; }

  int32_t n_columns() const { return static_cast<int32_t>(labels_.size()); }
  int32_t blank_id() const { return blank_id_; }

  std::vector<OutputBeam> decode(const float *log_probs, int32_t n_frames,
                                 const std::vector<std::string> &hotwords,
                                 const DecodeOptions &opts) const;

 private:
  std::vector<std::string> labels_;
  int32_t blank_id_ = -1;
  bool is_bpe_ = false;
  std::unique_ptr<LanguageModel> lm_;
};

// In-place log-softmax per row, then clip to [log(1e-15), 0].
void prepare_log_probs(float *log_probs, int32_t n_frames, int32_t n_columns);

}  // namespace ctcbd
