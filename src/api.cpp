// The C entry points. Everything here is a thin translation between the C
// ABI and beam_search.h: no decoding logic, and no exception may escape.

#include "ctc_beam_decoder.h"

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "beam_search.h"
#include "language_model.h"

namespace {

void set_error(char *err, int32_t err_len, const char *msg) {
  if (err == nullptr || err_len <= 0) return;
  std::strncpy(err, msg, static_cast<size_t>(err_len) - 1);
  err[err_len - 1] = '\0';
}

std::vector<std::string> collect(const char *const *items, int32_t n) {
  std::vector<std::string> v;
  for (int32_t i = 0; i < n; i++) {
    if (items == nullptr || items[i] == nullptr) continue;
    v.emplace_back(items[i]);
  }
  return v;
}

}  // namespace

struct ctcbd_decoder {
  ctcbd::Decoder impl;
  explicit ctcbd_decoder(const std::vector<std::string> &labels) : impl(labels) {}
};

struct ctcbd_result {
  std::vector<ctcbd::OutputBeam> beams;
};

extern "C" {

int ctcbd_abi_version(void) { return CTCBD_ABI_VERSION; }

ctcbd_decoder *ctcbd_create(const char *const *labels, int32_t n_labels, char *err,
                            int32_t err_len) {
  try {
    if (labels == nullptr || n_labels <= 0) {
      set_error(err, err_len, "no labels given");
      return nullptr;
    }
    std::vector<std::string> v;
    v.reserve(static_cast<size_t>(n_labels));
    for (int32_t i = 0; i < n_labels; i++) {
      if (labels[i] == nullptr) {
        set_error(err, err_len, "null label");
        return nullptr;
      }
      v.emplace_back(labels[i]);
    }
    return new ctcbd_decoder(v);
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    return nullptr;
  } catch (...) {
    set_error(err, err_len, "unknown error building decoder");
    return nullptr;
  }
}

int ctcbd_has_kenlm(void) { return ctcbd::LanguageModel::available() ? 1 : 0; }

ctcbd_decoder *ctcbd_create_with_lm(const char *const *labels, int32_t n_labels,
                                    const char *kenlm_path, const char *const *unigrams,
                                    int32_t n_unigrams, double alpha, double beta,
                                    double unk_score_offset, int32_t score_boundary, char *err,
                                    int32_t err_len) {
  ctcbd_decoder *dec = ctcbd_create(labels, n_labels, err, err_len);
  if (dec == nullptr) return nullptr;
  try {
    if (kenlm_path == nullptr || *kenlm_path == '\0') {
      set_error(err, err_len, "no language model path given");
      delete dec;
      return nullptr;
    }
    ctcbd::LanguageModelOptions opts;
    opts.alpha = alpha;
    opts.beta = beta;
    opts.unk_score_offset = unk_score_offset;
    opts.score_boundary = score_boundary != 0;
    // A NULL unigram list is not an empty one: it means no vocabulary is
    // known, which changes how word prefixes are scored.
    dec->impl.set_language_model(std::unique_ptr<ctcbd::LanguageModel>(new ctcbd::LanguageModel(
        kenlm_path, collect(unigrams, n_unigrams), unigrams != nullptr, opts)));
    return dec;
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    delete dec;
    return nullptr;
  } catch (...) {
    set_error(err, err_len, "unknown error loading language model");
    delete dec;
    return nullptr;
  }
}

void ctcbd_free(ctcbd_decoder *dec) { delete dec; }

int32_t ctcbd_n_columns(const ctcbd_decoder *dec) {
  return dec == nullptr ? 0 : dec->impl.n_columns();
}

int32_t ctcbd_blank_id(const ctcbd_decoder *dec) {
  return dec == nullptr ? -1 : dec->impl.blank_id();
}

void ctcbd_prepare(float *log_probs, int32_t n_frames, int32_t n_columns) {
  if (log_probs == nullptr || n_frames <= 0 || n_columns <= 0) return;
  ctcbd::prepare_log_probs(log_probs, n_frames, n_columns);
}

ctcbd_result *ctcbd_decode(ctcbd_decoder *dec, const float *log_probs, int32_t n_frames,
                           int32_t n_columns, int32_t beam_width, double beam_prune_logp,
                           double token_min_logp, int32_t prune_history,
                           const char *const *hotwords, int32_t n_hotwords, double hotword_weight,
                           char *err, int32_t err_len) {
  try {
    if (dec == nullptr || log_probs == nullptr) {
      set_error(err, err_len, "null decoder or logits");
      return nullptr;
    }
    if (n_frames <= 0) {
      set_error(err, err_len, "no frames to decode");
      return nullptr;
    }
    if (n_columns != dec->impl.n_columns()) {
      set_error(err, err_len,
                ("logits have " + std::to_string(n_columns) + " columns, vocabulary needs " +
                 std::to_string(dec->impl.n_columns()))
                    .c_str());
      return nullptr;
    }
    if (beam_width <= 0) {
      set_error(err, err_len, "beam_width must be positive");
      return nullptr;
    }
    const std::vector<std::string> hw = collect(hotwords, n_hotwords);
    ctcbd::DecodeOptions opts;
    opts.beam_width = beam_width;
    opts.beam_prune_logp = beam_prune_logp;
    opts.token_min_logp = token_min_logp;
    opts.prune_history = prune_history != 0;
    opts.hotword_weight = hotword_weight;

    // Owned until it is handed back, so a throw from decode() frees it.
    std::unique_ptr<ctcbd_result> res(new ctcbd_result());
    res->beams = dec->impl.decode(log_probs, n_frames, hw, opts);
    return res.release();
  } catch (const std::bad_alloc &) {
    set_error(err, err_len, "out of memory");
    return nullptr;
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    return nullptr;
  } catch (...) {
    set_error(err, err_len, "unknown error decoding");
    return nullptr;
  }
}

void ctcbd_sizes(const ctcbd_result *res, int32_t *n_beams, int32_t *total_words,
                 int32_t *text_bytes) {
  int32_t nb = 0, nw = 0, tb = 0;
  if (res != nullptr) {
    nb = static_cast<int32_t>(res->beams.size());
    for (const auto &b : res->beams) {
      nw += static_cast<int32_t>(b.frames.size());
      tb += static_cast<int32_t>(b.text.size()) + 1;
    }
  }
  if (n_beams) *n_beams = nb;
  if (total_words) *total_words = nw;
  if (text_bytes) *text_bytes = tb;
}

void ctcbd_pack(const ctcbd_result *res, char *text, int32_t *words_per_beam, int32_t *frames,
                double *logit_scores, double *combined_scores) {
  if (res == nullptr) return;
  size_t text_at = 0, frame_at = 0;
  for (size_t i = 0; i < res->beams.size(); i++) {
    const auto &b = res->beams[i];
    if (text != nullptr) {
      std::memcpy(text + text_at, b.text.data(), b.text.size());
      text[text_at + b.text.size()] = '\0';
    }
    text_at += b.text.size() + 1;
    if (words_per_beam != nullptr) words_per_beam[i] = static_cast<int32_t>(b.frames.size());
    for (const auto &f : b.frames) {
      if (frames != nullptr) {
        frames[frame_at] = f.first;
        frames[frame_at + 1] = f.second;
      }
      frame_at += 2;
    }
    if (logit_scores != nullptr) logit_scores[i] = b.logit_score;
    if (combined_scores != nullptr) combined_scores[i] = b.combined_score;
  }
}

void ctcbd_result_free(ctcbd_result *res) { delete res; }

}  // extern "C"
