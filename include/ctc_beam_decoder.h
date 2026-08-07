/* CTC prefix beam search with hotword biasing.  C API.
 *
 * Flat C entry points, an explicit ABI version, and output that the caller
 * copies out and then frees.  Deliberately the same shape as the parakeet.cpp
 * binding this sits next to, so one ctypes pattern covers both.
 *
 * Copyright 2026 Happy Prime Inc.  Apache-2.0.
 * Behaviour follows pyctcdecode (Copyright 2021-present Kensho Technologies,
 * LLC; Apache-2.0) — see NOTICE.
 */
#ifndef CTC_BEAM_DECODER_H
#define CTC_BEAM_DECODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define CTCBD_API __declspec(dllexport)
#else
#define CTCBD_API __attribute__((visibility("default")))
#endif

/* Bumped on any change to the signatures or semantics below.  The binding
 * checks it before doing anything else, so a mismatched library fails with a
 * clear message rather than a corrupt read. */
#define CTCBD_ABI_VERSION 2

CTCBD_API int ctcbd_abi_version(void);

typedef struct ctcbd_decoder ctcbd_decoder;
typedef struct ctcbd_result ctcbd_result;

/* Build a decoder for `n_labels` vocabulary entries, given as NUL-terminated
 * UTF-8.  Labels are normalised the way pyctcdecode's Alphabet normalises
 * them: `<unk>`-style entries become the BPE unknown token, and a blank is
 * appended if the vocabulary does not already contain the empty string — so
 * the accepted number of logit columns may be `n_labels + 1`.  Call
 * ctcbd_n_columns() to find out which.
 *
 * Returns NULL on failure, writing a message into `err`. */
CTCBD_API ctcbd_decoder *ctcbd_create(const char *const *labels, int32_t n_labels,
                                      char *err, int32_t err_len);

/* As ctcbd_create, plus shallow fusion with an n-gram language model.
 *
 * `kenlm_path` is an ARPA file (optionally gzipped) or a KenLM binary.
 * `unigrams` is the known vocabulary and may be NULL, which is not the same
 * as empty: with no vocabulary every word prefix counts as unknown and is
 * penalised, which is what the reference implementation does.
 *
 * Returns NULL on failure — including when the library was built without
 * KenLM, which ctcbd_has_kenlm() reports up front. */
CTCBD_API ctcbd_decoder *ctcbd_create_with_lm(const char *const *labels, int32_t n_labels,
                                              const char *kenlm_path,
                                              const char *const *unigrams, int32_t n_unigrams,
                                              double alpha, double beta,
                                              double unk_score_offset, int32_t score_boundary,
                                              char *err, int32_t err_len);

/* Whether this build can load a language model at all. */
CTCBD_API int ctcbd_has_kenlm(void);

CTCBD_API void ctcbd_free(ctcbd_decoder *dec);

/* Number of logit columns this decoder expects per frame. */
CTCBD_API int32_t ctcbd_n_columns(const ctcbd_decoder *dec);

/* Index of the blank label. */
CTCBD_API int32_t ctcbd_blank_id(const ctcbd_decoder *dec);

/* Decode `n_frames` x `n_columns` log-probabilities, row-major, float32.
 *
 * `log_probs` must already be log-softmaxed and clipped to
 * [log(1e-15), 0] — see ctcbd_prepare(), which does exactly that.
 *
 * `hotwords` may be NULL.  Passing `prune_history` non-zero trades n-best
 * diversity for speed and should stay off wherever the n-best list is used.
 *
 * Returns NULL on failure, writing a message into `err`. */
CTCBD_API ctcbd_result *ctcbd_decode(ctcbd_decoder *dec, const float *log_probs,
                                     int32_t n_frames, int32_t n_columns, int32_t beam_width,
                                     double beam_prune_logp, double token_min_logp,
                                     int32_t prune_history, const char *const *hotwords,
                                     int32_t n_hotwords, double hotword_weight, char *err,
                                     int32_t err_len);

/* In-place log-softmax over each row followed by a clip to [log(1e-15), 0].
 * Matches what pyctcdecode does to its input before decoding. */
CTCBD_API void ctcbd_prepare(float *log_probs, int32_t n_frames, int32_t n_columns);

/* Sizes needed to allocate for ctcbd_pack().  `text_bytes` counts the
 * terminating NUL of every beam. */
CTCBD_API void ctcbd_sizes(const ctcbd_result *res, int32_t *n_beams, int32_t *total_words,
                           int32_t *text_bytes);

/* Copy the result into caller-owned buffers, sized by ctcbd_sizes():
 *
 *   text            all beam texts, each NUL-terminated, in rank order
 *   words_per_beam  [n_beams]
 *   frames          [total_words * 2], start/end pairs, beams concatenated
 *   logit_scores    [n_beams]
 *   combined_scores [n_beams]
 *
 * Words are not emitted separately: a beam's words are its text split on
 * single spaces, which is how the reference implementation pairs them with
 * frames. */
CTCBD_API void ctcbd_pack(const ctcbd_result *res, char *text, int32_t *words_per_beam,
                          int32_t *frames, double *logit_scores, double *combined_scores);

CTCBD_API void ctcbd_result_free(ctcbd_result *res);

#ifdef __cplusplus
}
#endif

#endif /* CTC_BEAM_DECODER_H */
