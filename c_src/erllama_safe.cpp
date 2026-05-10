// Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
// See the LICENSE file at the project root.

// erllama_safe: C++-side shims around llama.cpp calls that can throw.
//
// llama.cpp exposes a C ABI but is implemented in C++. Some entry
// points (sampler chain init, token_to_piece, sampler_sample) can
// allocate via `new` or hit unhandled internal exceptions on bad
// inputs (e.g. out-of-vocab token IDs reaching `id_to_token.at(id)`).
// An uncaught exception unwinding through a C frame in the BEAM is
// undefined behaviour and typically aborts the VM.
//
// Each wrapper here is `extern "C"` and `noexcept` so it never
// propagates an exception across the C boundary. The main NIF C file
// calls these shims and treats their negative/null returns as errors.

#include "llama.h"

#include <climits>
#include <new>
#include <stdint.h>

// Sentinel returned by erllama_safe_decode on a thrown exception.
// Mirrors the macro in erllama_nif.c; both sides must agree.
#define ERLLAMA_DECODE_EXC_SENTINEL INT_MIN

extern "C" {

struct llama_sampler *
erllama_safe_sampler_chain_init(struct llama_sampler_chain_params p) noexcept {
    try {
        return llama_sampler_chain_init(p);
    } catch (...) {
        return nullptr;
    }
}

struct llama_sampler *erllama_safe_sampler_init_greedy(void) noexcept {
    try {
        return llama_sampler_init_greedy();
    } catch (...) {
        return nullptr;
    }
}

int erllama_safe_sampler_chain_add(struct llama_sampler *chain,
                                   struct llama_sampler *s) noexcept {
    try {
        llama_sampler_chain_add(chain, s);
        return 0;
    } catch (...) {
        return -1;
    }
}

// Returns 0 on success, -1 on a thrown exception. Even the cleanup
// path surfaces the failure so the C caller can map it to an Erlang
// error rather than swallowing it silently.
int erllama_safe_sampler_free(struct llama_sampler *s) noexcept {
    if (!s) return 0;
    try {
        llama_sampler_free(s);
        return 0;
    } catch (...) {
        return -1;
    }
}

// Returns the sampled token, or -1 on exception. Callers must verify
// against the vocabulary size since -1 is a meaningful signal here.
llama_token erllama_safe_sampler_sample(struct llama_sampler *s,
                                         struct llama_context *ctx,
                                         int32_t idx) noexcept {
    try {
        return llama_sampler_sample(s, ctx, idx);
    } catch (...) {
        return (llama_token) -1;
    }
}

int erllama_safe_sampler_accept(struct llama_sampler *s,
                                llama_token tok) noexcept {
    try {
        llama_sampler_accept(s, tok);
        return 0;
    } catch (...) {
        return -1;
    }
}

// Returns bytes written, negative needed-size (matching llama's API)
// when the buffer is too small, or INT32_MIN on a thrown exception
// (which the C caller treats as "invalid token" / hard failure).
int32_t erllama_safe_token_to_piece(const struct llama_vocab *vocab,
                                    llama_token tok, char *buf,
                                    int32_t buf_size,
                                    int32_t lstrip,
                                    bool special) noexcept {
    try {
        return llama_token_to_piece(vocab, tok, buf, buf_size, lstrip, special);
    } catch (...) {
        return INT32_MIN;
    }
}

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

int erllama_safe_backend_init(void) noexcept {
    try {
        llama_backend_init();
        return 0;
    } catch (...) {
        return -1;
    }
}

int erllama_safe_backend_free(void) noexcept {
    try {
        llama_backend_free();
        return 0;
    } catch (...) {
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Model / context lifecycle
// ---------------------------------------------------------------------------

struct llama_model *
erllama_safe_model_load_from_file(const char *path,
                                  struct llama_model_params params) noexcept {
    try {
        return llama_model_load_from_file(path, params);
    } catch (...) {
        return nullptr;
    }
}

int erllama_safe_model_free(struct llama_model *m) noexcept {
    if (!m) return 0;
    try {
        llama_model_free(m);
        return 0;
    } catch (...) {
        return -1;
    }
}

struct llama_context *
erllama_safe_init_from_model(struct llama_model *m,
                             struct llama_context_params params) noexcept {
    try {
        return llama_init_from_model(m, params);
    } catch (...) {
        return nullptr;
    }
}

int erllama_safe_free(struct llama_context *c) noexcept {
    if (!c) return 0;
    try {
        llama_free(c);
        return 0;
    } catch (...) {
        return -1;
    }
}

const struct llama_model *
erllama_safe_get_model(const struct llama_context *c) noexcept {
    try {
        return llama_get_model(c);
    } catch (...) {
        return nullptr;
    }
}

const struct llama_vocab *
erllama_safe_model_get_vocab(const struct llama_model *m) noexcept {
    try {
        return llama_model_get_vocab(m);
    } catch (...) {
        return nullptr;
    }
}

int32_t erllama_safe_vocab_n_tokens(const struct llama_vocab *v) noexcept {
    try {
        return llama_vocab_n_tokens(v);
    } catch (...) {
        return 0;
    }
}

int erllama_safe_vocab_is_eog(const struct llama_vocab *v,
                              llama_token tok) noexcept {
    try {
        return llama_vocab_is_eog(v, tok) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Tokenize / decode / state
// ---------------------------------------------------------------------------

// Returns same conventions as llama_tokenize; INT32_MIN on thrown
// exception so callers can disambiguate from "needed N more slots".
int32_t erllama_safe_tokenize(const struct llama_vocab *vocab,
                              const char *text, int32_t text_len,
                              llama_token *tokens, int32_t n_max,
                              bool add_special, bool parse_special) noexcept {
    try {
        return llama_tokenize(vocab, text, text_len, tokens, n_max,
                              add_special, parse_special);
    } catch (...) {
        return INT32_MIN;
    }
}

// Returns 0 ok, llama's negative error codes on decode failure, or
// ERLLAMA_DECODE_EXC_SENTINEL (INT_MIN) on a thrown exception.
int erllama_safe_decode(struct llama_context *c,
                        struct llama_batch batch) noexcept {
    try {
        return llama_decode(c, batch);
    } catch (...) {
        return ERLLAMA_DECODE_EXC_SENTINEL;
    }
}

// State seq APIs. Return SIZE_MAX on thrown exception (callers treat
// as failure).
size_t erllama_safe_state_seq_get_size(struct llama_context *c,
                                       int seq_id) noexcept {
    try {
        return llama_state_seq_get_size(c, (llama_seq_id) seq_id);
    } catch (...) {
        return SIZE_MAX;
    }
}

// Returns the number of bytes written, or SIZE_MAX on a thrown
// exception. Distinguishing SIZE_MAX from "wrote zero bytes" lets
// the caller surface a clean exception error.
size_t erllama_safe_state_seq_get_data(struct llama_context *c,
                                       uint8_t *dst, size_t size,
                                       int seq_id) noexcept {
    try {
        return llama_state_seq_get_data(c, dst, size, (llama_seq_id) seq_id);
    } catch (...) {
        return SIZE_MAX;
    }
}

size_t erllama_safe_state_seq_set_data(struct llama_context *c,
                                       const uint8_t *src, size_t size,
                                       int seq_id) noexcept {
    try {
        return llama_state_seq_set_data(c, src, size, (llama_seq_id) seq_id);
    } catch (...) {
        return 0;
    }
}

// Memory ops for KV cell removal.
int erllama_safe_memory_seq_rm(struct llama_context *c, int seq_id,
                               int p0, int p1) noexcept {
    try {
        llama_memory_t mem = llama_get_memory(c);
        if (!mem) return -1;
        return llama_memory_seq_rm(mem, (llama_seq_id) seq_id,
                                   (llama_pos) p0, (llama_pos) p1)
                   ? 0
                   : -1;
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
