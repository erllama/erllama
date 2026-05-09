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

#include <new>
#include <stdint.h>

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

void erllama_safe_sampler_free(struct llama_sampler *s) noexcept {
    if (!s) return;
    try {
        llama_sampler_free(s);
    } catch (...) {
        // best-effort; nothing to do
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

void erllama_safe_sampler_accept(struct llama_sampler *s,
                                 llama_token tok) noexcept {
    try {
        llama_sampler_accept(s, tok);
    } catch (...) {
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

}  // extern "C"
