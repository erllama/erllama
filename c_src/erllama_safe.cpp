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
#include <cstring>
#include <new>
#include <pthread.h>
#include <stdint.h>

#ifndef ERLLAMA_THREAD_LOCAL
#  if defined(__cplusplus) && __cplusplus >= 201103L
#    define ERLLAMA_THREAD_LOCAL thread_local
#  else
#    define ERLLAMA_THREAD_LOCAL __thread
#  endif
#endif

// Sentinel returned by erllama_safe_decode on a thrown exception.
// Mirrors the macro in erllama_nif.c; both sides must agree.
#define ERLLAMA_DECODE_EXC_SENTINEL INT_MIN

// Model-load classification. Mirrored in erllama_nif.c; both
// sides must agree on the integer values.
typedef enum {
    ERLLAMA_LOAD_OK        = 0,
    ERLLAMA_LOAD_FAILED    = 1,  // generic NULL return
    ERLLAMA_LOAD_MALFORMED = 2,  // captured ASSERT-style log line
    ERLLAMA_LOAD_EXCEPTION = 3,  // C++ exception caught
} erllama_load_status_t;

// Best-effort capture of the most recent llama log line on the
// calling thread. The _v2 model load wrapper clears the buffer
// immediately before calling llama_model_load_from_file and
// inspects it on a NULL return to classify malformed-GGUF cases.
//
// llama_log_set is process-global, but the capture buffer is
// thread-local: each loader thread reads only what was logged
// on that same thread, so concurrent loads cannot scramble each
// other's classification. A message emitted from a worker thread
// spawned inside llama.cpp will not be visible to the loader,
// in which case the load is reported as FAILED rather than
// MALFORMED -- informational degradation only.
static ERLLAMA_THREAD_LOCAL char t_last_log_line[512] = {0};

static void erllama_log_capture(enum ggml_log_level level,
                                const char *text, void *user_data) noexcept {
    (void) level;
    (void) user_data;
    if (!text) return;
    size_t n = strlen(text);
    if (n >= sizeof(t_last_log_line)) n = sizeof(t_last_log_line) - 1;
    memcpy(t_last_log_line, text, n);
    t_last_log_line[n] = '\0';
}

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

struct llama_sampler *erllama_safe_sampler_init_dist(uint32_t seed) noexcept {
    try {
        return llama_sampler_init_dist(seed);
    } catch (...) {
        return nullptr;
    }
}

struct llama_sampler *erllama_safe_sampler_init_top_k(int32_t k) noexcept {
    try {
        return llama_sampler_init_top_k(k);
    } catch (...) {
        return nullptr;
    }
}

struct llama_sampler *erllama_safe_sampler_init_top_p(float p,
                                                     size_t min_keep) noexcept {
    try {
        return llama_sampler_init_top_p(p, min_keep);
    } catch (...) {
        return nullptr;
    }
}

struct llama_sampler *erllama_safe_sampler_init_min_p(float p,
                                                     size_t min_keep) noexcept {
    try {
        return llama_sampler_init_min_p(p, min_keep);
    } catch (...) {
        return nullptr;
    }
}

struct llama_sampler *erllama_safe_sampler_init_temp(float t) noexcept {
    try {
        return llama_sampler_init_temp(t);
    } catch (...) {
        return nullptr;
    }
}

struct llama_sampler *
erllama_safe_sampler_init_penalties(int32_t last_n, float repeat, float freq,
                                    float present) noexcept {
    try {
        return llama_sampler_init_penalties(last_n, repeat, freq, present);
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

// Lazy-init wrapper: backend_init runs at most once across the
// process. The NIF load path no longer eagerly invokes it; instead,
// the first model load triggers this helper. This keeps cache-only
// workloads (and unit tests that never touch llama) free of the
// ggml_backend_load_all side effects, which on some platforms
// (notably FreeBSD when paired with another NIF that registers
// resources or signal handlers) can perturb process state in ways
// that break unrelated code.
int erllama_safe_backend_init_once(void) noexcept {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    static int rc = 0;
    pthread_once(&once, []() noexcept {
        try {
            llama_backend_init();
            // Best-effort log capture for malformed-GGUF
            // classification. See erllama_log_capture comment.
            llama_log_set(erllama_log_capture, nullptr);
            rc = 0;
        } catch (...) {
            rc = -1;
        }
    });
    return rc;
}

int erllama_safe_backend_free(void) noexcept {
    try {
        llama_backend_free();
        return 0;
    } catch (...) {
        return -1;
    }
}

// Clear the process-global log callback so a NIF unload cannot
// leave llama.cpp pointing at a function in a soon-to-be-unmapped
// shared object. Called from the NIF unload path.
void erllama_safe_log_unset(void) noexcept {
    try {
        llama_log_set(nullptr, nullptr);
    } catch (...) {
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

// _v2 returns the model and writes a status to *out_status. Status
// is the source of truth; the model pointer is NULL on every
// non-OK status. The captured log buffer is cleared before the
// call so a stale GGML_ASSERT line from a prior load cannot
// mis-classify an unrelated NULL return as MALFORMED.
struct llama_model *
erllama_safe_model_load_from_file_v2(const char *path,
                                     struct llama_model_params params,
                                     erllama_load_status_t *out_status) noexcept {
    if (out_status) *out_status = ERLLAMA_LOAD_FAILED;
    t_last_log_line[0] = '\0';

    struct llama_model *m = nullptr;
    try {
        m = llama_model_load_from_file(path, params);
    } catch (...) {
        if (out_status) *out_status = ERLLAMA_LOAD_EXCEPTION;
        return nullptr;
    }
    if (m) {
        if (out_status) *out_status = ERLLAMA_LOAD_OK;
        return m;
    }

    bool malformed = strstr(t_last_log_line, "GGML_ASSERT") != nullptr;
    if (out_status) {
        *out_status = malformed ? ERLLAMA_LOAD_MALFORMED : ERLLAMA_LOAD_FAILED;
    }
    return nullptr;
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

// Total byte size of the model on disk, used by list_models to
// derive vram_estimate_b. Returns 0 on exception (caller treats as
// "unknown").
uint64_t erllama_safe_model_size(const struct llama_model *m) noexcept {
    try {
        return llama_model_size(m);
    } catch (...) {
        return 0;
    }
}

// Total layer count, used to turn n_gpu_layers into a fraction
// for vram_estimate_b. Returns 0 on exception.
int32_t erllama_safe_model_n_layer(const struct llama_model *m) noexcept {
    try {
        return llama_model_n_layer(m);
    } catch (...) {
        return 0;
    }
}

uint32_t erllama_safe_n_ctx(const struct llama_context *c) noexcept {
    try {
        return llama_n_ctx(c);
    } catch (...) {
        return 0;
    }
}

uint32_t erllama_safe_n_batch(const struct llama_context *c) noexcept {
    try {
        return llama_n_batch(c);
    } catch (...) {
        return 0;
    }
}

// Backend device enumeration. Used by nif_vram_info to walk all
// loaded ggml backends and sum free/total memory across non-CPU
// devices. ggml_backend_dev_t is opaque, so we expose only an
// index-based interface across the C ABI rather than passing
// pointers through the NIF boundary.
size_t erllama_safe_backend_dev_count(void) noexcept {
    try {
        return ggml_backend_dev_count();
    } catch (...) {
        return 0;
    }
}

// Look up the device at `idx` and write its memory + type to the
// out-params. Returns 0 on success, -1 on exception or invalid
// index. On failure the out-params are left untouched.
int erllama_safe_backend_dev_info(size_t idx, size_t *free_b,
                                  size_t *total_b, int *dev_type) noexcept {
    try {
        ggml_backend_dev_t dev = ggml_backend_dev_get(idx);
        if (!dev) return -1;
        size_t free_v = 0, total_v = 0;
        ggml_backend_dev_memory(dev, &free_v, &total_v);
        if (free_b) *free_b = free_v;
        if (total_b) *total_b = total_v;
        if (dev_type) *dev_type = (int) ggml_backend_dev_type(dev);
        return 0;
    } catch (...) {
        return -1;
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

// llama_batch_{init,free,get_one} are C++ entry points exported as
// C ABI. Today the vendored implementation uses malloc rather than
// new, but the surface is C++ and may grow throwing call sites
// across vendor bumps. Wrap them so an exception cannot unwind
// into the C NIF frame. On thrown exception the init/get_one
// shims return a zero-initialised batch (.token == nullptr,
// .n_tokens == 0); callers already check .token / .pos / .n_seq_id
// for NULL and treat it as allocation failure.
struct llama_batch erllama_safe_batch_init(int32_t n_tokens, int32_t embd,
                                           int32_t n_seq_max) noexcept {
    try {
        return llama_batch_init(n_tokens, embd, n_seq_max);
    } catch (...) {
        struct llama_batch z = {0, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr};
        return z;
    }
}

struct llama_batch erllama_safe_batch_get_one(llama_token *tokens,
                                              int32_t n_tokens) noexcept {
    try {
        return llama_batch_get_one(tokens, n_tokens);
    } catch (...) {
        struct llama_batch z = {0, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr};
        return z;
    }
}

void erllama_safe_batch_free(struct llama_batch batch) noexcept {
    try {
        llama_batch_free(batch);
    } catch (...) {
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

// Largest position present in the seq_id's KV state, or -1 when
// the sequence is empty (matches llama's own empty-sequence
// sentinel). -2 on exception so callers can disambiguate.
long erllama_safe_memory_seq_pos_max(struct llama_context *c,
                                     int seq_id) noexcept {
    try {
        llama_memory_t mem = llama_get_memory(c);
        if (!mem) return -2;
        return (long) llama_memory_seq_pos_max(mem, (llama_seq_id) seq_id);
    } catch (...) {
        return -2;
    }
}

// Speculative-decoding helper. Build a llama_batch with logits[i]=1
// for every position, decode, fill out_argmax[i] with argmax over
// the model vocab at each position. Used by erllama:verify/4.
//
// Sampler state is intentionally untouched: this path goes
// straight through llama_get_logits_ith / argmax and bypasses any
// configured sampler chain, so the caller's c->smpl stays clean.
//
// Returns 0 ok; -1 on llama_decode failure; -2 on batch_init OOM
// or invalid n_tokens; -3 on C++ exception.
int erllama_safe_forward_with_argmax(struct llama_context *c,
                                     const llama_token *tokens,
                                     int32_t n_tokens,
                                     int32_t n_vocab,
                                     long start_pos,
                                     int32_t *out_argmax) noexcept {
    if (n_tokens <= 0 || n_vocab <= 0 || !tokens || !out_argmax) {
        return -2;
    }
    try {
        struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        if (!batch.token) {
            return -2;
        }
        for (int32_t i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = (llama_pos)(start_pos + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = (llama_seq_id) 0;
            batch.logits[i] = 1;
        }
        batch.n_tokens = n_tokens;
        int rc = llama_decode(c, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            return -1;
        }
        for (int32_t i = 0; i < n_tokens; i++) {
            float *logits = llama_get_logits_ith(c, i);
            int32_t best = 0;
            float best_v = logits[0];
            for (int32_t v = 1; v < n_vocab; v++) {
                if (logits[v] > best_v) {
                    best_v = logits[v];
                    best = v;
                }
            }
            out_argmax[i] = best;
        }
        llama_batch_free(batch);
        return 0;
    } catch (...) {
        return -3;
    }
}

// ---------------------------------------------------------------------------
// Chat templating (bucket C, C-NIF)
// ---------------------------------------------------------------------------

// Returns the GGUF-stored chat template, or nullptr if the model has
// none. Pass `name = nullptr` to get the default template; passing a
// name selects a named alternate (rare).
const char *erllama_safe_model_chat_template(const struct llama_model *m,
                                             const char *name) noexcept {
    try {
        return llama_model_chat_template(m, name);
    } catch (...) {
        return nullptr;
    }
}

// Renders messages through a chat template. Same return convention
// as llama_chat_apply_template: bytes written, or negative
// needed-size when the buffer is too small. INT32_MIN on a thrown
// exception.
int32_t erllama_safe_chat_apply_template(const char *tmpl,
                                         const struct llama_chat_message *msgs,
                                         size_t n_msgs,
                                         bool add_assistant,
                                         char *buf, int32_t buf_size) noexcept {
    try {
        return llama_chat_apply_template(tmpl, msgs, n_msgs, add_assistant,
                                         buf, buf_size);
    } catch (...) {
        return INT32_MIN;
    }
}

// ---------------------------------------------------------------------------
// Grammar sampler (bucket C, C-NIF)
// ---------------------------------------------------------------------------

// Returns a new grammar-aware sampler, or nullptr on a thrown
// exception or invalid grammar.
struct llama_sampler *
erllama_safe_sampler_init_grammar(const struct llama_vocab *vocab,
                                  const char *grammar_str,
                                  const char *grammar_root) noexcept {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// LoRA adapters
// ---------------------------------------------------------------------------

// Load an adapter from a GGUF file. Bound to the model: stays valid
// until the model is freed (or until adapter_lora_free is called).
struct llama_adapter_lora *
erllama_safe_adapter_lora_init(struct llama_model *model,
                               const char *path) noexcept {
    try {
        return llama_adapter_lora_init(model, path);
    } catch (...) {
        return nullptr;
    }
}

// Explicit free. Safe to call once at most per adapter; the model
// destructor frees any adapter that wasn't explicitly freed.
void erllama_safe_adapter_lora_free(struct llama_adapter_lora *a) noexcept {
    if (!a) return;
    try {
        llama_adapter_lora_free(a);
    } catch (...) {
        // swallow; double-free or destructor exception is unrecoverable
        // but must not unwind into BEAM.
    }
}

// Install a set of adapters with their scales on a context. Passing
// n_adapters = 0 detaches everything.
int erllama_safe_set_adapters_lora(struct llama_context *ctx,
                                   struct llama_adapter_lora **adapters,
                                   size_t n_adapters,
                                   float *scales) noexcept {
    try {
        return llama_set_adapters_lora(ctx, adapters, n_adapters, scales);
    } catch (...) {
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Embeddings (bucket C, C-NIF)
// ---------------------------------------------------------------------------

// Per-sequence pooled embedding vector. Returns nullptr if the
// context was not created with embeddings = true, the model has no
// pooling, or on a thrown exception.
float *erllama_safe_get_embeddings_seq(struct llama_context *c,
                                       int seq_id) noexcept {
    try {
        return llama_get_embeddings_seq(c, (llama_seq_id) seq_id);
    } catch (...) {
        return nullptr;
    }
}

// Last-token (non-pooled) embedding. Used as a fallback for models
// whose pooling_type is NONE.
float *erllama_safe_get_embeddings(struct llama_context *c) noexcept {
    try {
        return llama_get_embeddings(c);
    } catch (...) {
        return nullptr;
    }
}

// Returns the model's embedding dimension, or 0 on exception.
int32_t erllama_safe_n_embd(const struct llama_model *m) noexcept {
    try {
        return llama_model_n_embd(m);
    } catch (...) {
        return 0;
    }
}

// Sets the embeddings flag on a live context. Lets a single context
// be flipped to embeddings mode for a single call without
// reallocation. Returns 0 on success, -1 on exception.
int erllama_safe_set_embeddings(struct llama_context *c, bool value) noexcept {
    try {
        llama_set_embeddings(c, value);
        return 0;
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
