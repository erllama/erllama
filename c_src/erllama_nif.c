// Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
// See the LICENSE file at the project root.

/*
 * erllama_nif: single NIF for erllama (cache + llama.cpp surface).
 *
 * v0.2 surface:
 *   crc32c(IoData) -> non_neg_integer()              [dirty CPU]
 *   fsync_dir(Path) -> ok | {error, atom()}          [dirty IO]
 *   load_model(Path, Opts) -> {ok, ModelRes} | ...   [dirty IO]
 *   free_model(ModelRes) -> ok                       [regular]
 *   new_context(ModelRes, Opts) -> {ok, CtxRes} | .. [dirty CPU]
 *   free_context(CtxRes) -> ok                       [regular]
 *   tokenize(ModelRes, Text, Opts) -> [token_id()]   [dirty CPU]
 *   kv_pack(CtxRes, _Tokens, _NTokens) -> Binary     [dirty CPU]
 *   kv_unpack(CtxRes, Binary, SeqId) -> ok | err     [dirty CPU]
 *
 * Resource ownership: model and context resources hold pointers to
 * llama.cpp objects. Their destructors call llama_model_free /
 * llama_free. The context resource also holds a refcount on its
 * model resource via enif_keep_resource so the model survives as
 * long as any context derived from it does.
 */
#include <erl_nif.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Sentinel returned by erllama_safe_decode when llama_decode threw a
 * C++ exception. Distinct from any documented llama_decode return
 * (currently 0/1/-1/2). Defined here and in erllama_safe.cpp; both
 * sides must agree. */
#define ERLLAMA_DECODE_EXC_SENTINEL INT_MIN

/* Mirror of erllama_load_status_t in erllama_safe.cpp; both sides
 * must agree on the integer values. Used by the _v2 model load
 * wrapper to distinguish a generic NULL return from a captured
 * GGML_ASSERT-flavoured failure. */
typedef enum {
    ERLLAMA_LOAD_OK        = 0,
    ERLLAMA_LOAD_FAILED    = 1,
    ERLLAMA_LOAD_MALFORMED = 2,
    ERLLAMA_LOAD_EXCEPTION = 3,
} erllama_load_status_t;

#include "crc32c.h"
#include "llama.h"

/* Exception-safe wrappers for llama.cpp calls that can throw across
 * the C ABI. Implemented in c_src/erllama_safe.cpp. Each returns a
 * sentinel (NULL, 0, SIZE_MAX, INT32_MIN, etc.) on a thrown C++
 * exception so the C NIF can surface a clean {error, oom} or
 * {error, invalid_token} instead of letting an exception unwind into
 * a C frame. */
extern struct llama_sampler *erllama_safe_sampler_chain_init(
    struct llama_sampler_chain_params p);
extern struct llama_sampler *erllama_safe_sampler_init_greedy(void);
extern struct llama_sampler *erllama_safe_sampler_init_dist(uint32_t seed);
extern struct llama_sampler *erllama_safe_sampler_init_top_k(int32_t k);
extern struct llama_sampler *erllama_safe_sampler_init_top_p(float p,
                                                             size_t min_keep);
extern struct llama_sampler *erllama_safe_sampler_init_min_p(float p,
                                                             size_t min_keep);
extern struct llama_sampler *erllama_safe_sampler_init_temp(float t);
extern struct llama_sampler *erllama_safe_sampler_init_penalties(
    int32_t last_n, float repeat, float freq, float present);
extern int erllama_safe_sampler_chain_add(struct llama_sampler *chain,
                                          struct llama_sampler *s);
extern int erllama_safe_sampler_free(struct llama_sampler *s);
extern llama_token erllama_safe_sampler_sample(struct llama_sampler *s,
                                                struct llama_context *ctx,
                                                int32_t idx);
extern int erllama_safe_sampler_accept(struct llama_sampler *s,
                                       llama_token tok);
extern int32_t erllama_safe_token_to_piece(const struct llama_vocab *vocab,
                                           llama_token tok, char *buf,
                                           int32_t buf_size,
                                           int32_t lstrip,
                                           bool special);
extern int erllama_safe_backend_init(void);
extern int erllama_safe_backend_init_once(void);
extern int erllama_safe_backend_free(void);
extern struct llama_model *erllama_safe_model_load_from_file(
    const char *path, struct llama_model_params params);
extern struct llama_model *erllama_safe_model_load_from_file_v2(
    const char *path, struct llama_model_params params,
    erllama_load_status_t *out_status);
extern int erllama_safe_model_free(struct llama_model *m);
extern struct llama_context *erllama_safe_init_from_model(
    struct llama_model *m, struct llama_context_params params);
extern int erllama_safe_free(struct llama_context *c);
extern const struct llama_model *erllama_safe_get_model(
    const struct llama_context *c);
extern const struct llama_vocab *erllama_safe_model_get_vocab(
    const struct llama_model *m);
extern int32_t erllama_safe_vocab_n_tokens(const struct llama_vocab *v);
extern uint32_t erllama_safe_n_ctx(const struct llama_context *c);
extern uint32_t erllama_safe_n_batch(const struct llama_context *c);
extern int erllama_safe_vocab_is_eog(const struct llama_vocab *v,
                                     llama_token tok);
extern int32_t erllama_safe_tokenize(const struct llama_vocab *vocab,
                                     const char *text, int32_t text_len,
                                     llama_token *tokens, int32_t n_max,
                                     bool add_special, bool parse_special);
extern int erllama_safe_decode(struct llama_context *c,
                               struct llama_batch batch);
extern size_t erllama_safe_state_seq_get_size(struct llama_context *c,
                                              int seq_id);
extern size_t erllama_safe_state_seq_get_data(struct llama_context *c,
                                              uint8_t *dst, size_t size,
                                              int seq_id);
extern size_t erllama_safe_state_seq_set_data(struct llama_context *c,
                                              const uint8_t *src,
                                              size_t size, int seq_id);
extern int erllama_safe_memory_seq_rm(struct llama_context *c, int seq_id,
                                      int p0, int p1);
extern const char *erllama_safe_model_chat_template(const struct llama_model *m,
                                                    const char *name);
extern int32_t erllama_safe_chat_apply_template(
    const char *tmpl, const struct llama_chat_message *msgs, size_t n_msgs,
    bool add_assistant, char *buf, int32_t buf_size);
extern struct llama_sampler *erllama_safe_sampler_init_grammar(
    const struct llama_vocab *vocab, const char *grammar_str,
    const char *grammar_root);
extern struct llama_adapter_lora *erllama_safe_adapter_lora_init(
    struct llama_model *model, const char *path);
extern void erllama_safe_adapter_lora_free(struct llama_adapter_lora *a);
extern int erllama_safe_set_adapters_lora(struct llama_context *ctx,
                                          struct llama_adapter_lora **adapters,
                                          size_t n_adapters, float *scales);
extern float *erllama_safe_get_embeddings_seq(struct llama_context *c,
                                              int seq_id);
extern float *erllama_safe_get_embeddings(struct llama_context *c);
extern int32_t erllama_safe_n_embd(const struct llama_model *m);
extern int erllama_safe_set_embeddings(struct llama_context *c, bool value);

#ifndef ERLLAMA_MAX_TOKENS
/* Cap on accepted token-list inputs and tokenize output. The largest
 * practical context window today is ~10M; 1M tokens leaves plenty of
 * headroom while bounding worst-case allocations to ~4 MB and keeping
 * one bad request from tying up dirty schedulers indefinitely. */
#define ERLLAMA_MAX_TOKENS (1024 * 1024)
#endif

#ifndef ERLLAMA_MAX_TOKEN_TEXT
/* Largest text accepted by tokenize/3 (bytes). 4 MiB covers ~1 M
 * tokens at ~4 bytes each, well above any realistic chat prompt
 * while keeping a single bad request from chewing dirty-scheduler
 * time. Override at build time via -DERLLAMA_MAX_TOKEN_TEXT=N for
 * batch-tokenization workflows. */
#define ERLLAMA_MAX_TOKEN_TEXT (4 * 1024 * 1024)
#endif

/* =========================================================================
 * Atoms
 * ========================================================================= */

static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_error;
static ERL_NIF_TERM atom_load_failed;
static ERL_NIF_TERM atom_malformed_gguf;
static ERL_NIF_TERM atom_context_failed;
static ERL_NIF_TERM atom_tokenize_failed;
static ERL_NIF_TERM atom_pack_failed;
static ERL_NIF_TERM atom_unpack_failed;
static ERL_NIF_TERM atom_true;
static ERL_NIF_TERM atom_false;
static ERL_NIF_TERM atom_released;
static ERL_NIF_TERM atom_too_large;
static ERL_NIF_TERM atom_invalid_token;
static ERL_NIF_TERM atom_context_overflow;
static ERL_NIF_TERM atom_batch_overflow;
static ERL_NIF_TERM atom_oom;
static ERL_NIF_TERM atom_deferred;
static ERL_NIF_TERM atom_exception;
static ERL_NIF_TERM atom_no_logits;
static ERL_NIF_TERM atom_no_template;
static ERL_NIF_TERM atom_template_failed;
static ERL_NIF_TERM atom_grammar_failed;
static ERL_NIF_TERM atom_embed_failed;
static ERL_NIF_TERM atom_not_supported;
static ERL_NIF_TERM atom_invalid_content;

/* Forward decl: build_default_greedy_chain is defined in the sampler
 * section but used as a lazy fallback in nif_decode_one. */
static struct llama_sampler *build_default_greedy_chain(void);

/* Forward decl: adapter_dtor is defined later but registered in the
 * load callback. */
static void adapter_dtor(ErlNifEnv *env, void *obj);

/* Forward decl: sampler_dtor + the build helper, defined in the
 * sampler section. */
static void sampler_dtor(ErlNifEnv *env, void *obj);
static struct llama_sampler *build_sampler_chain_from_map(
    ErlNifEnv *env, ERL_NIF_TERM cfg, struct llama_context *ctx,
    ERL_NIF_TERM *out_err_atom);

/* =========================================================================
 * Resource types
 * ========================================================================= */

/* Per-resource mutex makes use-after-free between concurrent dirty
 * NIFs and an explicit free call impossible: every NIF entry that
 * dereferences a resource locks it, observes the pointer, and runs
 * llama under that lock; explicit frees take the same lock, so they
 * cannot interleave with a live llama call. The lock is held for the
 * duration of a llama op, but ops on different resources stay
 * independent. */
typedef struct {
    pthread_mutex_t mu;
    int mu_inited;                 /* guard pthread_mutex_destroy on error path */
    struct llama_model *model;     /* NULL after successful release */
    int active_contexts;           /* nif_new_context bumps; ctx_dtor decrements */
    int active_adapters;           /* nif_adapter_load bumps; adapter_dtor decrements */
    int release_pending;           /* free_model defers while either counter > 0 */
} erllama_model_t;

typedef struct {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_context *ctx;     /* NULL after successful release */
    erllama_model_t *model_res;    /* keep_resource'd by new_context */
    int decode_ready;              /* set after llama_decode; cleared after kv ops */
    /* Sampler chain cached on the first nif_decode_one call. The
     * chain is greedy-only and lives for the resource's lifetime;
     * a future sampler-config NIF would free + rebuild this under
     * the resource lock. */
    struct llama_sampler *smpl;
} erllama_context_t;

/* LoRA adapter resource. The adapter is bound to a model and stays
 * valid until the model is freed or adapter_lora_free is called
 * explicitly. The wrapping resource holds a keep-reference on its
 * model_res so the underlying llama_model* outlives the adapter even
 * if the user free_model's it. */
typedef struct {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_adapter_lora *adapter; /* NULL after explicit free */
    erllama_model_t *model_res;         /* keep_resource'd at init */
} erllama_adapter_t;

/* Sampler chain resource. Owned independently from the context so
 * multi-seq batching (v0.2+) can hold one chain per in-flight
 * request without contending on the context's cached `c->smpl`.
 * The chain is built from the same config map configure_sampler/2
 * consumes; freed explicitly via sampler_free/1 or implicitly by
 * the dtor. */
typedef struct {
    pthread_mutex_t mu;
    int mu_inited;
    struct llama_sampler *chain;   /* NULL after explicit free */
    erllama_context_t *ctx_res;    /* keep_resource'd at init */
} erllama_sampler_t;

static ErlNifResourceType *MODEL_RT;
static ErlNifResourceType *CTX_RT;
static ErlNifResourceType *ADAPTER_RT;
static ErlNifResourceType *SAMPLER_RT;

/* Drop the context's (or adapter's) reference on its model; if a
 * previous free_model/1 returned {ok, deferred} and the model is now
 * unreferenced by both contexts and adapters, actually free the
 * underlying llama_model* here. The decision is made under the lock
 * so concurrent destructions can't double-free. The free itself
 * runs while the lock is still held to keep the pointer
 * non-observable mid-teardown.
 *
 * Adapters share this gating because llama_model_free implicitly
 * frees any adapter that wasn't explicitly freed; if we freed the
 * model with an adapter wrapper still holding a (now dangling)
 * llama_adapter_lora* the next adapter_dtor would crash. */
static void context_drops_model(erllama_model_t *m) {
    pthread_mutex_lock(&m->mu);
    if (m->active_contexts > 0) {
        m->active_contexts--;
    }
    if (m->release_pending && m->active_contexts == 0
        && m->active_adapters == 0 && m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
        m->release_pending = 0;
    }
    pthread_mutex_unlock(&m->mu);
}

static void model_drops_adapter(erllama_model_t *m) {
    pthread_mutex_lock(&m->mu);
    if (m->active_adapters > 0) {
        m->active_adapters--;
    }
    if (m->release_pending && m->active_contexts == 0
        && m->active_adapters == 0 && m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
        m->release_pending = 0;
    }
    pthread_mutex_unlock(&m->mu);
}

/* Resource destructors run when the BEAM has no remaining references.
 * They must tolerate partial init: if alloc succeeded but mutex_init
 * failed, the dtor sees mu_inited=0 and skips pthread_mutex_destroy.
 * Pointer fields are zero-init'd by the allocation path so freeing a
 * NULL is a no-op here.
 *
 * Two accepted tradeoffs callers should know about:
 *
 *  1. A throwing llama destructor leaks the native object. C++
 *     destructors are required to be `noexcept`; if one throws
 *     anyway, the safe wrapper catches the exception and returns
 *     -1 but we still NULL the pointer so the destructor cannot
 *     be called twice. The native model/context is leaked rather
 *     than risking UB. Fix lives upstream in llama.cpp.
 *
 *  2. GC-triggered dtors run on the scheduler thread that
 *     triggered GC, not on a dirty scheduler. For prompt cleanup
 *     of a multi-MB model, callers should prefer
 *     `erllama:unload/1` (which terminates the per-model
 *     gen_statem and goes through `nif_free_context` -- a dirty
 *     CPU NIF) over relying on Erlang GC to destruct the
 *     resource. */
static void model_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_model_t *m = (erllama_model_t *) obj;
    /* The pointer is NULL after any successful or failed explicit
     * release, so this single check covers both paths and avoids
     * double-calling the safe wrapper. */
    if (m->model) {
        (void) erllama_safe_model_free(m->model);
        m->model = NULL;
    }
    if (m->mu_inited) {
        pthread_mutex_destroy(&m->mu);
        m->mu_inited = 0;
    }
}

static void ctx_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_context_t *c = (erllama_context_t *) obj;
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
        c->smpl = NULL;
    }
    if (c->ctx) {
        (void) erllama_safe_free(c->ctx);
        c->ctx = NULL;
    }
    if (c->model_res) {
        context_drops_model(c->model_res);
        enif_release_resource(c->model_res);
        c->model_res = NULL;
    }
    if (c->mu_inited) {
        pthread_mutex_destroy(&c->mu);
        c->mu_inited = 0;
    }
}

/* Sampler chain destructor. Frees the chain (which may be NULL if
 * the user called sampler_free explicitly) and drops the
 * keep-reference on the owning context. */
static void sampler_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_sampler_t *s = (erllama_sampler_t *) obj;
    if (s->chain) {
        (void) erllama_safe_sampler_free(s->chain);
        s->chain = NULL;
    }
    if (s->ctx_res) {
        enif_release_resource(s->ctx_res);
        s->ctx_res = NULL;
    }
    if (s->mu_inited) {
        pthread_mutex_destroy(&s->mu);
        s->mu_inited = 0;
    }
}

/* Adapter destructor. Explicit nif_adapter_free zeroes
 * a->adapter under the lock, so this destructor is either a no-op
 * (already freed) or the implicit final cleanup. Either way it
 * decrements the model's adapter count via model_drops_adapter
 * (which may complete a deferred free_model/1) and releases the
 * keep-reference on the model. */
static void adapter_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_adapter_t *a = (erllama_adapter_t *) obj;
    if (a->adapter) {
        erllama_safe_adapter_lora_free(a->adapter);
        a->adapter = NULL;
    }
    if (a->model_res) {
        model_drops_adapter(a->model_res);
        enif_release_resource(a->model_res);
        a->model_res = NULL;
    }
    if (a->mu_inited) {
        pthread_mutex_destroy(&a->mu);
        a->mu_inited = 0;
    }
}

/* =========================================================================
 * Load callback
 * ========================================================================= */

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void) priv_data;
    (void) load_info;

    if (erllama_crc32c_init() != 0) {
        return -1;
    }
    /* llama_backend_init() is deferred to first model load via
     * erllama_safe_backend_init_once(). NIF load only sets up
     * resources and atoms. Cache-only and cache-test workloads
     * never invoke ggml_backend_load_all, which on some platforms
     * (notably FreeBSD when paired with another NIF that uses
     * mmap and signal handlers) perturbs process state in ways
     * that break unrelated code paths. */

    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_load_failed = enif_make_atom(env, "load_failed");
    atom_malformed_gguf = enif_make_atom(env, "malformed_gguf");
    atom_context_failed = enif_make_atom(env, "context_failed");
    atom_tokenize_failed = enif_make_atom(env, "tokenize_failed");
    atom_pack_failed = enif_make_atom(env, "pack_failed");
    atom_unpack_failed = enif_make_atom(env, "unpack_failed");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
    atom_released = enif_make_atom(env, "released");
    atom_too_large = enif_make_atom(env, "too_large");
    atom_invalid_token = enif_make_atom(env, "invalid_token");
    atom_context_overflow = enif_make_atom(env, "context_overflow");
    atom_batch_overflow = enif_make_atom(env, "batch_overflow");
    atom_oom = enif_make_atom(env, "oom");
    atom_deferred = enif_make_atom(env, "deferred");
    atom_exception = enif_make_atom(env, "exception");
    atom_no_logits = enif_make_atom(env, "no_logits");
    atom_no_template = enif_make_atom(env, "no_template");
    atom_template_failed = enif_make_atom(env, "template_failed");
    atom_grammar_failed = enif_make_atom(env, "grammar_failed");
    atom_embed_failed = enif_make_atom(env, "embed_failed");
    atom_not_supported = enif_make_atom(env, "not_supported");
    atom_invalid_content = enif_make_atom(env, "invalid_content");

    MODEL_RT = enif_open_resource_type(
        env, NULL, "erllama_model", model_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!MODEL_RT) {
        return -1;
    }

    CTX_RT = enif_open_resource_type(
        env, NULL, "erllama_context", ctx_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!CTX_RT) {
        return -1;
    }

    ADAPTER_RT = enif_open_resource_type(
        env, NULL, "erllama_adapter", adapter_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!ADAPTER_RT) {
        return -1;
    }

    SAMPLER_RT = enif_open_resource_type(
        env, NULL, "erllama_sampler", sampler_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!SAMPLER_RT) {
        return -1;
    }

    return 0;
}

static void unload(ErlNifEnv *env, void *priv_data) {
    (void) env;
    (void) priv_data;
    /* If backend_init_once ran, free the global llama state so a
     * NIF reload (hot upgrade, test runner) doesn't leak. If it
     * never ran, llama_backend_free is a no-op. */
    (void) erllama_safe_backend_free();
}

/* =========================================================================
 * Helpers
 * ========================================================================= */

static int copy_path(ErlNifEnv *env, ERL_NIF_TERM term, char *out, size_t cap) {
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, term, &bin)) return 0;
    if (bin.size == 0 || bin.size >= cap) return 0;
    /* Reject embedded NUL: a Erlang binary like <<"real\0ignored">>
     * would be silently truncated by C string APIs to "real". */
    if (memchr(bin.data, '\0', bin.size) != NULL) return 0;
    memcpy(out, bin.data, bin.size);
    out[bin.size] = '\0';
    return 1;
}

/* Read an unsigned int but reject values that would wrap when cast
 * to int32_t. Used for llama options (n_gpu_layers, n_threads, etc.)
 * which are signed int32 fields in llama.cpp. */
static int get_map_int31(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, int32_t *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    unsigned int u;
    if (!enif_get_uint(env, v, &u)) return 0;
    if (u > (unsigned int) INT32_MAX) return 0;
    *out = (int32_t) u;
    return 1;
}

static int get_map_uint(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, unsigned int *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    return enif_get_uint(env, v, out);
}

/* Read a number from a map either as a float (`enif_get_double`) or as
 * an integer that gets promoted to double. Lets callers write
 * `temperature => 0` and `temperature => 0.7` interchangeably. */
static int get_map_double(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, double *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (enif_get_double(env, v, out)) return 1;
    long ll;
    if (enif_get_long(env, v, &ll)) {
        *out = (double) ll;
        return 1;
    }
    return 0;
}

static int get_map_bool(
    ErlNifEnv *env, ERL_NIF_TERM map, const char *key, int *out
) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (enif_compare(v, atom_true) == 0) {
        *out = 1;
        return 1;
    }
    if (enif_compare(v, atom_false) == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

/* =========================================================================
 * crc32c
 * ========================================================================= */

static ERL_NIF_TERM nif_crc32c(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, argv[0], &bin)) {
        return enif_make_badarg(env);
    }
    uint32_t crc = erllama_crc32c_update(0, bin.data, bin.size);
    return enif_make_uint(env, crc);
}

/* =========================================================================
 * Model
 * ========================================================================= */

static ERL_NIF_TERM nif_load_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    char path[4097];
    if (!copy_path(env, argv[0], path, sizeof(path))) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    if (erllama_safe_backend_init_once() != 0) {
        return enif_make_tuple2(env, atom_error, atom_load_failed);
    }

    struct llama_model_params params = llama_model_default_params();

    int32_t i32;
    if (get_map_int31(env, argv[1], "n_gpu_layers", &i32)) {
        params.n_gpu_layers = i32;
    }
    int b;
    if (get_map_bool(env, argv[1], "use_mmap", &b)) params.use_mmap = b ? true : false;
    if (get_map_bool(env, argv[1], "use_mlock", &b)) params.use_mlock = b ? true : false;
    if (get_map_bool(env, argv[1], "vocab_only", &b)) params.vocab_only = b ? true : false;

    erllama_load_status_t status = ERLLAMA_LOAD_FAILED;
    struct llama_model *model =
        erllama_safe_model_load_from_file_v2(path, params, &status);
    if (!model) {
        ERL_NIF_TERM why = (status == ERLLAMA_LOAD_MALFORMED)
                               ? atom_malformed_gguf
                               : atom_load_failed;
        return enif_make_tuple2(env, atom_error, why);
    }

    erllama_model_t *res = enif_alloc_resource(MODEL_RT, sizeof(*res));
    if (!res) {
        (void) erllama_safe_model_free(model);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    /* Zero-init so the destructor on the alloc-but-not-fully-set-up
     * path sees model=NULL and mu_inited=0 and skips the dangerous
     * frees. */
    memset(res, 0, sizeof(*res));
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        (void) erllama_safe_model_free(model);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    res->mu_inited = 1;
    res->model = model;
    res->active_contexts = 0;

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

/* free_model/1 returns:
 *   ok                 -> released; subsequent ops on the term return error
 *   {ok, deferred}     -> contexts or adapters still hold this model;
 *                         release flagged. The last context or adapter
 *                         destruction performs the actual llama_model_free
 *                         under context_drops_model / model_drops_adapter.
 *   {error, released}  -> already released
 *
 * The lock blocks for the duration of any concurrent dirty NIF using
 * this resource, which is the point: free can never interleave with a
 * live llama_model_* call. */
static ERL_NIF_TERM nif_free_model(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&m->mu);
    if (!m->model) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    if (m->active_contexts > 0 || m->active_adapters > 0) {
        m->release_pending = 1;
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_ok, atom_deferred);
    }
    struct llama_model *to_free = m->model;
    /* Free under the lock so a concurrent state read can't observe a
     * mid-free m->model. The pointer is nulled regardless of the
     * wrapper's return: calling llama_model_free again on a freed
     * pointer is a double-free, and llama destructors are required
     * to be noexcept anyway -- if one throws we leak the native
     * object rather than risk UB. */
    int rc = erllama_safe_model_free(to_free);
    m->model = NULL;
    m->release_pending = 0;
    pthread_mutex_unlock(&m->mu);
    if (rc != 0) {
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    return atom_ok;
}

/* =========================================================================
 * Context
 * ========================================================================= */

static ERL_NIF_TERM nif_new_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    struct llama_context_params params = llama_context_default_params();

    unsigned int u;
    if (get_map_uint(env, argv[1], "n_ctx", &u)) params.n_ctx = (uint32_t) u;
    if (get_map_uint(env, argv[1], "n_batch", &u)) params.n_batch = (uint32_t) u;
    if (get_map_uint(env, argv[1], "n_ubatch", &u)) params.n_ubatch = (uint32_t) u;
    if (get_map_uint(env, argv[1], "n_seq_max", &u)) params.n_seq_max = (uint32_t) u;
    int32_t i32;
    if (get_map_int31(env, argv[1], "n_threads", &i32)) params.n_threads = i32;
    if (get_map_int31(env, argv[1], "n_threads_batch", &i32)) {
        params.n_threads_batch = i32;
    }
    int b;
    if (get_map_bool(env, argv[1], "embeddings", &b)) params.embeddings = b ? true : false;
    if (get_map_bool(env, argv[1], "offload_kqv", &b)) params.offload_kqv = b ? true : false;

    pthread_mutex_lock(&m->mu);
    if (!m->model) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* If free_model/1 has been called and is waiting for the last
     * context to drop, do not let a new caller resurrect the model
     * by attaching another context. The {ok, deferred} return is
     * a release contract: no new contexts allowed past that point. */
    if (m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    struct llama_context *ctx = erllama_safe_init_from_model(m->model, params);
    if (!ctx) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_context_failed);
    }
    erllama_context_t *res = enif_alloc_resource(CTX_RT, sizeof(*res));
    if (!res) {
        (void) erllama_safe_free(ctx);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        (void) erllama_safe_free(ctx);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    res->mu_inited = 1;
    res->ctx = ctx;
    res->model_res = m;
    m->active_contexts++;
    enif_keep_resource(m);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

static ERL_NIF_TERM nif_free_context(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* Free under the lock so a concurrent reader cannot observe the
     * pointer mid-teardown. On exception we still NULL the pointer
     * to avoid a double-free path through the destructor; the native
     * object is leaked rather than risking UB. */
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
        c->smpl = NULL;
    }
    int free_rc = erllama_safe_free(c->ctx);
    c->ctx = NULL;
    c->decode_ready = 0;
    erllama_model_t *m = c->model_res;
    c->model_res = NULL;
    pthread_mutex_unlock(&c->mu);
    if (m) {
        context_drops_model(m);
        enif_release_resource(m);
    }
    if (free_rc != 0) {
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    return atom_ok;
}

/* =========================================================================
 * Tokenize
 * ========================================================================= */

static ERL_NIF_TERM nif_tokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    ErlNifBinary text;
    if (!enif_inspect_iolist_as_binary(env, argv[1], &text)) {
        return enif_make_badarg(env);
    }
    if (text.size > ERLLAMA_MAX_TOKEN_TEXT) {
        return enif_make_tuple2(env, atom_error, atom_too_large);
    }
    if (!enif_is_map(env, argv[2])) {
        return enif_make_badarg(env);
    }

    int add_special = 1;
    int parse_special = 0;
    int b;
    if (get_map_bool(env, argv[2], "add_special", &b)) add_special = b;
    if (get_map_bool(env, argv[2], "parse_special", &b)) parse_special = b;

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    int32_t text_len = (int32_t) text.size;
    int32_t n_max = text_len + 8;
    if (n_max < 16) n_max = 16;
    if (n_max > ERLLAMA_MAX_TOKENS) n_max = ERLLAMA_MAX_TOKENS;

    llama_token *tokens = (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) n_max);
    if (!tokens) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t n = erllama_safe_tokenize(
        vocab, (const char *) text.data, text_len, tokens,
        n_max, add_special ? true : false, parse_special ? true : false);
    if (n == INT32_MIN) {
        enif_free(tokens);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (n < 0) {
        int32_t needed = -n;
        if (needed > ERLLAMA_MAX_TOKENS) {
            enif_free(tokens);
            pthread_mutex_unlock(&m->mu);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
        enif_free(tokens);
        tokens = (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) needed);
        if (!tokens) {
            pthread_mutex_unlock(&m->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        n = erllama_safe_tokenize(
            vocab, (const char *) text.data, text_len, tokens,
            needed, add_special ? true : false, parse_special ? true : false);
    }
    pthread_mutex_unlock(&m->mu);
    if (n == INT32_MIN) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (n < 0) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_tokenize_failed);
    }

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, tokens[i]), list);
    }
    enif_free(tokens);
    return list;
}

/* =========================================================================
 * KV pack / unpack
 *
 * The 3-arg signatures preserve the v0.1 stub API. The Tokens and
 * NTokens / SeqId positional args are interpreted as documented in
 * include/llama.h: NTokens is unused (the in-memory API saves the
 * full state for the configured seq_id, defaulting to 0); SeqId is
 * the destination sequence id for unpack.
 * ========================================================================= */

static ERL_NIF_TERM nif_kv_pack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    /* Tokens (argv[1]) is informational; NTokens (argv[2]) ignored.
     * The model layer must have prefilled exactly the desired prefix
     * before calling kv_pack. argv[3], when present (arity 4),
     * specifies which sequence to extract from. Default 0 keeps
     * existing 3-arity callers working. */
    llama_seq_id seq_id = 0;
    if (argc == 4) {
        int sid;
        if (!enif_get_int(env, argv[3], &sid) || sid < 0) {
            return enif_make_badarg(env);
        }
        seq_id = (llama_seq_id) sid;
    }

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    size_t need = erllama_safe_state_seq_get_size(c->ctx, seq_id);
    if (need == SIZE_MAX) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (need == 0) {
        pthread_mutex_unlock(&c->mu);
        ErlNifBinary empty;
        if (!enif_alloc_binary(0, &empty)) {
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        return enif_make_binary(env, &empty);
    }
    ErlNifBinary out;
    if (!enif_alloc_binary(need, &out)) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    size_t written = erllama_safe_state_seq_get_data(
        c->ctx, out.data, out.size, seq_id);
    pthread_mutex_unlock(&c->mu);
    if (written == SIZE_MAX) {
        enif_release_binary(&out);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (written == 0 || written > need) {
        enif_release_binary(&out);
        return enif_make_tuple2(env, atom_error, atom_pack_failed);
    }
    if (written < need) {
        if (!enif_realloc_binary(&out, written)) {
            enif_release_binary(&out);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
    }
    return enif_make_binary(env, &out);
}

static ERL_NIF_TERM nif_kv_unpack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    ErlNifBinary in;
    if (!enif_inspect_binary(env, argv[1], &in)) {
        return enif_make_badarg(env);
    }
    int seq_id;
    if (!enif_get_int(env, argv[2], &seq_id) || seq_id < 0) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    size_t consumed = erllama_safe_state_seq_set_data(
        c->ctx, in.data, in.size, seq_id);
    /* kv_unpack only restores KV cells, not the per-context logits
     * buffer; the model layer must drop the last cell and re-prefill
     * it before the next sample. Mark the context as not ready until
     * that primer runs. */
    c->decode_ready = 0;
    pthread_mutex_unlock(&c->mu);
    if (consumed == 0 || consumed != in.size) {
        return enif_make_tuple2(env, atom_error, atom_unpack_failed);
    }
    return atom_ok;
}

/* Remove the cells in [p0, p1) from the given sequence. p0 < 0 means
 * 0; p1 < 0 means infinity. Returns ok or {error, partial}. The save
 * format only stores KV cells; the per-context logits buffer is not
 * restored. So after kv_unpack the model layer drops the last cell of
 * the saved sequence and re-prefills the corresponding token to
 * regenerate logits for the next sample. */
static ERL_NIF_TERM nif_kv_seq_rm(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    int seq_id, p0, p1;
    if (!enif_get_int(env, argv[1], &seq_id) || seq_id < 0 ||
        !enif_get_int(env, argv[2], &p0) ||
        !enif_get_int(env, argv[3], &p1)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    int rc = erllama_safe_memory_seq_rm(c->ctx, seq_id, p0, p1);
    /* Removing cells invalidates last-decode logits; force a fresh
     * prefill before the next sample. */
    c->decode_ready = 0;
    pthread_mutex_unlock(&c->mu);
    if (rc != 0) {
        return enif_make_tuple2(env, atom_error, atom_unpack_failed);
    }
    return atom_ok;
}

/* =========================================================================
 * Prefill / decode_one / detokenize
 * ========================================================================= */

/* read_token_list: returns 1 ok, 0 badarg, -1 oom, -2 too_large,
 * -3 invalid_token (out-of-range value). */
static int read_token_list(ErlNifEnv *env, ERL_NIF_TERM list,
                           llama_token **out, int32_t *out_len) {
    unsigned int n;
    if (!enif_get_list_length(env, list, &n)) return 0;
    if (n > (unsigned int) ERLLAMA_MAX_TOKENS) return -2;
    if (n == 0) {
        *out = NULL;
        *out_len = 0;
        return 1;
    }
    llama_token *toks = enif_alloc(sizeof(llama_token) * (size_t) n);
    if (!toks) return -1;
    ERL_NIF_TERM head, tail = list;
    unsigned int i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        int v;
        if (!enif_get_int(env, head, &v)) {
            enif_free(toks);
            return 0;
        }
        if (v < 0) {
            enif_free(toks);
            return -3;
        }
        toks[i++] = (llama_token) v;
    }
    *out = toks;
    *out_len = (int32_t) n;
    return 1;
}

static ERL_NIF_TERM token_list_error(ErlNifEnv *env, int rc) {
    switch (rc) {
        case -1: return enif_make_tuple2(env, atom_error, atom_oom);
        case -2: return enif_make_tuple2(env, atom_error, atom_too_large);
        case -3: return enif_make_tuple2(env, atom_error, atom_invalid_token);
        default: return enif_make_badarg(env);
    }
}

static ERL_NIF_TERM nif_prefill(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        return atom_ok;
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* Validate token IDs against the model vocab before handing them
     * to llama_decode. An out-of-range positive ID would otherwise
     * reach `id_to_token.at(id)` deep inside llama and throw a C++
     * exception across the C ABI. */
    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    /* Fail closed if the vocab lookup failed: without n_vocab we
     * cannot validate token IDs, and an out-of-range positive ID
     * would reach `id_to_token.at(id)` deep inside llama and throw
     * a C++ exception across the C ABI. */
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&c->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
    }
    /* Bounds-check against the live context. llama_decode dereferences
     * past the KV slab when n_tokens >= n_ctx, and is undefined when
     * n_tokens > n_batch -- both produce SIGSEGV under real load. */
    uint32_t n_ctx = erllama_safe_n_ctx(c->ctx);
    uint32_t n_batch = erllama_safe_n_batch(c->ctx);
    if (n_ctx == 0 || n_batch == 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if ((uint32_t) n >= n_ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_context_overflow);
    }
    if ((uint32_t) n > n_batch) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }
    struct llama_batch batch = llama_batch_get_one(tokens, n);
    int dr = erllama_safe_decode(c->ctx, batch);
    if (dr == 0) c->decode_ready = 1;
    pthread_mutex_unlock(&c->mu);
    enif_free(tokens);
    if (dr == ERLLAMA_DECODE_EXC_SENTINEL) {
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (dr != 0) {
        return enif_make_tuple2(env, atom_error, enif_make_int(env, dr));
    }
    return atom_ok;
}

static ERL_NIF_TERM nif_decode_one(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* `llama_sampler_sample` -> `llama_get_logits_ith` aborts via
     * GGML_ASSERT(logits != nullptr) when no decode has produced
     * sample-able logits yet. We can't catch that abort, so we refuse
     * to call sampler_sample unless the last successful op was a
     * decode (set by nif_prefill / by ourselves below). kv_unpack and
     * kv_seq_rm clear the flag; the model layer must re-prefill the
     * last token before sampling. */
    if (!c->decode_ready) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_no_logits);
    }

    /* Lazy-init the sampler chain on first use as a greedy fallback,
     * matching the behaviour callers got before configure_sampler/2
     * existed. The model layer should normally call configure_sampler
     * once per request before the first decode; this fallback keeps
     * the cache-only and stub-backed call sites working without
     * touching them. */
    if (!c->smpl) {
        struct llama_sampler *fallback = build_default_greedy_chain();
        if (!fallback) {
            pthread_mutex_unlock(&c->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        c->smpl = fallback;
    }

    /* llama_sampler_sample calls llama_sampler_accept on the chain
     * internally; the chain stays cached, so accept lands on the
     * cached object. */
    llama_token tok = erllama_safe_sampler_sample(c->smpl, c->ctx, -1);
    if (tok < 0) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int eog = vocab ? erllama_safe_vocab_is_eog(vocab, tok) : 0;

    llama_token tok_buf = tok;
    struct llama_batch batch = llama_batch_get_one(&tok_buf, 1);
    int rc = erllama_safe_decode(c->ctx, batch);
    if (rc == 0) c->decode_ready = 1;
    else c->decode_ready = 0;
    pthread_mutex_unlock(&c->mu);
    if (rc == ERLLAMA_DECODE_EXC_SENTINEL) {
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (rc != 0) {
        return enif_make_tuple2(env, atom_error, enif_make_int(env, rc));
    }
    ERL_NIF_TERM tag = eog ? enif_make_atom(env, "eog") : atom_ok;
    return enif_make_tuple2(env, tag, enif_make_int(env, tok));
}

static ERL_NIF_TERM nif_detokenize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        ErlNifBinary empty;
        if (!enif_alloc_binary(0, &empty)) {
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        return enif_make_binary(env, &empty);
    }

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    int32_t n_vocab = erllama_safe_vocab_n_tokens(vocab);
    /* Fail closed if the vocab lookup gave us no usable size: without
     * n_vocab we cannot validate token IDs, and an out-of-range
     * positive ID would reach `id_to_token.at(id)` deep inside llama
     * and throw across the C ABI. Mirrors the prefill path. */
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }
    /* Validate before any token_to_piece call so out-of-range IDs do
     * not reach `id_to_token.at(id)` and trigger an internal throw. */
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&m->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
    }

    /* Per-token piece, concatenated. Pieces are typically a handful
     * of bytes; we grow the buffer on demand and re-call
     * llama_token_to_piece with a sized buffer when 256 bytes isn't
     * enough (it returns the negative needed size). The safe wrapper
     * returns INT32_MIN on a thrown C++ exception. */
    char small_piece[256];
    /* Guard the size computation: clamp n to a sane upper bound so
     * gcc's range analysis can prove cap fits. 16M tokens is far
     * beyond any realistic prompt; reject earlier rather than
     * overflow. */
    if (n < 0 || n > (1 << 24)) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_too_large);
    }
    size_t cap = (size_t) n * 32u + 16u;
    char *out = enif_alloc(cap);
    if (!out) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    size_t used = 0;
    int err = 0;
    for (int32_t i = 0; i < n; i++) {
        char *piece_buf = small_piece;
        int32_t buf_size = (int32_t) sizeof(small_piece);
        char *grown = NULL;
        int32_t got = erllama_safe_token_to_piece(
            vocab, tokens[i], piece_buf, buf_size, 0, false);
        if (got == INT32_MIN) {
            err = 1;
            break;
        }
        if (got < 0) {
            int32_t need = -got;
            if (need <= 0 || need > (1 << 20)) {
                err = 1;
                break;
            }
            grown = enif_alloc((size_t) need);
            if (!grown) { err = 2; break; }
            piece_buf = grown;
            got = erllama_safe_token_to_piece(
                vocab, tokens[i], piece_buf, need, 0, false);
            if (got == INT32_MIN || got < 0) {
                enif_free(grown);
                err = 1;
                break;
            }
        }
        if (used + (size_t) got > cap) {
            size_t new_cap = (used + (size_t) got) * 2 + 16;
            char *new_out = enif_realloc(out, new_cap);
            if (!new_out) {
                if (grown) enif_free(grown);
                err = 2;
                break;
            }
            out = new_out;
            cap = new_cap;
        }
        memcpy(out + used, piece_buf, (size_t) got);
        used += (size_t) got;
        if (grown) enif_free(grown);
    }
    pthread_mutex_unlock(&m->mu);
    enif_free(tokens);
    if (err) {
        enif_free(out);
        if (err == 2) return enif_make_tuple2(env, atom_error, atom_oom);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }

    ErlNifBinary outbin;
    if (!enif_alloc_binary(used, &outbin)) {
        enif_free(out);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    memcpy(outbin.data, out, used);
    enif_free(out);
    return enif_make_binary(env, &outbin);
}

/* =========================================================================
 * fsync_dir (existing)
 * ========================================================================= */

static ERL_NIF_TERM make_errno_atom(ErlNifEnv *env, int e) {
    const char *name;
    switch (e) {
        case EACCES:    name = "eacces";    break;
        case EBUSY:     name = "ebusy";     break;
        case EEXIST:    name = "eexist";    break;
        case EINVAL:    name = "einval";    break;
        case EIO:       name = "eio";       break;
        case EISDIR:    name = "eisdir";    break;
        case ELOOP:     name = "eloop";     break;
        case EMFILE:    name = "emfile";    break;
        case ENAMETOOLONG: name = "enametoolong"; break;
        case ENFILE:    name = "enfile";    break;
        case ENOENT:    name = "enoent";    break;
        case ENOMEM:    name = "enomem";    break;
        case ENOSPC:    name = "enospc";    break;
        case ENOTDIR:   name = "enotdir";   break;
        case EPERM:     name = "eperm";     break;
        case EROFS:     name = "erofs";     break;
#ifdef EINTEGRITY
        /* FreeBSD fsync(2) returns EINTEGRITY on filesystem
         * integrity errors (ZFS checksum failure, ufs2 sb
         * mismatch). Surface it instead of mapping to "unknown". */
        case EINTEGRITY: name = "eintegrity"; break;
#endif
        default:        name = "unknown";   break;
    }
    return enif_make_atom(env, name);
}

static ERL_NIF_TERM nif_fsync_dir(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    char path[4097];
    /* copy_path rejects empty inputs, oversize inputs, and embedded
     * NUL bytes (which would otherwise let `<<"a\0b">>` be passed to
     * open() as just "a"). */
    if (!copy_path(env, argv[0], path, sizeof(path))) {
        return enif_make_badarg(env);
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return enif_make_tuple2(env, atom_error, make_errno_atom(env, errno));
    }
    int rc = fsync(fd);
    int saved = errno;
    close(fd);
    if (rc != 0) {
        return enif_make_tuple2(env, atom_error, make_errno_atom(env, saved));
    }
    return atom_ok;
}

/* =========================================================================
 * Chat templating
 * =========================================================================
 *
 * nif_apply_chat_template renders a normalised chat request through
 * the model's chat template (read from GGUF metadata) and tokenises
 * the result. The Request map carries:
 *
 *   #{ messages := [#{role := binary(), content := binary()}]
 *    , system   => binary() | undefined
 *    , tools    => [#{name := binary(), description => binary(),
 *                     schema => map()}] | undefined
 *    }
 *
 * `tools` are inlined as a synthetic system addendum because
 * llama_chat_apply_template does not take a tools field. Models that
 * embed tool definitions in their template (llama-3.1, hermes-2,
 * qwen2.5) read them from the system block.
 */

/* Pull a binary value out of `Map[Key]`. Returns 1 with `bin` filled
 * on success, 0 if the key is missing or not a binary. The returned
 * `bin` points into a process-owned region; copy before unlocking
 * any cross-call resource. */
static int get_map_bin(ErlNifEnv *env, ERL_NIF_TERM map, const char *key,
                      ErlNifBinary *bin) {
    ERL_NIF_TERM v;
    ERL_NIF_TERM k = enif_make_atom(env, key);
    if (!enif_get_map_value(env, map, k, &v)) return 0;
    if (!enif_inspect_iolist_as_binary(env, v, bin)) return 0;
    return 1;
}

static void free_chat_msgs(struct llama_chat_message *msgs, int n) {
    for (int i = 0; i < n; i++) {
        if (msgs[i].role) enif_free((char *) msgs[i].role);
        if (msgs[i].content) enif_free((char *) msgs[i].content);
    }
}

/* Iterate over a list of message maps and fill `out_msgs` with
 * llama_chat_message structs. Each message is `#{role := ..., content := ...}`.
 * The role and content strings are allocated with enif_alloc and the
 * caller must free them via free_chat_msgs.
 *
 * On error the helper frees the role+content allocations it placed
 * into out[idx0..idx-1] before returning. The caller's free_chat_msgs
 * call (over its pre-call n_msgs range) does not overlap that range,
 * so no double-free is reachable.
 *
 * Returns the number of messages on success, -1 on bad input
 * (missing role/content key, role not iolist-binary), -2 on OOM,
 * or -3 when `content` is present but not iolist-binary
 * (Anthropic-style content blocks). The caller distinguishes -3
 * to surface {error, invalid_content} rather than badarg.
 */
static int build_chat_msgs_from_list(
    ErlNifEnv *env, ERL_NIF_TERM list,
    struct llama_chat_message *out, int max_out, int idx0
) {
    int idx = idx0;
    int err = 0;
    ERL_NIF_TERM head, tail = list;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        if (idx >= max_out) { err = -1; goto cleanup; }
        if (!enif_is_map(env, head)) { err = -1; goto cleanup; }
        ErlNifBinary role_bin, content_bin;
        if (!get_map_bin(env, head, "role", &role_bin)) { err = -1; goto cleanup; }
        ERL_NIF_TERM content_term;
        if (!enif_get_map_value(env, head, enif_make_atom(env, "content"),
                                &content_term)) {
            err = -1;
            goto cleanup;
        }
        if (!enif_inspect_iolist_as_binary(env, content_term, &content_bin)) {
            err = -3;
            goto cleanup;
        }
        char *role = enif_alloc(role_bin.size + 1);
        if (!role) { err = -2; goto cleanup; }
        memcpy(role, role_bin.data, role_bin.size);
        role[role_bin.size] = '\0';
        char *content = enif_alloc(content_bin.size + 1);
        if (!content) {
            enif_free(role);
            err = -2;
            goto cleanup;
        }
        memcpy(content, content_bin.data, content_bin.size);
        content[content_bin.size] = '\0';
        out[idx].role = role;
        out[idx].content = content;
        idx++;
    }
    return idx;

cleanup:
    free_chat_msgs(out + idx0, idx - idx0);
    return err;
}

/* Build a synthetic system content string that prepends the user-
 * supplied system text and renders tools as a textual list, so models
 * whose chat templates honour tool definitions in the system block
 * (llama-3.1+, hermes-2-pro, qwen2.5) see them. Caller frees with
 * enif_free.
 *
 * Returns the malloced string or NULL on OOM. `*out_len` is set to
 * the strlen for convenience. */
static char *build_system_content(ErlNifEnv *env, ERL_NIF_TERM request_map,
                                  size_t *out_len) {
    ErlNifBinary system_bin = {0};
    int has_system = get_map_bin(env, request_map, "system", &system_bin);

    ERL_NIF_TERM tools_term;
    int has_tools =
        enif_get_map_value(env, request_map, enif_make_atom(env, "tools"),
                           &tools_term)
        && enif_is_list(env, tools_term);

    if (!has_system && !has_tools) {
        if (out_len) *out_len = 0;
        return NULL;
    }

    /* Render: `<system>\n\nAvailable tools:\n  - name: description\n...` */
    size_t cap = 256;
    if (has_system) cap += system_bin.size;
    char *buf = enif_alloc(cap);
    if (!buf) return NULL;
    size_t pos = 0;
    if (has_system) {
        memcpy(buf + pos, system_bin.data, system_bin.size);
        pos += system_bin.size;
    }
    if (has_tools) {
        const char *header = (has_system ? "\n\nAvailable tools:\n" :
                                            "Available tools:\n");
        size_t header_len = strlen(header);
        if (pos + header_len + 1 > cap) {
            cap = (pos + header_len + 1) * 2;
            char *nbuf = enif_realloc(buf, cap);
            if (!nbuf) { enif_free(buf); return NULL; }
            buf = nbuf;
        }
        memcpy(buf + pos, header, header_len);
        pos += header_len;
        ERL_NIF_TERM head, tail = tools_term;
        while (enif_get_list_cell(env, tail, &head, &tail)) {
            if (!enif_is_map(env, head)) continue;
            ErlNifBinary name_bin, desc_bin;
            if (!get_map_bin(env, head, "name", &name_bin)) continue;
            int has_desc = get_map_bin(env, head, "description", &desc_bin);
            size_t needed = 4 + name_bin.size + 2 +
                             (has_desc ? desc_bin.size : 0) + 1;
            if (pos + needed + 1 > cap) {
                cap = (pos + needed + 1) * 2;
                char *nbuf = enif_realloc(buf, cap);
                if (!nbuf) { enif_free(buf); return NULL; }
                buf = nbuf;
            }
            memcpy(buf + pos, "  - ", 4); pos += 4;
            memcpy(buf + pos, name_bin.data, name_bin.size); pos += name_bin.size;
            if (has_desc) {
                memcpy(buf + pos, ": ", 2); pos += 2;
                memcpy(buf + pos, desc_bin.data, desc_bin.size); pos += desc_bin.size;
            }
            buf[pos++] = '\n';
        }
    }
    buf[pos] = '\0';
    if (out_len) *out_len = pos;
    return buf;
}

static ERL_NIF_TERM nif_apply_chat_template(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    /* Read the messages list from the request. */
    ERL_NIF_TERM messages_term;
    if (!enif_get_map_value(env, argv[1],
                            enif_make_atom(env, "messages"), &messages_term)
        || !enif_is_list(env, messages_term)) {
        return enif_make_badarg(env);
    }
    unsigned msg_len;
    if (!enif_get_list_length(env, messages_term, &msg_len)) {
        return enif_make_badarg(env);
    }

    /* +1 for an optional synthetic system message at the front. */
    int max_msgs = (int) msg_len + 1;
    struct llama_chat_message *msgs =
        enif_alloc(sizeof(struct llama_chat_message) * (size_t) max_msgs);
    if (!msgs) return enif_make_tuple2(env, atom_error, atom_oom);
    memset(msgs, 0, sizeof(struct llama_chat_message) * (size_t) max_msgs);

    int n_msgs = 0;
    char *synthetic_system = build_system_content(env, argv[1], NULL);
    if (synthetic_system) {
        char *role = enif_alloc(7);
        if (!role) {
            enif_free(synthetic_system);
            enif_free(msgs);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        memcpy(role, "system", 7);
        msgs[0].role = role;
        msgs[0].content = synthetic_system;
        n_msgs = 1;
    }

    int built = build_chat_msgs_from_list(
        env, messages_term, msgs, max_msgs, n_msgs);
    if (built < 0) {
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        switch (built) {
            case -2: return enif_make_tuple2(env, atom_error, atom_oom);
            case -3: return enif_make_tuple2(env, atom_error, atom_invalid_content);
            default: return enif_make_badarg(env);
        }
    }
    n_msgs = built;

    pthread_mutex_lock(&m->mu);
    if (!m->model || m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const char *tmpl = erllama_safe_model_chat_template(m->model, NULL);
    if (!tmpl || tmpl[0] == '\0') {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        return enif_make_tuple2(env, atom_error, atom_no_template);
    }

    /* Render. Start with a 4 KiB buffer; grow on negative-needed-size. */
    int32_t buf_size = 4096;
    char *buf = enif_alloc((size_t) buf_size);
    if (!buf) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t written = erllama_safe_chat_apply_template(
        tmpl, msgs, (size_t) n_msgs, true, buf, buf_size);
    if (written < 0 && written != INT32_MIN) {
        int32_t needed = -written;
        if (needed > (int32_t) ERLLAMA_MAX_TOKEN_TEXT) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
        enif_free(buf);
        buf_size = needed + 16;
        buf = enif_alloc((size_t) buf_size);
        if (!buf) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        written = erllama_safe_chat_apply_template(
            tmpl, msgs, (size_t) n_msgs, true, buf, buf_size);
    }
    if (written < 0) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        enif_free(buf);
        return enif_make_tuple2(env, atom_error,
                                written == INT32_MIN ? atom_exception
                                                     : atom_template_failed);
    }

    /* Tokenise the rendered string. parse_special=true so chat-template
     * tokens (`<|user|>`, `<|im_start|>`, etc.) become their special
     * token ids rather than text fragments. */
    const struct llama_vocab *vocab = erllama_safe_model_get_vocab(m->model);
    if (!vocab) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        enif_free(buf);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    int32_t n_max = written + 8;
    if (n_max < 16) n_max = 16;
    if (n_max > ERLLAMA_MAX_TOKENS) n_max = ERLLAMA_MAX_TOKENS;
    llama_token *tokens = enif_alloc(sizeof(llama_token) * (size_t) n_max);
    if (!tokens) {
        pthread_mutex_unlock(&m->mu);
        free_chat_msgs(msgs, n_msgs);
        enif_free(msgs);
        enif_free(buf);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t n = erllama_safe_tokenize(vocab, buf, written, tokens, n_max,
                                      true, true);
    if (n < 0 && n != INT32_MIN) {
        int32_t needed = -n;
        if (needed > ERLLAMA_MAX_TOKENS) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
        enif_free(tokens);
        tokens = enif_alloc(sizeof(llama_token) * (size_t) needed);
        if (!tokens) {
            pthread_mutex_unlock(&m->mu);
            free_chat_msgs(msgs, n_msgs);
            enif_free(msgs);
            enif_free(buf);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        n = erllama_safe_tokenize(vocab, buf, written, tokens, needed,
                                  true, true);
    }
    pthread_mutex_unlock(&m->mu);
    free_chat_msgs(msgs, n_msgs);
    enif_free(msgs);
    enif_free(buf);
    if (n == INT32_MIN) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (n < 0) {
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_tokenize_failed);
    }

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, tokens[i]), list);
    }
    enif_free(tokens);
    return enif_make_tuple2(env, atom_ok, list);
}

/* =========================================================================
 * Embeddings
 * =========================================================================
 *
 * Decodes a token list with the embeddings flag flipped on, then
 * reads the per-sequence pooled vector via llama_get_embeddings_seq.
 * Falls back to llama_get_embeddings (last-token) for models whose
 * pooling_type is NONE. The context must have been opened with
 * embeddings = true at new_context/2 time, otherwise the underlying
 * llama_decode allocates causal-LM logits buffers and the
 * embeddings reads return NULL.
 */
static ERL_NIF_TERM nif_embed(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    llama_token *tokens = NULL;
    int32_t n = 0;
    int rc = read_token_list(env, argv[1], &tokens, &n);
    if (rc != 1) return token_list_error(env, rc);
    if (n == 0) {
        if (tokens) enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_invalid_token);
    }

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_model *model = erllama_safe_get_model(c->ctx);
    const struct llama_vocab *vocab =
        model ? erllama_safe_model_get_vocab(model) : NULL;
    int32_t n_vocab = vocab ? erllama_safe_vocab_n_tokens(vocab) : 0;
    if (n_vocab <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    for (int32_t i = 0; i < n; i++) {
        if (tokens[i] >= n_vocab) {
            pthread_mutex_unlock(&c->mu);
            enif_free(tokens);
            return enif_make_tuple2(env, atom_error, atom_invalid_token);
        }
    }
    int32_t n_embd = erllama_safe_n_embd(model);
    if (n_embd <= 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_embed_failed);
    }
    /* Bounds-check before touching context state. Same SIGSEGV path
     * as nif_prefill: llama_decode walks past the KV slab when
     * n_tokens >= n_ctx, undefined when n_tokens > n_batch. */
    uint32_t n_ctx = erllama_safe_n_ctx(c->ctx);
    uint32_t n_batch = erllama_safe_n_batch(c->ctx);
    if (n_ctx == 0 || n_batch == 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if ((uint32_t) n >= n_ctx) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_context_overflow);
    }
    if ((uint32_t) n > n_batch) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_batch_overflow);
    }

    /* Flip on embeddings for this call; the caller may have left it
     * off for normal causal-lm decode. We do not flip it back here -
     * the next decode_one call would read garbage logits. The model
     * layer is responsible for using a dedicated context for
     * embeddings, or for arranging not to mix modes on the same ctx. */
    if (erllama_safe_set_embeddings(c->ctx, true) != 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }

    struct llama_batch batch = llama_batch_get_one(tokens, n);
    int dr = erllama_safe_decode(c->ctx, batch);
    if (dr == ERLLAMA_DECODE_EXC_SENTINEL) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    if (dr != 0) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, enif_make_int(env, dr));
    }
    /* The `decode_ready` flag implies "logits are ready for sampling";
     * after an embeddings decode the logits buffer is repurposed and a
     * follow-on decode_one would crash. Force it off so the model
     * layer must explicitly re-prefill before sampling. */
    c->decode_ready = 0;

    /* Try the pooled vector first; fall back to last-token. */
    float *embd = erllama_safe_get_embeddings_seq(c->ctx, 0);
    if (!embd) {
        embd = erllama_safe_get_embeddings(c->ctx);
    }
    if (!embd) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_embed_failed);
    }

    /* Copy the floats out of the context-owned buffer before unlocking. */
    double *vec = enif_alloc(sizeof(double) * (size_t) n_embd);
    if (!vec) {
        pthread_mutex_unlock(&c->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    for (int32_t i = 0; i < n_embd; i++) vec[i] = (double) embd[i];
    pthread_mutex_unlock(&c->mu);
    enif_free(tokens);

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int32_t i = n_embd - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_double(env, vec[i]), list);
    }
    enif_free(vec);
    return enif_make_tuple2(env, atom_ok, list);
}

/* =========================================================================
 * Sampler config
 *
 * configure_sampler/2 is the one entry point that builds the per-context
 * sampler chain. It accepts a config map carrying any of: grammar,
 * repetition_penalty, top_k, top_p, min_p, temperature, seed. Missing
 * fields are skipped; a temperature of 0.0 (or no sampling params at
 * all) ends the chain in greedy.
 *
 * set_grammar/2 is a backwards-compatible alias that builds the same
 * chain with only a grammar entry. clear_sampler/1 drops the cached
 * chain so the next decode_one lazy-inits greedy.
 * ========================================================================= */

static struct llama_sampler *build_default_greedy_chain(void) {
    struct llama_sampler_chain_params sp =
        llama_sampler_chain_default_params();
    struct llama_sampler *chain = erllama_safe_sampler_chain_init(sp);
    if (!chain) return NULL;
    struct llama_sampler *greedy = erllama_safe_sampler_init_greedy();
    if (!greedy) {
        (void) erllama_safe_sampler_free(chain);
        return NULL;
    }
    if (erllama_safe_sampler_chain_add(chain, greedy) != 0) {
        (void) erllama_safe_sampler_free(greedy);
        (void) erllama_safe_sampler_free(chain);
        return NULL;
    }
    return chain;
}

/* Append one stage to a chain, freeing the chain and returning NULL on
 * failure so callers can write a tight cleanup ladder. */
static int chain_append(struct llama_sampler *chain,
                        struct llama_sampler *stage) {
    if (!stage) return -1;
    if (erllama_safe_sampler_chain_add(chain, stage) != 0) {
        (void) erllama_safe_sampler_free(stage);
        return -1;
    }
    return 0;
}

/* Build a sampler chain from a config map. On failure returns NULL and
 * sets *out_err_atom to one of: atom_oom, atom_grammar_failed,
 * atom_badarg. The lock must already be held by the caller (vocab
 * lookup uses c->ctx). */
static struct llama_sampler *
build_sampler_chain_from_map(ErlNifEnv *env, ERL_NIF_TERM cfg,
                             struct llama_context *ctx,
                             ERL_NIF_TERM *out_err_atom) {
    if (!enif_is_map(env, cfg)) {
        *out_err_atom = enif_make_atom(env, "badarg");
        return NULL;
    }

    /* Grammar requires the vocab; everything else does not. */
    ErlNifBinary grammar_bin;
    int has_grammar = 0;
    {
        ERL_NIF_TERM v;
        if (enif_get_map_value(env, cfg, enif_make_atom(env, "grammar"), &v)) {
            if (!enif_inspect_iolist_as_binary(env, v, &grammar_bin) ||
                grammar_bin.size == 0) {
                *out_err_atom = enif_make_atom(env, "badarg");
                return NULL;
            }
            has_grammar = 1;
        }
    }

    int32_t i32;
    double f64;
    int has_top_k = get_map_int31(env, cfg, "top_k", &i32);
    int32_t top_k_val = has_top_k ? i32 : 0;

    int has_top_p = get_map_double(env, cfg, "top_p", &f64);
    double top_p_val = has_top_p ? f64 : 1.0;

    int has_min_p = get_map_double(env, cfg, "min_p", &f64);
    double min_p_val = has_min_p ? f64 : 0.0;

    int has_temp = get_map_double(env, cfg, "temperature", &f64);
    double temp_val = has_temp ? f64 : 0.0;

    int has_rep = get_map_double(env, cfg, "repetition_penalty", &f64);
    double rep_val = has_rep ? f64 : 1.0;

    uint32_t seed_val = 0;
    int has_seed = 0;
    {
        ERL_NIF_TERM v;
        if (enif_get_map_value(env, cfg, enif_make_atom(env, "seed"), &v)) {
            unsigned long seed_ul;
            if (!enif_get_ulong(env, v, &seed_ul)) {
                *out_err_atom = enif_make_atom(env, "badarg");
                return NULL;
            }
            seed_val = (uint32_t) seed_ul;
            has_seed = 1;
        }
    }

    struct llama_sampler_chain_params sp =
        llama_sampler_chain_default_params();
    struct llama_sampler *chain = erllama_safe_sampler_chain_init(sp);
    if (!chain) {
        *out_err_atom = atom_oom;
        return NULL;
    }

    if (has_grammar) {
        const struct llama_model *model = erllama_safe_get_model(ctx);
        const struct llama_vocab *vocab =
            model ? erllama_safe_model_get_vocab(model) : NULL;
        if (!vocab) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_exception;
            return NULL;
        }
        char *gstr = enif_alloc(grammar_bin.size + 1);
        if (!gstr) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
        memcpy(gstr, grammar_bin.data, grammar_bin.size);
        gstr[grammar_bin.size] = '\0';
        struct llama_sampler *g =
            erllama_safe_sampler_init_grammar(vocab, gstr, "root");
        enif_free(gstr);
        if (!g) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_grammar_failed;
            return NULL;
        }
        if (chain_append(chain, g) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
    }

    if (has_rep && rep_val != 1.0) {
        if (chain_append(chain,
                         erllama_safe_sampler_init_penalties(
                             64, (float) rep_val, 0.0f, 0.0f)) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
    }
    if (has_top_k && top_k_val > 0) {
        if (chain_append(chain,
                         erllama_safe_sampler_init_top_k(top_k_val)) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
    }
    if (has_top_p && top_p_val < 1.0) {
        if (chain_append(chain,
                         erllama_safe_sampler_init_top_p((float) top_p_val,
                                                         1)) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
    }
    if (has_min_p && min_p_val > 0.0) {
        if (chain_append(chain,
                         erllama_safe_sampler_init_min_p((float) min_p_val,
                                                         1)) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
    }
    if (has_temp && temp_val > 0.0) {
        if (chain_append(chain,
                         erllama_safe_sampler_init_temp((float) temp_val)) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
        if (chain_append(chain,
                         erllama_safe_sampler_init_dist(seed_val)) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
    } else {
        /* temperature == 0 or absent: greedy terminal. */
        if (chain_append(chain, erllama_safe_sampler_init_greedy()) != 0) {
            (void) erllama_safe_sampler_free(chain);
            *out_err_atom = atom_oom;
            return NULL;
        }
        (void) has_seed; /* seed without temperature is a no-op. */
    }

    return chain;
}

static ERL_NIF_TERM nif_configure_sampler(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }

    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    ERL_NIF_TERM err = atom_oom;
    struct llama_sampler *chain =
        build_sampler_chain_from_map(env, argv[1], c->ctx, &err);
    if (!chain) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, err);
    }
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
    }
    c->smpl = chain;
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}

/* Backwards-compatible: builds a chain with only a grammar entry. */
static ERL_NIF_TERM nif_set_grammar(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void) argc;
    ERL_NIF_TERM cfg = enif_make_new_map(env);
    enif_make_map_put(env, cfg, enif_make_atom(env, "grammar"), argv[1], &cfg);
    ERL_NIF_TERM new_argv[2] = {argv[0], cfg};
    return nif_configure_sampler(env, 2, new_argv);
}

static ERL_NIF_TERM nif_clear_sampler(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (c->smpl) {
        (void) erllama_safe_sampler_free(c->smpl);
        c->smpl = NULL;
    }
    pthread_mutex_unlock(&c->mu);
    return atom_ok;
}

/* =========================================================================
 * LoRA adapters
 * ========================================================================= */

static ERL_NIF_TERM nif_adapter_load(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_model_t *m;
    if (!enif_get_resource(env, argv[0], MODEL_RT, (void **) &m)) {
        return enif_make_badarg(env);
    }
    char path[4097];
    if (!copy_path(env, argv[1], path, sizeof(path))) {
        return enif_make_badarg(env);
    }

    pthread_mutex_lock(&m->mu);
    if (!m->model) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    /* Mirror nif_new_context: refuse to attach a new adapter once
     * free_model/1 has flagged the model for deferred release.
     * Otherwise a new wrapper could resurrect an outgoing model. */
    if (m->release_pending) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    struct llama_adapter_lora *adapter =
        erllama_safe_adapter_lora_init(m->model, path);
    if (!adapter) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_load_failed);
    }
    erllama_adapter_t *res =
        enif_alloc_resource(ADAPTER_RT, sizeof(*res));
    if (!res) {
        erllama_safe_adapter_lora_free(adapter);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        erllama_safe_adapter_lora_free(adapter);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    res->mu_inited = 1;
    res->adapter = adapter;
    res->model_res = m;
    m->active_adapters++;
    enif_keep_resource(m);
    pthread_mutex_unlock(&m->mu);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

static ERL_NIF_TERM nif_adapter_free(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_adapter_t *a;
    if (!enif_get_resource(env, argv[0], ADAPTER_RT, (void **) &a)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&a->mu);
    if (!a->adapter) {
        pthread_mutex_unlock(&a->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    erllama_safe_adapter_lora_free(a->adapter);
    a->adapter = NULL;
    pthread_mutex_unlock(&a->mu);
    return atom_ok;
}

/* Install a set of adapters with scales on a context. Takes a list of
 * {AdapterRes, Scale} pairs; an empty list detaches everything.
 * The model layer is responsible for tracking the current attachment
 * set; this NIF just plumbs through to llama_set_adapters_lora.
 *
 * Concurrency: every unique adapter wrapper's mu is held for the
 * full duration of the llama call. nif_adapter_free, which also
 * takes a->mu, is therefore blocked from racing the read of
 * a->adapter against its use in the native call. To avoid AB-BA
 * between two concurrent set_adapters callers passing overlapping
 * adapter sets in different orders, locks are taken in pointer
 * order (qsort by wrapper address). The user-supplied list order
 * is preserved in the arrays passed to llama via orig_idx. */
typedef struct {
    erllama_adapter_t *w;
    float scale;
    unsigned orig_idx;
} adapter_entry_t;

static int adapter_entry_cmp(const void *a, const void *b) {
    erllama_adapter_t *aw = ((const adapter_entry_t *) a)->w;
    erllama_adapter_t *bw = ((const adapter_entry_t *) b)->w;
    if (aw < bw) return -1;
    if (aw > bw) return 1;
    return 0;
}

static ERL_NIF_TERM nif_set_adapters(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM list = argv[1];
    unsigned n;
    if (!enif_get_list_length(env, list, &n)) {
        return enif_make_badarg(env);
    }

    struct llama_adapter_lora **adapters = NULL;
    float *scales = NULL;
    adapter_entry_t *entries = NULL;
    if (n > 0) {
        adapters = enif_alloc(sizeof(*adapters) * n);
        scales = enif_alloc(sizeof(*scales) * n);
        entries = enif_alloc(sizeof(*entries) * n);
        if (!adapters || !scales || !entries) {
            if (adapters) enif_free(adapters);
            if (scales) enif_free(scales);
            if (entries) enif_free(entries);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
    }

    /* Resolve the list in user order. No locks taken yet: we just
     * gather the wrapper pointers and scales. */
    ERL_NIF_TERM head, tail = list;
    unsigned i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        int arity;
        const ERL_NIF_TERM *pair;
        if (!enif_get_tuple(env, head, &arity, &pair) || arity != 2) {
            goto badarg;
        }
        erllama_adapter_t *a;
        if (!enif_get_resource(env, pair[0], ADAPTER_RT, (void **) &a)) {
            goto badarg;
        }
        double scale;
        if (!enif_get_double(env, pair[1], &scale)) {
            long ll;
            if (enif_get_long(env, pair[1], &ll)) {
                scale = (double) ll;
            } else {
                goto badarg;
            }
        }
        entries[i].w = a;
        entries[i].scale = (float) scale;
        entries[i].orig_idx = i;
        i++;
    }

    /* Sort by wrapper pointer so locks always go in a consistent
     * order across concurrent set_adapters callers. Same wrapper
     * appearing twice (caller error or duplicate-scale convention)
     * collapses to one lock acquisition; the user-supplied
     * adapters[]/scales[] still receive both entries via orig_idx. */
    if (n > 1) qsort(entries, n, sizeof(*entries), adapter_entry_cmp);

    /* Lock each unique wrapper in sorted order. On a released
     * adapter, unlock everything held so far and bail out. */
    unsigned k = 0;
    for (k = 0; k < n; k++) {
        if (k > 0 && entries[k].w == entries[k - 1].w) continue;
        pthread_mutex_lock(&entries[k].w->mu);
        if (!entries[k].w->adapter) {
            for (unsigned j = 0; j <= k; j++) {
                if (j > 0 && entries[j].w == entries[j - 1].w) continue;
                pthread_mutex_unlock(&entries[j].w->mu);
            }
            if (adapters) enif_free(adapters);
            if (scales) enif_free(scales);
            if (entries) enif_free(entries);
            return enif_make_tuple2(env, atom_error, atom_released);
        }
    }

    /* Build native arrays in the user's original order. All
     * a->adapter reads happen under the corresponding a->mu held
     * above, so a concurrent nif_adapter_free cannot null any of
     * these pointers between read and use. */
    for (k = 0; k < n; k++) {
        adapters[entries[k].orig_idx] = entries[k].w->adapter;
        scales[entries[k].orig_idx] = entries[k].scale;
    }

    pthread_mutex_lock(&c->mu);
    int rc;
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        for (k = 0; k < n; k++) {
            if (k > 0 && entries[k].w == entries[k - 1].w) continue;
            pthread_mutex_unlock(&entries[k].w->mu);
        }
        if (adapters) enif_free(adapters);
        if (scales) enif_free(scales);
        if (entries) enif_free(entries);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    rc = erllama_safe_set_adapters_lora(c->ctx, adapters, n, scales);
    pthread_mutex_unlock(&c->mu);

    for (k = 0; k < n; k++) {
        if (k > 0 && entries[k].w == entries[k - 1].w) continue;
        pthread_mutex_unlock(&entries[k].w->mu);
    }

    if (adapters) enif_free(adapters);
    if (scales) enif_free(scales);
    if (entries) enif_free(entries);

    if (rc != 0) {
        return enif_make_tuple2(env, atom_error, atom_exception);
    }
    return atom_ok;

badarg:
    if (adapters) enif_free(adapters);
    if (scales) enif_free(scales);
    if (entries) enif_free(entries);
    return enif_make_badarg(env);
}

/* =========================================================================
 * Per-request sampler resource (Phase 4 infrastructure)
 *
 * Wraps a llama_sampler_chain built by build_sampler_chain_from_map
 * so multiple in-flight requests (v0.2+) can hold independent chains.
 * The v0.1 model layer still uses configure_sampler/2 against the
 * context's cached `c->smpl`; this resource is the building block
 * for the eventual decode_and_sample_batch NIF.
 * ========================================================================= */

static ERL_NIF_TERM nif_sampler_new(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    if (!enif_is_map(env, argv[1])) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&c->mu);
    if (!c->ctx) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    ERL_NIF_TERM err = atom_oom;
    struct llama_sampler *chain =
        build_sampler_chain_from_map(env, argv[1], c->ctx, &err);
    pthread_mutex_unlock(&c->mu);
    if (!chain) {
        return enif_make_tuple2(env, atom_error, err);
    }
    erllama_sampler_t *res = enif_alloc_resource(SAMPLER_RT, sizeof(*res));
    if (!res) {
        (void) erllama_safe_sampler_free(chain);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    memset(res, 0, sizeof(*res));
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        (void) erllama_safe_sampler_free(chain);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    res->mu_inited = 1;
    res->chain = chain;
    res->ctx_res = c;
    enif_keep_resource(c);

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

static ERL_NIF_TERM nif_sampler_free(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    (void) argc;
    erllama_sampler_t *s;
    if (!enif_get_resource(env, argv[0], SAMPLER_RT, (void **) &s)) {
        return enif_make_badarg(env);
    }
    pthread_mutex_lock(&s->mu);
    if (!s->chain) {
        pthread_mutex_unlock(&s->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    (void) erllama_safe_sampler_free(s->chain);
    s->chain = NULL;
    pthread_mutex_unlock(&s->mu);
    return atom_ok;
}

static ErlNifFunc nif_funcs[] = {
    {"nif_crc32c",       1, nif_crc32c,       ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack",      3, nif_kv_pack,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack",      4, nif_kv_pack,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_unpack",    3, nif_kv_unpack,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_seq_rm",    4, nif_kv_seq_rm,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_fsync_dir",    1, nif_fsync_dir,    ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_load_model",   2, nif_load_model,   ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_free_model",   1, nif_free_model,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_new_context",  2, nif_new_context,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_free_context", 1, nif_free_context, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_tokenize",     3, nif_tokenize,     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_prefill",      2, nif_prefill,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_decode_one",   1, nif_decode_one,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_detokenize",   2, nif_detokenize,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_apply_chat_template", 2, nif_apply_chat_template, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_embed",        2, nif_embed,        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_set_grammar",  2, nif_set_grammar,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_configure_sampler", 2, nif_configure_sampler,
        ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_clear_sampler", 1, nif_clear_sampler, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_adapter_load", 2, nif_adapter_load, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_adapter_free", 1, nif_adapter_free, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_set_adapters", 2, nif_set_adapters, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_sampler_new",  2, nif_sampler_new,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_sampler_free", 1, nif_sampler_free, ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(erllama_nif, nif_funcs, load, NULL, NULL, unload)
