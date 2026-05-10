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
extern int erllama_safe_model_free(struct llama_model *m);
extern struct llama_context *erllama_safe_init_from_model(
    struct llama_model *m, struct llama_context_params params);
extern int erllama_safe_free(struct llama_context *c);
extern const struct llama_model *erllama_safe_get_model(
    const struct llama_context *c);
extern const struct llama_vocab *erllama_safe_model_get_vocab(
    const struct llama_model *m);
extern int32_t erllama_safe_vocab_n_tokens(const struct llama_vocab *v);
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
static ERL_NIF_TERM atom_context_failed;
static ERL_NIF_TERM atom_tokenize_failed;
static ERL_NIF_TERM atom_pack_failed;
static ERL_NIF_TERM atom_unpack_failed;
static ERL_NIF_TERM atom_true;
static ERL_NIF_TERM atom_false;
static ERL_NIF_TERM atom_released;
static ERL_NIF_TERM atom_too_large;
static ERL_NIF_TERM atom_invalid_token;
static ERL_NIF_TERM atom_oom;
static ERL_NIF_TERM atom_deferred;
static ERL_NIF_TERM atom_exception;
static ERL_NIF_TERM atom_no_logits;

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
    int release_pending;           /* free_model when active_contexts hit 0 */
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

static ErlNifResourceType *MODEL_RT;
static ErlNifResourceType *CTX_RT;

/* Drop the context's reference on its model; if a previous
 * free_model/1 returned {ok, deferred} and the model is now
 * unreferenced, actually free the underlying llama_model* here. The
 * decision is made under the lock so concurrent context destructions
 * can't double-free. The free itself runs while the lock is still
 * held to keep the pointer non-observable mid-teardown. */
static void context_drops_model(erllama_model_t *m) {
    pthread_mutex_lock(&m->mu);
    if (m->active_contexts > 0) {
        m->active_contexts--;
    }
    if (m->release_pending && m->active_contexts == 0 && m->model) {
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
    atom_context_failed = enif_make_atom(env, "context_failed");
    atom_tokenize_failed = enif_make_atom(env, "tokenize_failed");
    atom_pack_failed = enif_make_atom(env, "pack_failed");
    atom_unpack_failed = enif_make_atom(env, "unpack_failed");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
    atom_released = enif_make_atom(env, "released");
    atom_too_large = enif_make_atom(env, "too_large");
    atom_invalid_token = enif_make_atom(env, "invalid_token");
    atom_oom = enif_make_atom(env, "oom");
    atom_deferred = enif_make_atom(env, "deferred");
    atom_exception = enif_make_atom(env, "exception");
    atom_no_logits = enif_make_atom(env, "no_logits");

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

    struct llama_model *model = erllama_safe_model_load_from_file(path, params);
    if (!model) {
        return enif_make_tuple2(env, atom_error, atom_load_failed);
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
 *   {ok, deferred}     -> contexts still hold this model; release flagged.
 *                         The last context destruction performs the actual
 *                         llama_model_free under context_drops_model.
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
    if (m->active_contexts > 0) {
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
    (void) argc;
    erllama_context_t *c;
    if (!enif_get_resource(env, argv[0], CTX_RT, (void **) &c)) {
        return enif_make_badarg(env);
    }
    /* Tokens (argv[1]) is informational; NTokens (argv[2]) ignored.
     * The model layer must have prefilled exactly the desired prefix
     * before calling kv_pack. */
    llama_seq_id seq_id = 0;

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

    /* Lazy-init the sampler chain on first use; subsequent calls
     * reuse it. Re-allocating chain + greedy + add per token costs
     * ~1 us each on macOS but adds up across long generations. */
    if (!c->smpl) {
        struct llama_sampler_chain_params sp =
            llama_sampler_chain_default_params();
        struct llama_sampler *chain = erllama_safe_sampler_chain_init(sp);
        if (!chain) {
            pthread_mutex_unlock(&c->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        struct llama_sampler *greedy = erllama_safe_sampler_init_greedy();
        if (!greedy) {
            (void) erllama_safe_sampler_free(chain);
            pthread_mutex_unlock(&c->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        if (erllama_safe_sampler_chain_add(chain, greedy) != 0) {
            (void) erllama_safe_sampler_free(greedy);
            (void) erllama_safe_sampler_free(chain);
            pthread_mutex_unlock(&c->mu);
            return enif_make_tuple2(env, atom_error, atom_oom);
        }
        c->smpl = chain;
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

static ErlNifFunc nif_funcs[] = {
    {"nif_crc32c",       1, nif_crc32c,       ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack",      3, nif_kv_pack,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
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
    {"nif_detokenize",   2, nif_detokenize,   ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(erllama_nif, nif_funcs, load, NULL, NULL, unload)
