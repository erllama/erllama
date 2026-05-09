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
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "crc32c.h"
#include "llama.h"

#ifndef ERLLAMA_MAX_TOKENS
/* Cap on accepted token-list inputs and tokenize output. INT32_MAX/2
 * is well above the largest practical context window and keeps token
 * count multiplications away from int32_t overflow. */
#define ERLLAMA_MAX_TOKENS (INT32_MAX / 2)
#endif

#ifndef ERLLAMA_MAX_TOKEN_TEXT
/* Largest text accepted by tokenize/3 (bytes). Rejects requests whose
 * length wouldn't fit in int32_t even before signedness concerns. */
#define ERLLAMA_MAX_TOKEN_TEXT (256 * 1024 * 1024)
#endif

/* =========================================================================
 * Atoms
 * ========================================================================= */

static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_error;
static ERL_NIF_TERM atom_not_implemented;
static ERL_NIF_TERM atom_load_failed;
static ERL_NIF_TERM atom_context_failed;
static ERL_NIF_TERM atom_tokenize_failed;
static ERL_NIF_TERM atom_pack_failed;
static ERL_NIF_TERM atom_unpack_failed;
static ERL_NIF_TERM atom_true;
static ERL_NIF_TERM atom_false;
static ERL_NIF_TERM atom_in_use;
static ERL_NIF_TERM atom_released;
static ERL_NIF_TERM atom_too_large;
static ERL_NIF_TERM atom_invalid_token;
static ERL_NIF_TERM atom_oom;

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
    struct llama_model *model;     /* NULL after release */
    int active_contexts;           /* nif_new_context bumps; ctx_dtor decrements */
} erllama_model_t;

typedef struct {
    pthread_mutex_t mu;
    struct llama_context *ctx;     /* NULL after release */
    erllama_model_t *model_res;    /* keep_resource'd by new_context */
} erllama_context_t;

static ErlNifResourceType *MODEL_RT;
static ErlNifResourceType *CTX_RT;

static void model_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_model_t *m = (erllama_model_t *) obj;
    if (m->model) {
        llama_model_free(m->model);
        m->model = NULL;
    }
    pthread_mutex_destroy(&m->mu);
}

static void ctx_dtor(ErlNifEnv *env, void *obj) {
    (void) env;
    erllama_context_t *c = (erllama_context_t *) obj;
    if (c->ctx) {
        llama_free(c->ctx);
        c->ctx = NULL;
    }
    if (c->model_res) {
        pthread_mutex_lock(&c->model_res->mu);
        if (c->model_res->active_contexts > 0) {
            c->model_res->active_contexts--;
        }
        pthread_mutex_unlock(&c->model_res->mu);
        enif_release_resource(c->model_res);
        c->model_res = NULL;
    }
    pthread_mutex_destroy(&c->mu);
}

/* =========================================================================
 * Load callback
 * ========================================================================= */

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void) priv_data;
    (void) load_info;

    erllama_crc32c_init();
    llama_backend_init();

    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_not_implemented = enif_make_atom(env, "not_implemented");
    atom_load_failed = enif_make_atom(env, "load_failed");
    atom_context_failed = enif_make_atom(env, "context_failed");
    atom_tokenize_failed = enif_make_atom(env, "tokenize_failed");
    atom_pack_failed = enif_make_atom(env, "pack_failed");
    atom_unpack_failed = enif_make_atom(env, "unpack_failed");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
    atom_in_use = enif_make_atom(env, "in_use");
    atom_released = enif_make_atom(env, "released");
    atom_too_large = enif_make_atom(env, "too_large");
    atom_invalid_token = enif_make_atom(env, "invalid_token");
    atom_oom = enif_make_atom(env, "oom");

    MODEL_RT = enif_open_resource_type(
        env, NULL, "erllama_model", model_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!MODEL_RT) return -1;

    CTX_RT = enif_open_resource_type(
        env, NULL, "erllama_context", ctx_dtor, ERL_NIF_RT_CREATE, NULL);
    if (!CTX_RT) return -1;

    return 0;
}

/* =========================================================================
 * Helpers
 * ========================================================================= */

static int copy_path(ErlNifEnv *env, ERL_NIF_TERM term, char *out, size_t cap) {
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, term, &bin)) return 0;
    if (bin.size == 0 || bin.size >= cap) return 0;
    memcpy(out, bin.data, bin.size);
    out[bin.size] = '\0';
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

    struct llama_model_params params = llama_model_default_params();

    unsigned int u;
    if (get_map_uint(env, argv[1], "n_gpu_layers", &u)) {
        params.n_gpu_layers = (int32_t) u;
    }
    int b;
    if (get_map_bool(env, argv[1], "use_mmap", &b)) params.use_mmap = b ? true : false;
    if (get_map_bool(env, argv[1], "use_mlock", &b)) params.use_mlock = b ? true : false;
    if (get_map_bool(env, argv[1], "vocab_only", &b)) params.vocab_only = b ? true : false;

    struct llama_model *model = llama_model_load_from_file(path, params);
    if (!model) {
        return enif_make_tuple2(env, atom_error, atom_load_failed);
    }

    erllama_model_t *res = enif_alloc_resource(MODEL_RT, sizeof(*res));
    if (!res) {
        llama_model_free(model);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        llama_model_free(model);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    res->model = model;
    res->active_contexts = 0;

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return enif_make_tuple2(env, atom_ok, term);
}

/* free_model/1 returns:
 *   ok               -> released; subsequent ops on the term return error
 *   {error, in_use}  -> contexts still hold this model; will be freed when
 *                       the last context destructs (no-op-now is safe)
 *   {error, released}-> already released by a prior free_model call
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
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_in_use);
    }
    llama_model_free(m->model);
    m->model = NULL;
    pthread_mutex_unlock(&m->mu);
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
    if (get_map_uint(env, argv[1], "n_threads", &u)) params.n_threads = (int32_t) u;
    if (get_map_uint(env, argv[1], "n_threads_batch", &u)) {
        params.n_threads_batch = (int32_t) u;
    }
    int b;
    if (get_map_bool(env, argv[1], "embeddings", &b)) params.embeddings = b ? true : false;
    if (get_map_bool(env, argv[1], "offload_kqv", &b)) params.offload_kqv = b ? true : false;

    pthread_mutex_lock(&m->mu);
    if (!m->model) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    struct llama_context *ctx = llama_init_from_model(m->model, params);
    if (!ctx) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_context_failed);
    }
    erllama_context_t *res = enif_alloc_resource(CTX_RT, sizeof(*res));
    if (!res) {
        llama_free(ctx);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    if (pthread_mutex_init(&res->mu, NULL) != 0) {
        enif_release_resource(res);
        llama_free(ctx);
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
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
    llama_free(c->ctx);
    c->ctx = NULL;
    erllama_model_t *m = c->model_res;
    c->model_res = NULL;
    pthread_mutex_unlock(&c->mu);
    if (m) {
        pthread_mutex_lock(&m->mu);
        if (m->active_contexts > 0) m->active_contexts--;
        pthread_mutex_unlock(&m->mu);
        enif_release_resource(m);
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
    if (!m->model) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(m->model);

    int32_t text_len = (int32_t) text.size;
    int32_t n_max = text_len + 8;
    if (n_max < 16) n_max = 16;
    if (n_max > ERLLAMA_MAX_TOKENS) n_max = ERLLAMA_MAX_TOKENS;

    llama_token *tokens = (llama_token *) enif_alloc(sizeof(llama_token) * (size_t) n_max);
    if (!tokens) {
        pthread_mutex_unlock(&m->mu);
        return enif_make_tuple2(env, atom_error, atom_oom);
    }
    int32_t n = llama_tokenize(
        vocab, (const char *) text.data, text_len, tokens,
        n_max, add_special ? true : false, parse_special ? true : false);
    if (n < 0) {
        /* Negative return = -needed. INT32_MIN would overflow on
         * negation, so guard. */
        if (n == INT32_MIN) {
            enif_free(tokens);
            pthread_mutex_unlock(&m->mu);
            return enif_make_tuple2(env, atom_error, atom_too_large);
        }
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
        n = llama_tokenize(
            vocab, (const char *) text.data, text_len, tokens,
            needed, add_special ? true : false, parse_special ? true : false);
    }
    pthread_mutex_unlock(&m->mu);
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
    size_t need = llama_state_seq_get_size(c->ctx, seq_id);
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
    size_t written = llama_state_seq_get_data(c->ctx, out.data, out.size, seq_id);
    pthread_mutex_unlock(&c->mu);
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
    size_t consumed = llama_state_seq_set_data(
        c->ctx, in.data, in.size, (llama_seq_id) seq_id);
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
    llama_memory_t mem = llama_get_memory(c->ctx);
    if (!mem) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, atom_unpack_failed);
    }
    bool ok = llama_memory_seq_rm(mem, (llama_seq_id) seq_id,
                                  (llama_pos) p0, (llama_pos) p1);
    pthread_mutex_unlock(&c->mu);
    if (!ok) {
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
    struct llama_batch batch = llama_batch_get_one(tokens, n);
    int dr = llama_decode(c->ctx, batch);
    pthread_mutex_unlock(&c->mu);
    enif_free(tokens);
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

    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sp);
    if (!smpl) {
        pthread_mutex_unlock(&c->mu);
        return enif_make_tuple2(env, atom_error, enif_make_atom(env, "sampler_failed"));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_token tok = llama_sampler_sample(smpl, c->ctx, -1);
    llama_sampler_accept(smpl, tok);
    llama_sampler_free(smpl);

    const struct llama_model *model = llama_get_model(c->ctx);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int eog = llama_vocab_is_eog(vocab, tok) ? 1 : 0;

    llama_token tok_buf = tok;
    struct llama_batch batch = llama_batch_get_one(&tok_buf, 1);
    int rc = llama_decode(c->ctx, batch);
    pthread_mutex_unlock(&c->mu);
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
    if (!m->model) {
        pthread_mutex_unlock(&m->mu);
        enif_free(tokens);
        return enif_make_tuple2(env, atom_error, atom_released);
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(m->model);

    /* Per-token piece, concatenated. Pieces are typically a handful
     * of bytes; we grow the buffer on demand and re-call
     * llama_token_to_piece with a sized buffer when 256 bytes isn't
     * enough (it returns the negative needed size). */
    char small_piece[256];
    size_t cap = (size_t) n * 32 + 16;
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
        int32_t got = llama_token_to_piece(
            vocab, tokens[i], piece_buf, buf_size, 0, false);
        if (got < 0) {
            int32_t need = -got;
            if (need <= 0 || need > (1 << 20)) {
                err = 1;
                break;
            }
            grown = enif_alloc((size_t) need);
            if (!grown) { err = 2; break; }
            piece_buf = grown;
            got = llama_token_to_piece(
                vocab, tokens[i], piece_buf, need, 0, false);
            if (got < 0) {
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
        default:        name = "unknown";   break;
    }
    return enif_make_atom(env, name);
}

static ERL_NIF_TERM nif_fsync_dir(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    ErlNifBinary path_bin;
    if (!enif_inspect_iolist_as_binary(env, argv[0], &path_bin)) {
        return enif_make_badarg(env);
    }
    if (path_bin.size == 0 || path_bin.size > 4096) {
        return enif_make_badarg(env);
    }
    char path[4097];
    memcpy(path, path_bin.data, path_bin.size);
    path[path_bin.size] = '\0';

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
 * Suppress unused-stub warning for atom_not_implemented now that the
 * real API is in place.
 * ========================================================================= */
static ERL_NIF_TERM use_atom_not_implemented(void) {
    return atom_not_implemented;
}

static ErlNifFunc nif_funcs[] = {
    {"nif_crc32c",       1, nif_crc32c,       ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack",      3, nif_kv_pack,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_unpack",    3, nif_kv_unpack,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_seq_rm",    4, nif_kv_seq_rm,    ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_fsync_dir",    1, nif_fsync_dir,    ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_load_model",   2, nif_load_model,   ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_free_model",   1, nif_free_model,   0},
    {"nif_new_context",  2, nif_new_context,  ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_free_context", 1, nif_free_context, 0},
    {"nif_tokenize",     3, nif_tokenize,     ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_prefill",      2, nif_prefill,      ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_decode_one",   1, nif_decode_one,   ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_detokenize",   2, nif_detokenize,   ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(erllama_nif, nif_funcs, load, NULL, NULL, NULL)

/* The unused_stub function is referenced once to satisfy -Wunused. */
__attribute__((unused)) static void *_unused_anchor(void) {
    return (void *) use_atom_not_implemented;
}
