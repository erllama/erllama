/*
 * erllama_kv_nif: payload-side NIFs for the erllama_cache subsystem.
 *
 * Surface (v0.1):
 *   crc32c(IoData) -> non_neg_integer()    [dirty CPU]
 *
 * Stubs returning {error, not_implemented} until step 2b lands:
 *   kv_pack(Ctx, Tokens, NTokens) -> Binary
 *   kv_unpack(Ctx, Binary, SeqId) -> ok | {error, _}
 *
 * No file I/O happens here; the cache layer assembles framed .kvc
 * files in Erlang and only feeds opaque payload bytes to the NIF.
 */
#include <erl_nif.h>

#include <string.h>

#include "crc32c.h"

static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_error;
static ERL_NIF_TERM atom_not_implemented;

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void) priv_data;
    (void) load_info;

    erllama_crc32c_init();

    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_not_implemented = enif_make_atom(env, "not_implemented");

    return 0;
}

static ERL_NIF_TERM nif_crc32c(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, argv[0], &bin)) {
        return enif_make_badarg(env);
    }
    uint32_t crc = erllama_crc32c_update(0, bin.data, bin.size);
    return enif_make_uint(env, crc);
}

static ERL_NIF_TERM nif_kv_pack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    return enif_make_tuple2(env, atom_error, atom_not_implemented);
}

static ERL_NIF_TERM nif_kv_unpack(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void) argc;
    (void) argv;
    return enif_make_tuple2(env, atom_error, atom_not_implemented);
}

static ErlNifFunc nif_funcs[] = {
    {"nif_crc32c", 1, nif_crc32c, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack", 3, nif_kv_pack, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_unpack", 3, nif_kv_unpack, ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(erllama_kv_nif, nif_funcs, load, NULL, NULL, NULL)
