/*
 * erllama_nif: the single NIF for erllama (cache + future llama.cpp
 * surface).
 *
 * v0.1 surface:
 *   crc32c(IoData) -> non_neg_integer()    [dirty CPU]
 *
 * Stubs returning {error, not_implemented} until the llama.cpp
 * wiring lands:
 *   kv_pack(Ctx, Tokens, NTokens) -> Binary
 *   kv_unpack(Ctx, Binary, SeqId) -> ok | {error, _}
 *
 * Future additions live here too (single .so): load_model,
 * free_model, new_context, free_context, tokenize, detokenize,
 * prefill, decode_async.
 *
 * No file I/O happens here; the cache layer assembles framed .kvc
 * files in Erlang and only feeds opaque payload bytes to the NIF.
 */
#include <erl_nif.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

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

/* Map a few common errno values to atoms; everything else maps to
 * `unknown` (the actual integer is not exposed since the NIF doesn't
 * link erl_driver). */
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

/* fsync(2) on a directory: opens path read-only, fsyncs the fd, closes.
 * Used by the disk tier to make link(2)/unlink(2) operations on the
 * directory entry durable across a kernel crash. */
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

static ErlNifFunc nif_funcs[] = {
    {"nif_crc32c", 1, nif_crc32c, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_pack", 3, nif_kv_pack, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_kv_unpack", 3, nif_kv_unpack, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_fsync_dir", 1, nif_fsync_dir, ERL_NIF_DIRTY_JOB_IO_BOUND}
};

ERL_NIF_INIT(erllama_nif, nif_funcs, load, NULL, NULL, NULL)
