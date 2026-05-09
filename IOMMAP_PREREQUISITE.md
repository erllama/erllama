# iommap zero-copy region binary: upstream prerequisite for erllama

This document is a self-contained prompt for implementing a single new
primitive in the public iommap library. erllama's KV cache disk tier
depends on it for the load-side zero-copy path. The work is small and
should land as a single PR.

- Repository: github.com/benoitc/erlang-iommap
- Hex package: hex.pm/packages/iommap
- Public Erlang module: `iommap`

You are an expert in Erlang/OTP NIF design, memory management, and the
BEAM binary subsystem. We need to add one new primitive so a downstream
caller (an LLM KV cache) can obtain a refcounted load-side view into a
memory-mapped region, then hand that view directly to *another* NIF as
`ErlNifBinary` with no memcpy through the BEAM.

## What exists today (do not change)

`iommap:pread/3` already returns a binary, but it allocates a fresh
Erlang binary via `enif_make_new_binary` and `memcpy`s data out of the
mapped region. That is correct for general use but not for hot
zero-copy paths. Existing API and behaviour MUST stay intact:

```
open/2,3, close/1, pread/3, pwrite/3, sync/1,2,
truncate/2, advise/4, position/1
```

All NIFs except `position/1` are marked `ERL_NIF_DIRTY_JOB_IO_BOUND`;
keep that scheme.

## What to add

```erlang
-spec region_binary(Handle, Offset, Length)
        -> {ok, binary()} | {error, Reason} when
    Handle :: handle(),
    Offset :: non_neg_integer(),
    Length :: non_neg_integer(),
    Reason :: badarg | closed | out_of_bounds.
```

Returns a refcounted resource binary whose underlying memory is a
pointer + length into the iommap region. NO copy of the bytes is made.

## Lifetime: two NIF resources

Currently iommap likely uses one NIF resource per open handle, which
holds the fd, the mapping pointer, and the rwlock. That structure
cannot support `region_binary` correctly because:

- BEAM-side reachability of the *handle term* and BEAM-side
  reachability of a *region binary derived from the handle* are
  independent. Either may outlive the other.
- `enif_resource_refcount` is not part of the public API; the design
  must not depend on introspecting reference counts.

Refactor to two resources:

```c
typedef struct iommap_mapping {
    /* Owns the actual mmap region. Its destructor calls munmap. */
    int       fd;
    void*     base;       /* page-aligned mmap base */
    size_t    size;
    pthread_rwlock_t rwlock;
} iommap_mapping_t;

typedef struct iommap_handle {
    /* Owns the logical handle. Its destructor releases the mapping ref. */
    iommap_mapping_t* mapping;   /* held via enif_keep_resource */
    _Atomic int       closed;    /* 0 = open, 1 = logically closed */
} iommap_handle_t;
```

Resource types `IOMMAP_HANDLE_RT` and `IOMMAP_MAPPING_RT`, both
registered in `iommap_nif.c`'s `load/3`.

### `open/2,3`

Allocate one `iommap_mapping_t` (calls `mmap`) and one
`iommap_handle_t` that holds a reference to the mapping via
`enif_keep_resource`. The handle is the term returned to Erlang
(wrapped as a resource term).

### `close/1`

Sets `closed = 1` on the handle. Calls
`enif_release_resource(mapping)`. Does NOT touch `mapping->base` or
`munmap`. The handle term itself remains valid (subsequent operations
return `{error, closed}`); when the handle term is GC'd, the handle
destructor runs.

The mapping's destructor runs when the mapping's reference count
naturally falls to zero, i.e., when the handle has been released
(via close or GC of the handle term) AND every `region_binary`
derived from this handle has been GC'd. At that point `munmap`,
`close(fd)`, and `pthread_rwlock_destroy` run in the mapping
destructor.

### `region_binary/3`

```c
ERL_NIF_TERM region_binary_nif(ErlNifEnv* env, int argc,
                                const ERL_NIF_TERM argv[]) {
    iommap_handle_t* h;
    if (!enif_get_resource(env, argv[0], IOMMAP_HANDLE_RT, (void**)&h))
        return enif_make_badarg(env);
    if (atomic_load(&h->closed))
        return make_error(env, "closed");

    ErlNifUInt64 off, len;
    if (!enif_get_uint64(env, argv[1], &off) ||
        !enif_get_uint64(env, argv[2], &len))
        return enif_make_badarg(env);

    iommap_mapping_t* m = h->mapping;
    pthread_rwlock_rdlock(&m->rwlock);
    if (off + len > m->size) {
        pthread_rwlock_unlock(&m->rwlock);
        return make_error(env, "out_of_bounds");
    }
    /* Build the resource binary against the MAPPING resource, not the
       handle. The mapping retains itself across handle close. */
    ERL_NIF_TERM bin = enif_make_resource_binary(env, m,
                                                 (char*)m->base + off, len);
    pthread_rwlock_unlock(&m->rwlock);
    return enif_make_tuple2(env, atom_ok, bin);
}
```

`enif_make_resource_binary` increments the mapping's reference count
internally; the BEAM holds one reference per outstanding binary. When
all those binaries (and any sub-binaries) are GC'd, the mapping
destructor runs.

### Result

- Handle GC'd: handle destructor runs, calls
  `enif_release_resource(mapping)` if not already closed.
- Mapping has zero references: mapping destructor runs, calls
  `munmap`, `close(fd)`, `pthread_rwlock_destroy`.
- `close/1` is decoupled from `munmap`; outstanding region_binaries
  always remain valid.
- No `enif_resource_refcount` introspection.

Mark `region_binary` as `ERL_NIF_DIRTY_JOB_IO_BOUND`.

## Truncation safety: explicit non-goal

Once `region_binary/3` returns, the resulting binary is read by the
BEAM *outside* any NIF call: sub-binary creation, message send, GC,
binary hashing, ETS insert/lookup. If the underlying file is truncated
externally, those reads can hit unmapped pages and produce SIGBUS
**outside any handler this library can install**.

The library does NOT attempt to convert delayed SIGBUS into
`{error, sigbus}`. Document explicitly:

> `region_binary/3` is unsafe to use against files that may be
> truncated by external processes while a returned binary is
> reachable. The caller is responsible for ensuring exclusive access
> to the underlying file for the lifetime of any region_binary, or
> for accepting that external truncation may crash the VM.
>
> Callers needing safety against external mutation must use `pread/3`
> (which copies and is unaffected).

No SIGBUS handler is installed for this primitive.

## Tests

1. Round-trip: open, pwrite, region_binary, byte-equal to pread.
2. Zero-copy proof: 64 MiB region_binary, sub-binary in middle;
   allocated heap delta is O(metadata), not O(slice). `recon_alloc`
   or `process_info(self(), binary)`.
3. Lifetime, handle outlives binary: open, region_binary 4 KiB, drop
   the binary, GC, then `position/1` on the handle still works
   (mapping not unmapped).
4. Lifetime, binary outlives handle: open, region_binary 4 KiB,
   `close/1` the handle, then read from the binary; must return
   correct bytes (mapping kept alive by binary).
5. Drop binary AND close handle, then GC: a fresh `open/2` of the
   same path must succeed (mapping released).
6. Eviction race: N processes hold region_binaries; another process
   calls `close/1`; all binaries remain valid; assert no segfault.
7. Property test (proper): random open/region_binary/close/GC
   sequences never leak fds, never segfault on traffic that does not
   externally truncate, never produce wrong bytes.

## Documentation

- README.md: new "zero-copy region binaries" section. Include the
  truncation-safety warning verbatim. Document the two-resource
  lifetime so users understand why `close/1` no longer guarantees
  immediate `munmap`.
- moduledoc in `src/iommap.erl`: list the new function, repeat the
  warning.
- CHANGELOG: minor version bump (additive). Note the `close/1`
  behaviour change for handles with outstanding region_binaries.

## What NOT to do

- Do NOT change `pread/3` behaviour.
- Do NOT add a hugepages flag in this PR.
- Do NOT add MAP_ANONYMOUS.
- Do NOT introspect resource refcounts.
- Do NOT install a SIGBUS handler for region_binary.
- Do NOT batch this with other API additions. One primitive, one PR.

## Acceptance criteria

- `iommap:region_binary/3` exists, exported, type-spec'd, doc'd.
- Two-resource lifetime in place; `close/1` releases the handle's
  reference to the mapping; the mapping's destructor performs
  `munmap`.
- All existing iommap tests pass on Linux x86-64, Linux ARM64,
  macOS ARM64, FreeBSD, OpenBSD.
- New tests cover the seven cases above.
- A 64 MiB region_binary produces O(1) BEAM heap growth.
- README, moduledoc, CHANGELOG updated, including the truncation
  warning and lifetime explanation.
- Hex.pm version bump prepared but not published; the maintainer
  cuts the release after review.
