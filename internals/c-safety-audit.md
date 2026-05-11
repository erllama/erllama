# C / NIF safety audit

Audit branch: `audit/c-safety-review`. Scope: `c_src/erllama_nif.c`,
`c_src/erllama_safe.cpp`, `c_src/crc32c.{c,h}`. The upstream
`c_src/llama.cpp/` submodule was not examined.

The audit looked for segfaults, deadlocks, infinite loops, leaks, and
non-thread-safe operations. Findings are grouped by severity. Line
numbers are against the tree at the time of audit.

## High severity

### H1. Adapter outlives its underlying `llama_model` — use-after-free on adapter free

- Sites: `nif_free_model` (`c_src/erllama_nif.c:602-633`),
  `nif_adapter_load` (`c_src/erllama_nif.c:2051-2095`),
  `nif_adapter_free` (`c_src/erllama_nif.c:2097-2113`),
  `adapter_dtor` (`c_src/erllama_nif.c:351-366`).
- Failure mode: `llama.h` documents that "loaded adapters that are not
  manually freed will be freed when the associated model is deleted".
  `nif_free_model` ignores adapters: when `active_contexts == 0` it
  frees the model immediately, which implicitly frees every adapter
  derived from it. The wrapping `erllama_adapter_t` keeps the *resource*
  alive (via `enif_keep_resource`) but `a->adapter` is now a dangling
  pointer. A subsequent `nif_adapter_free` or `adapter_dtor` calls
  `llama_adapter_lora_free` on freed memory — UAF / double-free.
- Fix sketch: track `active_adapters` on `erllama_model_t` next to
  `active_contexts`. `nif_free_model` should set `release_pending`
  whenever either counter is non-zero. Decrement `active_adapters`
  from the adapter destructor (mirror of `context_drops_model`) and
  perform the deferred free there.

### H2. Adapter pointer race in `nif_set_adapters`

- Site: `c_src/erllama_nif.c:2144-2191`.
- Failure mode: for each adapter the code locks `a->mu`, copies
  `a->adapter` into the array, then unlocks immediately. The pointer
  array is then passed to `erllama_safe_set_adapters_lora` outside any
  adapter lock. A concurrent `nif_adapter_free` on any of the listed
  adapters will null the underlying llama object between the unlock
  and the llama call — the call sees a dangling pointer.
- Fix sketch: either (a) hold every `a->mu` for the duration of
  `set_adapters_lora` (lock them in deterministic order — e.g. by
  resource pointer — to avoid AB-BA), or (b) add an `in_use` counter
  to `erllama_adapter_t` that `nif_adapter_free` waits on, or (c)
  serialise adapter mutation through a coarser lock (one per model
  is enough since adapters are always model-scoped).

## Medium severity

### M1. Partial chat-message build leaks role/content pairs

- Site: `build_chat_msgs_from_list` (`c_src/erllama_nif.c:1377-1405`),
  caller `nif_apply_chat_template` (`c_src/erllama_nif.c:1533-1541`).
- Failure mode: when the loop fails on message k > 0 (bad map,
  missing role/content, or OOM allocating the content string), the
  helper returns `-1` / `-2` without freeing the role+content pairs
  it already wrote into `out[idx0..idx-1]`. The caller invokes
  `free_chat_msgs(msgs, n_msgs)` with the pre-call `n_msgs`
  (0 or 1 for the optional synthetic-system entry), so every
  successfully-built message before the failure leaks. Per-message
  leak is `enif_alloc(role_bin.size + 1) + enif_alloc(content_bin.size + 1)`.
- Fix sketch: have the helper free `out[idx0..idx-1]` before
  returning, or accept an `int *out_idx` so the caller can free up
  to `idx` on error.

## Low severity / contention notes

### L1. `nif_detokenize` holds `m->mu` across the whole token loop

- Site: `c_src/erllama_nif.c:1165-1263`.
- Not a deadlock, but the model lock is held across N
  `llama_token_to_piece` calls plus per-call `enif_realloc`. For a
  multi-MB detokenize the entire model surface (tokenize,
  apply_chat_template, free_model contention) stalls for the
  duration. Consider releasing the lock per chunk or splitting
  detokenize into a per-token primitive driven from Erlang.

### L2. `nif_kv_pack` allocates a fresh empty binary on `need == 0`

- Site: `c_src/erllama_nif.c:873-880`. Minor. Returning a shared
  empty term would be cheaper, but `enif_alloc_binary(0, ...)` is
  legal per ERTS docs and the path is rare.

## Considered and ruled out

### Per-resource mutex pattern

`erllama_model_t`, `erllama_context_t`, `erllama_adapter_t`, and
`erllama_sampler_t` each carry a `pthread_mutex_t` that every NIF
entry takes before dereferencing the wrapped llama pointer. This
serialises concurrent dirty-NIF calls with explicit `free_*` calls
on the same resource: a free cannot interleave with a live llama
call. Different resources stay independent. Intentional and
correct.

### `context_drops_model` deferred-free

The combination of `m->active_contexts` and `m->release_pending`
under `m->mu` is the only state involved. The single decrement
that observes `release_pending && active_contexts == 0 && m->model`
performs the free atomically, all under the lock. Race-free for
contexts. (Adapters are *not* covered — see H1.)

### `enif_keep_resource` / `enif_release_resource` pairing

- Context wrapper keeps the model resource at `nif_new_context:700`
  and releases it at `ctx_dtor:318` and `nif_free_context:735`. One
  keep, one release per lifecycle path.
- Adapter wrapper keeps the model at `nif_adapter_load:2090`,
  releases at `adapter_dtor:359`. Single pair.
- Sampler wrapper keeps the context at `nif_sampler_new:2249`,
  releases at `sampler_dtor:338`. Single pair.

### `pthread_mutex_init` failure paths

All four resource constructors (`nif_load_model`, `nif_new_context`,
`nif_adapter_load`, `nif_sampler_new`) zero-init the resource before
attempting `pthread_mutex_init`. On init failure they
`enif_release_resource` (which will run the destructor with
`mu_inited == 0`) and free the underlying llama object explicitly.
The destructors all guard `pthread_mutex_destroy` on `mu_inited`.
Clean.

### `enif_alloc(sizeof(T) * n)` overflow

`n` is always a validated `int32_t` capped at `ERLLAMA_MAX_TOKENS`
(`1 << 20`) or `(1 << 24)` in detokenize, or it comes from
`enif_get_list_length` (unsigned int) and ends up multiplied by
small `sizeof`s. On 64-bit platforms `size_t` is 8 bytes; no
overflow is reachable. 32-bit BEAM would be exposed in the
`nif_set_adapters` path but that target is not supported.

### `llama_context` thread-safety

`llama.cpp` contexts are not thread-safe. Every entry that touches a
context (`erllama_safe_decode`, `erllama_safe_state_seq_*`,
`erllama_safe_sampler_sample`, `erllama_safe_get_embeddings*`,
`erllama_safe_set_adapters_lora`, `erllama_safe_set_embeddings`) is
gated by the matching `c->mu`. Same for `m->mu` on model calls.
Single-threaded per resource is the design.

### `pthread_once` use in `crc32c_init` and `erllama_safe_backend_init_once`

Both correctly serialise one-time initialisation; the static `rc`
read by `erllama_safe_backend_init_once` after `pthread_once`
returns is covered by the happens-before guarantees of pthread_once.

### Infinite loops

None found. Every loop is bounded by validated input size
(`ERLLAMA_MAX_TOKENS`, `(1 << 24)`, `ERLLAMA_MAX_TOKEN_TEXT`,
`grammar_bin.size`, `enif_get_list_length` capped), by external
list iteration with `enif_get_list_cell` returning false on
exhaustion, or by `pthread_once`.

## Suggested fix priority

1. **H1**: track adapters on the model, defer free while adapters
   exist. Prevents the most likely VM crash in real use (anyone
   calling `unload/1` while still holding a LoRA reference).
2. **H2**: hold adapter locks (or use an `in_use` flag) across
   `set_adapters_lora`.
3. **M1**: free partial work in `build_chat_msgs_from_list` on
   error to plug the per-bad-request leak.
4. **L1** is a contention/SLA concern, not a correctness bug;
   address only if `detokenize` shows up in scheduler stall traces.
