# NIF safety

The single NIF (`erllama_nif`) is the only place erllama touches
unmanaged memory. This document captures the patterns that keep the
BEAM alive in the face of llama.cpp's quirks.

## One NIF, dirty schedulers only

There is exactly one `.so` (`priv/erllama_nif.so`) housing every
native call: cache pack/unpack, model load, context construction,
tokenisation, decode, kv-pack, kv-unpack. Every export is marked
`ERL_NIF_DIRTY_JOB_CPU_BOUND` or `ERL_NIF_DIRTY_JOB_IO_BOUND`. None
run on the regular scheduler.

Why one NIF? Because the resource lifetime story is easier when
every type that participates lives in the same shared library.
Cross-NIF resource handling works in principle but the failure
modes (mismatched destructor pointers when one side reloads) are
not worth the modularity.

Why dirty schedulers? Because llama_decode of a 4096-token batch
takes hundreds of milliseconds on CPU; calling that on a regular
scheduler would starve every other process on the system.

## Two-resource lifetime

`erllama_nif` has two resource types whose interplay matters:

```
  model_ref       owns: weights mmap, vocab, llama_model*
  context_ref     owns: llama_context*; holds keep-resource on its model_ref
```

`enif_keep_resource` keeps the model alive as long as any context
references it. `free_model/1` from Erlang flips a flag on the
model_ref and returns `{ok, deferred}` if any context still holds
a keep-reference; the actual `llama_free_model` runs in the **last
context's** destructor when its keep-reference is finally released.

This is the only way to make `free_model` safe under concurrent
free + context destruction, given that llama.cpp's
`llama_free_context` reads from the model.

## Per-resource mutex

Each resource carries a `pthread_mutex` that every NIF function
acquires before touching the underlying llama struct. This serves
two ends:

1. **Race-free explicit free.** `erllama_nif:free_context/1`
   acquires the mutex, sets a freed flag, and calls
   `llama_free_context` from a deferred destructor. Subsequent NIF
   calls see the freed flag and return `{error, freed}` cleanly,
   even if they were already in flight on a different scheduler.
2. **Single-context concurrency.** llama.cpp does not document
   re-entrancy guarantees on `llama_decode`. The mutex pessimises:
   one decode at a time per context. This matches the gen_statem's
   single-token-at-a-time loop and costs nothing in practice.

We do **not** use `enif_resource_refcount` for any decision. The
function exists but is not part of the public API; relying on it
for control flow has bitten projects in the past when OTP changes
its internals. Two-resource keep-references give the same effect
through documented APIs.

## Exception safety: the `extern "C" noexcept` shim

llama.cpp is C++. It throws `std::bad_alloc` on OOM and various
domain-specific exceptions on bad input. An exception unwinding
across the C-NIF boundary into BEAM's stack is undefined behaviour
and crashes the VM in ugly ways.

Every llama call is wrapped in a thin shim
(`c_src/erllama_safe.cpp`):

```cpp
extern "C" int erllama_safe_decode(llama_context* ctx,
                                   const llama_batch* batch) noexcept {
    try {
        return llama_decode(ctx, *batch);
    } catch (const std::bad_alloc&) {
        return ERLLAMA_E_OOM;
    } catch (const std::exception&) {
        return ERLLAMA_E_EXCEPTION;
    } catch (...) {
        return ERLLAMA_E_EXCEPTION;
    }
}
```

The C NIF maps each sentinel to a real Erlang error tuple
(`{error, oom}`, `{error, exception}`, …). The BEAM never sees a
C++ exception cross the boundary.

## `decode_one` defensive check

`llama_decode_one` calls `llama_get_logits_ith(-1)`, which inside
llama.cpp asserts `logits != nullptr`. If a caller calls
`decode_one/1` before any `prefill/2`, that assert fires
`GGML_ASSERT(...)` and aborts the process — the BEAM dies.

The Erlang-side guard is `decode_ready` flag on the context_ref:
set to true on every successful prefill or decode. `decode_one/1`
returns `{error, no_logits}` if the flag is false instead of
calling into llama. The C side double-checks; both layers must
fail closed before we hand a context with no decode history to
`llama_decode`.

## Disk reads: plain read I/O, not mmap

The disk tier reads cache files via `file:read_file/1` into a fresh
BEAM heap binary. mmap was an option in earlier revisions; we
removed it because:

- The process already mmaps the GGUF (multi-GB on a 70B-class
  model); a second mmap per cache restore doubles the VM footprint.
- A region binary returned to the BEAM survives the closing NIF
  call. Any external truncation of the cache file then crashes the
  VM with SIGBUS in places no NIF can intercept (sub-binary
  creation, message send, GC).
- ds4 — the design erllama draws from — makes the same choice for
  the same reasons.

For typical cache files (1–100 MB) the kernel page cache already
makes the second `read(2)` cheap; mmap's zero-copy benefit only
shows up at sizes where the trade-off ceases to be acceptable.

## `fsync_dir` and link durability

`erllama_nif:fsync_dir/1` opens a directory fd and `fsync(2)`s it.
The publish protocol calls this after `link(2)` to ensure the
directory entry is on stable storage; without it, a crash between
`link` and the next inode flush could lose the linked file.

Path validation rejects embedded NUL bytes — a common footgun on
paths constructed from user input.

## Test surface

| Concern | Test |
|---|---|
| Concurrent free + decode | `erllama_nif_safety_tests:free_during_decode/0` |
| `decode_one` without prefill | `erllama_nif_safety_tests:decode_one_no_logits/0` |
| llama exception caught at boundary | `erllama_nif_safety_tests:bad_input_returns_error/0` |
| Two-resource model survives last context drop | `erllama_nif_lifetime_tests:deferred_free/0` |
| `fsync_dir` rejects NUL paths | `erllama_nif_tests:fsync_dir_rejects_nul/0` |

If any of these flip on a change, the change is wrong. The tests
exist precisely because the failure modes they cover crash the
BEAM, not "return an error" — there is no graceful fallback.
