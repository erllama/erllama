# Cache design

This is the contributor's-eye view of why the cache looks the way
it does. The user-facing description of *what* the cache provides
lives in the [caching guide](../guides/caching.md). This document
is the *why*.

## Token-exact, not approximate

Cache keys are SHA-256 over `(model_fp || quant || ctx_params ||
tokens_le32)`. The full token list goes into the hash, encoded as
little-endian u32. Two calls with different tokenisations of the
"same" prompt produce different keys — and rightly so, because the
KV state is a function of the tokens, not the surface text.

Approximate or fuzzy matching was an early temptation. We rejected
it for two reasons:

1. **Correctness is not a tunable.** A "close enough" cache hit
   silently changes the model's output for the user. There is no
   useful way to surface "we used a similar but not identical
   cache row" to a downstream caller. Either the state is the right
   state or it isn't.
2. **Approximate match needs a candidate proposer.** The proposer
   has to know what semantic neighbours look like. That bakes in a
   policy decision (which embedding model? which distance metric?)
   that does not generalise across tenants. Out of scope for v1;
   tracked but not roadmapped.

The longest-prefix walk solves the practical case approximate
matching tries to address — "this prompt is yesterday's prompt plus
a new turn" — without weakening the guarantee. It walks the new
prompt's tokens backward by the configured stride and probes the
exact-key index at each alignment. The longest hit wins. Strictly
exact, by construction.

## Multi-tier, not just RAM

We ship three tiers because no single layer is the right answer
across deployments:

- **`ram`** (ETS slabs) — lowest latency, smallest budget. ETS reads
  are sub-µs hot-path-friendly; writes are funnelled through one
  owner process per table.
- **`ram_file`** (`/dev/shm`) — fast and effectively unlimited by
  process address space. Survives a model-supervisor restart but
  not a node restart.
- **`disk`** — survives everything. The cheap tier; deploy with the
  largest quota of the three.

Each tier is independently supervised, has its own byte budget, and
its own LRU. A save written to one tier never moves; the disk tier
is intentionally not a "promotion target" of the RAM tier — that
would force every saved row to be re-encoded twice.

Why is the disk tier first-class instead of a "fallback"? Because
big-model deployments cannot fit a working set of warm KV state in
RAM alongside the weights. A 70B-class model in Q4 takes ~40 GB
of RAM for weights alone; a 30 000-token KV state can easily exceed
1 GB. With ten warm sessions you've blown a 24 GB GPU. Disk is the
realistic place for that working set, and modern NVMe is fast
enough to keep restore cost in the millisecond range.

## Sole-writer arbitration

The meta server (`erllama_cache_meta_srv`) is the only process that
mutates the meta ETS, the LRU, and the reservation table. Every
write — claim, release, evict, save announce — goes through a
gen_server call. Reads stay on ETS directly via `ets:lookup/2`.

The split is deliberate: hot-path reads must not contend on a
gen_server message queue, but writes must serialise so we never
race two reservations for the same key. `ets:select_replace/2` was
considered for in-place atomic updates but rejected — the
reservation state machine is rich enough that single-row CAS would
not be enough, and we'd need a lock anyway.

## Save reasons taxonomy

Five reasons, each with distinct semantics:

| Reason | Sync? | Trigger | Why it's a separate reason |
|---|---|---|---|
| `cold` | async | After a cold prefill, at trimmed-prefix boundary | First save for this prefix; we want it on stable storage as soon as possible. |
| `continued` | async | Every `continued_interval` tokens during generation | Keeps the cache useful even if generation is interrupted. |
| `finish` | async | End of generation, captures prompt+reply | Multi-turn flows resume from this row. |
| `evict` | sync | Holder asked to release | Pressure-driven; must complete before the slab returns to the pool. |
| `shutdown` | sync | `prep_stop` or `unload/1` | Best-effort save before the model dies; capped by `evict_save_timeout_ms`. |

The async/sync distinction is load-bearing. `cold` and `continued`
must not block the request path; `evict` and `shutdown` must block
because the holder is going away.

The taxonomy is a direct port from
[antirez/ds4](https://github.com/antirez/ds4). We kept the names
because the semantics map cleanly and any future reader of either
codebase wins by recognising them.

## What the cache is not

- **Not a session manager.** It does not track conversations or
  authenticate callers. The session layer above passes a
  `parent_key` if it has one; the cache treats it as a hint, not a
  capability.
- **Not a request scheduler.** Concurrency, queueing, and rate
  limiting live above the cache. The cache only owns "the
  on-disk/in-RAM mapping from token-prefix to KV bytes".
- **Not a generic blob store.** Slab format is opinionated:
  fixed-size per-layer regions, ds4-derived 48-byte header, CRC32C
  trailer. Repurposing the format for non-llama.cpp data would be
  miserable.
- **Not GPU-aware.** The cache stores KV bytes; whether they go on
  GPU or CPU is a property of the `llama_context*` that consumes
  them. The cache doesn't care.

## Two remaining v1 deferrals

Both intentional:

1. **No semantic candidate proposer.** Discussed above. v1 is
   exact-only.
2. **No KV state compression.** TurboQuant is unproven at this
   layer. Generic lz4/zstd would help a little on some quant
   schemes and hurt others. The breakeven is unclear and we don't
   have benchmark data we trust enough to ship a default.

Both are tracked for v2; neither blocks production use of v1.
