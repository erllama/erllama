# Caching guide

erllama's KV cache turns a multi-second prefill into a millisecond
restore. This guide is the operator's-eye view: what it does, when
it kicks in, and which knobs to touch.

## The mental model

A transformer's "KV state" is the per-layer key/value tensors
produced while reading the prompt. Once you have them, generating
the next token costs one forward pass. Without them, you have to
re-read every token of the prompt from scratch.

erllama's cache stores those tensors keyed on the **exact tokens
that produced them**:

```
key = sha256(model_fingerprint || quant || ctx_params || tokens_le32)
```

Same tokens → same key → guaranteed-correct restore. There is no
fuzzy matching layer; "close enough" is not allowed at this level.

## Three tiers

```
ram       ETS slabs in BEAM heap. Lowest latency, smallest budget.
ram_file  Files on /dev/shm. Fast, capped only by tmpfs size.
disk      Files on persistent storage. Survives restarts.
```

Each tier is an independently-supervised gen_server with its own
byte quota and its own LRU. A save is written to the tier you
configure on the model; reads consult an in-memory index that fans
out to the right tier.

The disk tier is **a first-class citizen**: large models that
wouldn't fit alongside a working set of warm KV state in RAM can
let the disk tier hold most of the cache, and warm-restore in
milliseconds when a hit comes in. This is the same idea ds4 uses
for DeepSeek V4, generalised here to any GGUF llama.cpp can load.

## When does a save happen?

The per-model `gen_statem` fires saves at five well-defined moments,
each with its own `save_reason`:

| Reason | When | Sync? |
|---|---|---|
| `cold` | Right after a cold prefill, at the trimmed-prefix boundary. Async — the writer pool does the work. |
| `continued` | Every `continued_interval` tokens during generation. Async. |
| `finish` | At the end of a completion, capturing prompt + reply. Async. |
| `evict` | When a holder is asked to release its slab. Sync (pause decode, pack, release). |
| `shutdown` | On `prep_stop` or `unload/1`. Sync, capped by `evict_save_timeout_ms`. |

Async saves go through `erllama_cache_writer` (a poolboy pool of
dirty-IO workers). Sync saves block the calling process until the
file is on stable storage.

## When does a hit happen?

Three lookup paths, in order of preference:

1. **Exact key.** Caller passes the exact `parent_key` from the
   previous turn. Cheapest. Used by Erlang-native multi-turn flows.
2. **Resume.** Caller passes a `parent_key` from an earlier turn,
   and the new prompt strictly extends the cached prefix.
3. **Longest-prefix walk.** No `parent_key` supplied. The cache
   walks the new prompt's tokens backward by the configured stride
   (`boundary_align_tokens`) and probes the index for each
   alignment. The longest cached prefix wins.

For stateless callers — OpenAI/Anthropic-shaped HTTP APIs that
resend the full conversation each turn — option 3 is what you want.
You don't have to do anything; just call `erllama:complete/2`.

## Save policy gates

Saving every prefix would flood the writer pool. erllama gates saves
behind a few thresholds, all overridable per-model.

| Gate | Default | What it does |
|---|---|---|
| `min_tokens` | 512 | Skip saves shorter than this. Prefills under 512 tokens are usually cheaper than the round-trip to disk. |
| `cold_min_tokens` | 512 | Don't fire a `cold` save for shorter prefills. |
| `cold_max_tokens` | 30 000 | Cap on cold-save size. Protects against pathological prompts. |
| `continued_interval` | 2048 | Fire a `continued` save every N generated tokens. |
| `boundary_trim_tokens` | 32 | Drop the last N tokens before saving. Mid-token, mid-sentence boundaries make poor resume points; trim to a safe alignment. |
| `boundary_align_tokens` | 2048 | Round trim down to a multiple of this. Sets the longest-prefix walk's stride. |
| `session_resume_wait_ms` | 500 | When a `parent_key` is supplied and the cache sees an in-flight finish save, wait up to this long for it to publish before falling through to a fresh prefill. |

Bigger `boundary_align_tokens` = fewer probes per longest-prefix
walk but coarser hit alignment. 2048 is the default; 256 makes
hits more likely on shorter prompts at the cost of more probes.

## Memory-pressure-driven eviction

`erllama_scheduler` is a polling gen_server that watches a pluggable
pressure source and evicts cache slabs when pressure crosses a
watermark. Off by default. Enable in `sys.config`:

```erlang
{erllama, [
  {scheduler, #{
    enabled         => true,
    pressure_source => system,        %% portable, memsup-backed
    interval_ms     => 5000,
    high_watermark  => 0.85,
    low_watermark   => 0.75,
    evict_tiers     => [ram, ram_file] %% disk fills to its own quota
  }}
]}.
```

Sources shipped:

- `noop` — always reports zero pressure.
- `system` — OTP `memsup`. Linux, macOS, BSD, Windows.
- `nvidia_smi` — sums VRAM across all visible NVIDIA GPUs.
- `{module, M}` — calls `M:sample/0`. Implement
  `-behaviour(erllama_pressure)` to write your own.

When the source reports `Used / Total >= high_watermark`, the
scheduler asks the cache to evict enough bytes to drop the ratio
below `low_watermark`, scoped to the configured tiers.

## Inspecting the cache

```erlang
%% Hit/miss/save counters and per-path latency totals.
erllama_cache:get_counters().

%% Every row in the index, raw tuples:
%%   {Key, Tier, Size, LastUsedNs, Refcount, Status, HeaderBin,
%%    Location, TokensRef, Hits}
erllama_cache_meta_srv:dump().

%% Synchronous full eviction pass: returns {evicted, N}.
erllama_cache:gc().

%% Free at least N bytes, oldest LRU first: returns {evicted, N, BytesFreed}.
erllama_cache:evict_bytes(256 * 1024 * 1024).
erllama_cache:evict_bytes(256 * 1024 * 1024, [ram, ram_file]).
```

The counter map is documented inline on
`erllama_cache:get_counters/0` — call it from a shell to see the
keys for your build.

## Disabling the cache

For benchmarking or sanity checks: load the model with `tier => ram`
and a tiny `min_tokens` to bypass saves entirely, or set the
application env to disable all saves at the policy level:

```erlang
{erllama, [
  {min_tokens, infinity}       %% nothing ever clears the gate
]}.
```

There is no global "off switch" — disabling was an explicit
non-goal. The cache is the product.

## See also

- [Loading a model](loading.md) — option-by-option walkthrough.
- [Configuration reference](configuration.md) — every knob,
  with defaults.
- Internals: [cache design](../internals/cache-design.md) and
  [publish protocol](../internals/publish-protocol.md) for the
  reasons behind the choices.
