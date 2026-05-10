# erllama

Native Erlang/OTP wrapper around `llama.cpp` with full supervision, a
tiered KV cache, and memory-pressure-aware eviction.

The cache subsystem is the centrepiece: an inference run can save its
KV state to disk and a later run with the same prompt resumes from
that state in milliseconds instead of re-running prefill. The cache
key is the `(model fingerprint, quant, ctx params, exact tokens)`
tuple — hits are token-exact by construction.

```erlang
{ok, _} = application:ensure_all_started(erllama).
{ok, M} = erllama:load_model(#{
    backend     => erllama_model_llama,
    model_path  => "/srv/models/tinyllama-1.1b-chat.gguf",
    %% ...cache + context fields, see "Configuration" below
}).
{ok, Reply, _Tokens} = erllama:complete(M, <<"Once upon a time">>).
```

Multi-turn chat reuses the previous turn's saved state via the
session-layer `parent_key` mechanism; the cache layer waits up to
`session_resume_wait_ms` for an in-flight `finish` save to publish
before falling through to cold prefill.

## Status

**Pre-release (`v0.1`).** The cache subsystem, scheduler, and the
NIF surface for llama.cpp are implemented and covered by 157 EUnit +
11 PropEr + 5 stub Common Test cases. An end-to-end CT suite gated on
`LLAMA_TEST_MODEL` runs against a real GGUF (5 cases, all passing
locally with TinyLlama 1.1B Q4_K_M).

Out of scope for `v1`:

- Semantic candidate proposer (vector-index-driven warmth) — deferred to v2.
- KV state compression (TurboQuant or generic lz4/zstd) — uncompressed in v1.
- Sampler / RNG state persistence for mid-stream resume — turn boundaries only.
- HTTP API (Cowboy + OpenAI-compat) — not yet wired.

## Requirements

- Erlang/OTP **28** (see `.tool-versions`)
- rebar3 **3.25+**
- C++17 toolchain (Apple clang or recent gcc; `cmake` >= 3.20)
- Apple Silicon: Metal + Accelerate are auto-detected.
- Linux: BLAS auto-detected; CUDA off by default (toggle via
  `ERLLAMA_OPTS=-DGGML_CUDA=ON`).

## Build

```bash
rebar3 compile          # builds llama.cpp + the NIF
rebar3 shell            # boots the umbrella
```

Vendored llama.cpp lives at `c_src/llama.cpp` (see
`UPDATE_LLAMA.md` for the bump procedure).

## Architecture

```
erllama_sup (one_for_one)
├── erllama_cache_sup
│   ├── erllama_cache_meta_srv   sole writer; meta + LRU + reservations
│   ├── erllama_cache_ram        RAM tier (ETS slab store)
│   ├── erllama_cache_disk_srv   disk tier (read/write or iommap)
│   └── erllama_cache_writer     writer pool with leak-proof semaphore
├── erllama_model_sup            simple_one_for_one for dynamic models
└── erllama_scheduler            memory-pressure poller (off by default)
```

Inside a request:

1. `erllama:complete/2` enters the per-model `gen_statem`.
2. `idle → prefilling`: tokenize, then either hit the cache and
   `kv_unpack` (warm) or run `llama_decode` over the prompt (cold).
   On cold, fire an async `cold` save at the trimmed-prefix boundary.
3. `generating`: token-by-token `llama_decode` with greedy sampling.
   Every `continued_interval` tokens the model fires an async
   `continued` save.
4. `generating → idle`: fire an async `finish` save for the full
   prompt + reply. The state is then evictable.

Save reasons are `cold | continued | finish | evict | shutdown`. The
publish protocol is `meta-reserve → temp write → datasync → link(2)`
(atomic create-if-not-exists), with stage-aware cleanup if the writer
crashes.

See `AGENTS.md` for the contributor guide. Behaviour invariants are
encoded in the EUnit, PropEr, and Common Test suites — the cache
publish protocol in particular has dedicated tests for stage-aware
cleanup, EEXIST adopt/replace, and TTL refresh.

## Configuration

Application env (see `config/sys.config` for the full default set):

```erlang
{erllama, [
  {tiers, [
    #{backend => ram,                              quota_mb => 4096},
    #{backend => {ram_file, "/dev/shm/erllama"},   quota_mb => 16384},
    #{backend => {disk, "/var/lib/erllama/kvc"},   quota_mb => 65536}
  ]},

  %% Save-policy gates
  {min_tokens,             512},
  {cold_min_tokens,        512},
  {cold_max_tokens,      30000},
  {continued_interval,    2048},
  {boundary_trim_tokens,    32},
  {boundary_align_tokens, 2048},

  %% Cache flow tunables
  {evict_save_timeout_ms, 30000},
  {session_resume_wait_ms,  500},
  {fingerprint_mode,       safe},   %% safe | gguf_chunked | fast_unsafe
  {disk_io,                auto},   %% auto | iommap | read_write

  %% Memory-pressure-driven eviction (off by default)
  {scheduler, #{
    enabled         => false,
    pressure_source => noop,        %% noop | system | nvidia_smi | {module, M}
    interval_ms     => 5000,
    high_watermark  => 0.85,
    low_watermark   => 0.75,
    min_evict_bytes => 1048576,
    evict_tiers     => [ram, ram_file]
  }}
]}.
```

Per-model config (passed to `erllama:load_model/1,2`):

```erlang
#{
  backend         => erllama_model_llama,    %% or erllama_model_stub
  model_path      => "/srv/models/x.gguf",
  model_opts      => #{n_gpu_layers => 99},
  context_opts    => #{n_ctx => 4096, n_batch => 512},
  tier_srv        => default_disk,
  tier            => disk,                   %% disk | ram_file
  fingerprint     => <<32 bytes>>,
  fingerprint_mode=> safe,
  quant_type      => q4_k_m,
  quant_bits      => 4,
  ctx_params_hash => <<32 bytes>>,
  context_size    => 4096,
  policy => #{ ... per-model overrides of the gates above ... }
}
```

## API

### `erllama` — public façade

| Function | Purpose |
|---|---|
| `load_model/1`, `/2` | Start a supervised model under `erllama_model_sup`. The 1-arg form auto-generates an id. Returns `{ok, ModelId}`. Idempotent against `{already_started, _}` (returns `{error, already_loaded}`). |
| `unload/1` | Terminate a loaded model. Pid or registered name. |
| `complete/2`, `/3` | Run a completion. Opts: `response_tokens`, `seed`, `parent_key`. Returns `{ok, ReplyBin, [TokenId]}`. |
| `models/0` | List currently-loaded model pids. |
| `counters/0` | Snapshot of cache counters as a map (alias for `erllama_cache:get_counters/0`). |

### `erllama_cache` — cache façade

| Function | Purpose |
|---|---|
| `get_counters/0` | Snapshot map. See "Counters" below. |
| `reset_counters/0` | Set every slot to 0 (tests / fresh-boot semantics). |
| `gc/0` | Synchronous full eviction pass. Returns `{evicted, NumRows}`. |
| `evict_bytes/1` | Evict the oldest-LRU rows until at least `Target` bytes are freed. Returns `{evicted, NumRows, BytesFreed}`. |
| `evict_bytes/2` | Same with a tier filter (`all` or `[ram \| ram_file \| disk]`). The scheduler uses this. |

**Counters** returned by `get_counters/0`:

| Key | Meaning |
|---|---|
| `hits_exact` | Lookups that hit a published row by exact key. |
| `hits_resume` | Lookups that took the `parent_key` resume path (strict-prefix verified). |
| `hits_longest_prefix` | Lookups that found a cached prefix via the stride walk (no `parent_key` supplied). |
| `misses` | Lookups that fell through to cold prefill. |
| `saves_cold` | Async cold saves at the trimmed prefix. |
| `saves_continued` | Async continued saves every `continued_interval` tokens. |
| `saves_finish` | Async finish saves at end of generation. |
| `saves_evict` | Sync evict saves driven by external pressure. |
| `saves_shutdown` | Sync shutdown saves on `prep_stop`. |
| `evictions` | Rows the meta server has actually dropped. |
| `corrupt_files` | Files rejected by parse + deleted by the load path. |
| `duplicate_dropped` | Save reservations short-circuited by `already_present`. |
| `pack_total_ns` | Cumulative monotonic-time spent in `kv_pack` calls (save path). Divide by saves to get average pack cost. |
| `load_total_ns` | Cumulative monotonic-time spent in `pin_and_load` (warm-read path: checkout + tier load + kv_unpack + checkin). Divide by exact+resume+longest-prefix hits to get average load cost. |
| `longest_prefix_probes` | Sum of probes across all `lookup_longest_prefix` calls. Divide by `hits_longest_prefix + misses` (since the last call) to get average walk depth. |
| `longest_prefix_ns` | Cumulative time spent in the longest-prefix walk. With `longest_prefix_probes` gives ns/probe. |
| `bytes_ram`, `bytes_ramfile`, `bytes_disk` | Reserved for tier byte accounting. |

**Metrics are always on.** Per-counter cost is `persistent_term:get/2`
(single pointer deref) + `atomics:add/3` (lock-free CAS) ≈ 10–20 ns;
`erlang:monotonic_time(nanosecond)` adds another 10–50 ns. Net
overhead per `pin_and_load` or `lookup_longest_prefix` call is on
the order of 70–200 ns, against operations that cost 100–1000× more
(meta_srv hop, tier load, kv_unpack, SHA-256 over multi-KB
prefixes). Counter slots are `u64`; `longest_prefix_ns` would need
>580 years at 1 ns/probe to wrap. Don't disable metrics without
profiler evidence that they're a hot spot — they essentially can't
be at these ratios.

### `erllama_scheduler` — memory-pressure-driven eviction

A `gen_server` under `erllama_sup`. Polls a pluggable pressure source
on `interval_ms` and asks the cache to evict slabs (default: only
`ram` and `ram_file` tiers) when used/total crosses `high_watermark`,
targeting a drop to `low_watermark`.

| Function | Purpose |
|---|---|
| `enable/1` | Toggle the periodic poller. |
| `set_pressure_source/1` | Switch the active source at runtime. |
| `set_thresholds/2` | `(High, Low)` with `0.0 <= Low < High <= 1.0`. |
| `sample/0` | Read the source once without acting. Returns `{Used, Total}`. |
| `force_check/0` | Sample now and apply the policy. Returns `{evicted, _, _}` or `{skipped, below_watermark \| disabled \| nothing_to_evict}`. |
| `status/0` | Snapshot map: enabled, source, thresholds, last reading, last eviction. |

### `erllama_pressure` — pressure-source behaviour

Stateless module. `sample(Source) -> {Used, Total}` dispatches to the
appropriate sampler. Built-in sources:

- `noop` — always `{0, 1}`.
- `system` — portable system memory via OTP's `memsup`
  (started on demand). Linux, macOS, BSD, Windows.
- `nvidia_smi` — sums VRAM across all visible NVIDIA GPUs by
  exec'ing `nvidia-smi`.
- `{module, M}` — calls `M:sample/0`. Implement
  `-behaviour(erllama_pressure)` to opt in.

### `erllama_nif` — low-level surface (mostly internal)

Used by the `erllama_model_llama` backend; exposed for advanced
callers and tests. Resources (`model_ref`, `context_ref`) carry a
per-resource mutex so explicit `free_model/1` / `free_context/1`
cannot race with concurrent dirty NIF calls.

| Function | Purpose |
|---|---|
| `crc32c/1` | CRC32C (Castagnoli) of an iodata. |
| `fsync_dir/1` | `fsync(2)` a directory file descriptor. Rejects paths with embedded NUL. |
| `load_model/2`, `free_model/1` | Open/close a GGUF. `free_model` returns `{ok, deferred}` if contexts still reference the model; the last context's destructor performs the actual free. |
| `new_context/2`, `free_context/1` | Construct/release a `llama_context*`. Holds a keep-resource on the model. |
| `tokenize/3`, `detokenize/2` | Vocab-validated; out-of-range token IDs are rejected before reaching llama internals. |
| `prefill/2`, `decode_one/1` | Run `llama_decode` over a token list / advance one greedy step. `decode_one` refuses with `{error, no_logits}` if no decode has produced sample-able state, instead of triggering `GGML_ASSERT`. |
| `kv_pack/3`, `kv_unpack/3` | Save/restore the seq-0 KV state to/from a binary. |
| `kv_seq_rm/4` | Drop cells `[P0, P1)` from a sequence. Used by the model layer after `kv_unpack` to drop the last cell + re-prefill it (regenerating the per-context logits buffer that the state save does not persist). |

All NIF calls are wrapped in `extern "C"` `noexcept` C++ shims
(`erllama_safe.cpp`) that catch any thrown exception inside llama and
return a sentinel; the C NIF maps each to a real Erlang error tuple
(`{error, oom}`, `{error, exception}`, `{error, invalid_token}`, …)
rather than letting an exception unwind across the C/BEAM boundary.

## Testing

| Command | What it runs |
|---|---|
| `rebar3 fmt` | Auto-format with erlfmt. Run first. |
| `rebar3 eunit` | 157 unit tests. |
| `rebar3 proper` | 11 property tests. |
| `rebar3 ct` | 5 stub-backend end-to-end Common Test cases. |
| `rebar3 lint` | Elvis. |
| `rebar3 dialyzer` | Static analysis. |
| `rebar3 xref` | Cross-reference. |

End-to-end against a real model:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.gguf \
  rebar3 ct --suite=test/erllama_real_model_SUITE
```

Without the env var the suite skips, so the default `rebar3 ct` stays
green on machines without a GGUF on hand.

## Safety

- Per-resource `pthread_mutex` makes concurrent `free_*/1` and dirty
  NIF ops safe (no UAF).
- `free_model/1` with active contexts is deferred to the last context
  destruction; the model is never freed while a context's internal
  pointer references it.
- `erllama_safe.cpp` catches any C++ exception in llama internals and
  surfaces a clean Erlang error.
- `decode_one/1` refuses without `decode_ready` so a misuse cannot
  trigger `GGML_ASSERT(logits != nullptr)` and abort the BEAM.
- Cache directory is `flock(LOCK_EX)`-protected when `disk_io =
  iommap`. External truncation of cached files would otherwise SIGBUS
  the BEAM during sub-binary or message access.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Benoit Chesneau.

The vendored `c_src/llama.cpp/` retains its upstream MIT license; see
`c_src/llama.cpp/LICENSE`.
