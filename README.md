# erllama

[![CI](https://github.com/erllama/erllama/actions/workflows/ci.yml/badge.svg)](https://github.com/erllama/erllama/actions/workflows/ci.yml)
[![Hex.pm](https://img.shields.io/hexpm/v/erllama.svg)](https://hex.pm/packages/erllama)

erllama is a native Erlang/OTP wrapper around `llama.cpp` with a
**token-exact, multi-tier, supervised KV cache**. It turns a
multi-second prefill into a millisecond restore, and lets you keep
**more warm state than fits in RAM** by promoting cold-but-popular
prefixes down to the disk tier.

If you have ever waited five seconds for a chat assistant to
acknowledge "hello" — that's prompt prefill. erllama caches the
work so the second turn, the third turn, and every subsequent agent
sharing the same system prompt skip it.

## A 30-second taste

```erlang
1> {ok, _} = application:ensure_all_started(erllama).
2> Path = "/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf".
3> {ok, Bin} = file:read_file(Path).
4> {ok, M} = erllama:load_model(#{
       backend     => erllama_model_llama,
       model_path  => Path,
       fingerprint => crypto:hash(sha256, Bin)
   }).
{ok, <<"erllama_model_2375">>}

5> {ok, #{reply := Reply, finish_key := FK}} =
       erllama:complete(M, <<"Once upon a time">>).
%% ~3 s on a CPU box. Prompt prefill, async cold save fired.

6> {ok, #{reply := Reply2}} = erllama:complete(M, <<"Once upon a time">>).
%% ~10 ms. Cache hit; KV state restored, one decode for fresh logits.

7> {ok, _} = erllama:complete(M, <<"Once upon a time, in a quiet village">>).
%% ~50 ms. Longest-prefix walk found the cached row even though
%% the new prompt is longer.

8> {ok, _} = erllama:complete(M, <<"and they lived happily ever after">>,
                               #{parent_key => FK}).
%% Exact session resume from the saved row in step 5.
```

`load_model/1` returns a binary `model_id` that is also the registered
name for the underlying gen_statem. Pass it to `complete/2,3`,
`unload/1`, etc.

That is the whole pitch. The cache is on by default, runs under
its own supervisor, and never returns approximate matches.

## What you get

- **Many models in one BEAM.** Load TinyLlama and Llama-3-8B side by
  side, hot-swap a model without bouncing the cache, give each model
  its own `policy` and `tier`. One shared cache; rows are
  fingerprint-segregated so models never collide on identical
  prompts.
- **Token-exact hits.** Cache key is
  `sha256(model_fp || quant || ctx_params || tokens_le32)`. Same
  tokens, same key, guaranteed-correct restore.
- **Three storage tiers.** ETS slabs for the hottest data, files
  on `/dev/shm` for warm working set, on-disk files (plain read
  I/O) for everything else. Each tier supervised independently
  with its own quota and LRU.
- **Bigger than RAM.** Disk is a first-class tier, not a fallback.
  A 70B model in Q4 already takes ~40 GB of weights; the disk tier
  holds the warm KV state your working set needs without crowding
  weights out of RAM.
- **Shared-prefix hits across agents.** Spawn N workers that all
  start with the same system prompt: the first cold-prefills, every
  subsequent worker gets a longest-prefix hit on the shared part.
- **Multi-turn warmth.** Pass the previous turn's `parent_key` and
  the cache waits up to `session_resume_wait_ms` for the in-flight
  finish save to publish.
- **Stateless-friendly.** OpenAI/Anthropic-shaped servers that
  resend the full conversation each turn get hits automatically
  through a longest-prefix walk. No `parent_key` needed.
- **Crash-safe saves.** Reserve, write temp, validate, atomic
  `link(2)`, announce. Two-stage TTL cleanup adopts orphans on
  writer crash.
- **Memory-pressure-driven eviction.** Pluggable pressure source
  (`memsup`, `nvidia-smi`, or your own callback). Off by default.
- **Always-on metrics.** Hits, misses, saves, evictions, and
  per-path latency totals exposed via `erllama_cache:get_counters/0`.
  Per-counter cost is ~10-20 ns; you cannot meaningfully turn them
  off.
- **Multi-sequence batched scheduler.** Opt in with
  `context_opts.n_seq_max > 1` and up to N requests prefill and
  decode concurrently through a single `llama_decode` per tick.
  Default `n_seq_max => 1` keeps single-tenant behaviour
  bit-identical to 0.1.
- **Chunked prefill.** A long prompt is sliced into per-tick
  chunks (`prefill_chunk_size`, default `max(64, n_batch div 4)`)
  so it never monopolises the batch and concurrent decoders keep
  making progress.
- **Lock-free observability.** `erllama:phase/1`,
  `pending_len/1`, `last_cache_hit/1`, and `queue_depth/1` read a
  public ETS row without crossing the model gen_statem, so a
  router can probe "is this model busy?" without serialising
  behind the work it is probing.

## Installation

erllama targets Erlang/OTP **28** with rebar3 **3.25+**.

Add to `rebar.config`:

```erlang
{deps, [
    {erllama, "~> 0.4"}
]}.
```

Then in your supervision tree, wait for the application to start
before loading models:

```erlang
ok = application:ensure_started(erllama).
```

The first compile builds vendored llama.cpp (~3 minutes on a fast
machine). Subsequent builds are cached. See [requirements](#requirements)
for the toolchain.

## Documentation

| Guide | What it covers |
|---|---|
| [Loading a model](guides/loading.md) | Every option to `erllama:load_model/1,2`, with examples and pitfalls. |
| [Caching](guides/caching.md) | Tiers, save reasons, lookup paths, watermarks. The operator's manual. |
| [Configuration](guides/configuration.md) | Full `sys.config` and per-model option reference. |
| [Building](guides/building.md) | Platform-specific build notes (Linux, macOS, FreeBSD), CUDA/Metal toggles, common build issues. |
| [Examples](guides/examples.md) | Drop-in patterns for one-shot completion, stateless HTTP servers, multi-turn sessions, concurrent agents, cache inspection. |

For the API reference (`erllama`, `erllama_cache`, `erllama_scheduler`,
`erllama_nif`), see the **[generated module docs on
HexDocs](https://hexdocs.pm/erllama)** or run `rebar3 ex_doc`
locally.

For the design rationale behind the cache:

- [Cache design](internals/cache-design.md) — why multi-tier, why
  token-exact, what was deliberately left out.
- [Publish protocol](internals/publish-protocol.md) — the
  five-stage crash-safe save protocol.
- [NIF safety](internals/nif-safety.md) — two-resource lifetime,
  exception shim, why disk reads use plain `file:read_file/1`.

## Many models in one BEAM

Each loaded model is its own supervised `gen_statem` under
`erllama_model_sup`. The cache is process-wide and segregates rows
by fingerprint, so the only thing two models share is the byte
budget.

```erlang
{ok, _} = erllama:load_model(<<"tiny">>, TinyConfig).
{ok, _} = erllama:load_model(<<"big">>,  BigConfig).

{ok, #{reply := R1}} = erllama:complete(<<"tiny">>, <<"summarise: ...">>).
{ok, #{reply := R2}} = erllama:complete(<<"big">>,  <<"deep analysis of: ...">>).

ok = erllama:unload(<<"tiny">>).
```

| Capability | How |
|---|---|
| N models in one BEAM | `load_model/2` per binary id; each is one `gen_statem` |
| No cross-model collisions | Cache key includes the model fingerprint |
| Hot-swap a model | `unload/1` then `load_model/2`; the cache survives |
| Per-model `policy` | `policy => #{...}` on the load; merges over app-env defaults |
| Per-model `tier` | `tier_srv => MyDisk, tier => disk` per model |
| Shared-prefix hits across agents | Longest-prefix walk on every cold prompt |
| Concurrent saves bounded | Single writer pool with a leak-proof semaphore |

Tested end-to-end in
`test/erllama_SUITE.erl:concurrent_complete_under_writer_cap` —
four models with distinct fingerprints running parallel completions
under one writer cap.

## A slightly longer example

A real load with all the cache parameters. The disk tier requires a
running `erllama_cache_disk_srv` started by the operator; the RAM tier
(`erllama_cache_ram`) starts automatically with the application.

```erlang
{ok, _} = erllama_cache_disk_srv:start_link(my_disk, "/var/lib/erllama/kvc"),
{ok, Bin} = file:read_file("/srv/models/llama-3.1-8b.Q4_K_M.gguf"),
Fp = crypto:hash(sha256, Bin),
CtxHash = crypto:hash(sha256, term_to_binary({8192, 4096})),

{ok, M} = erllama:load_model(#{
    backend          => erllama_model_llama,
    model_path       => "/srv/models/llama-3.1-8b.Q4_K_M.gguf",
    model_opts       => #{
        n_gpu_layers => 99,
        split_mode   => layer,           %% multi-GPU split policy
        main_gpu     => 0,
        tensor_split => [0.5, 0.5]       %% 50/50 across two devices
    },
    context_opts     => #{
        n_ctx        => 8192,
        n_batch      => 4096,
        n_seq_max    => 4,               %% admit up to 4 concurrent reqs
        flash_attn   => auto,            %% boolean() | auto
        type_k       => f16,             %% KV cache element type
        type_v       => f16
    },
    fingerprint      => Fp,
    fingerprint_mode => safe,
    quant_type       => q4_k_m,
    quant_bits       => 4,
    ctx_params_hash  => CtxHash,
    context_size     => 8192,
    tier_srv         => my_disk,
    tier             => disk,
    policy           => #{
        boundary_trim_tokens   => 32,
        boundary_align_tokens  => 256,
        session_resume_wait_ms => 500,
        prefill_chunk_size     => 1024   %% cap per-tick prefill slice
    }
}).
```

Stateless OpenAI/Anthropic-shaped server:

```erlang
handle_completion(ModelId, Prompt) ->
    {ok, #{reply := Reply}} = erllama:complete(ModelId, Prompt),
    Reply.
```

No `parent_key`. The cache walks the new prompt backward by the
configured stride and finds the longest cached prefix. If the new
prompt is yesterday's conversation plus one fresh turn, the walk
hits.

Stateful Erlang-native multi-turn: the session layer threads
`parent_key` between turns. `complete/2,3` already returns the
`finish_key` for the full context; pass it as `parent_key` on the
next turn.

```erlang
%% First turn: cold prefill. The model fires an async finish save
%% and returns its key.
{ok, #{reply := R1, finish_key := ParentKey}} =
    erllama:complete(M, Prompt1),

%% Second turn: pass ParentKey to skip the longest-prefix walk.
{ok, #{reply := R2}} =
    erllama:complete(M, Prompt2, #{parent_key => ParentKey}).
```

Inspect cache state from a shell:

```erlang
1> erllama_cache:get_counters().
#{hits_exact => 142, hits_resume => 17, hits_longest_prefix => 89,
  misses => 12, saves_cold => 12, saves_continued => 67,
  saves_finish => 31, evictions => 3, ...}

2> erllama_cache_meta_srv:dump().
%% List of raw ETS rows:
%%   {Key, Tier, Size, LastUsedNs, Refcount, Status, HeaderBin,
%%    Location, TokensRef, Hits}
[{<<_:256>>, disk, 8388608, 1737..., 0, available, _, _, _, 4}, ...]
```

Per-model observability is lock-free (reads a public ETS row, never
crosses the model gen_statem):

```erlang
1> erllama:phase(<<"big">>).
generating
2> erllama:pending_len(<<"big">>).
3
3> erllama:last_cache_hit(<<"big">>).
#{kind => partial, prefix_len => 1024}
4> erllama:queue_depth(<<"big">>).
2
```

Warm a session without sampling. Useful for priming the cache
before a burst of short follow-ups, or for holding a warm context
across long pauses:

```erlang
{ok, Toks} = erllama:tokenize(M, SystemPrompt),
{ok, #{finish_key := FK}} = erllama:prefill_only(M, Toks),
%% Later turns pass FK as parent_key to skip the longest-prefix walk.
{ok, #{reply := R}} =
    erllama:complete(M, UserTurn, #{parent_key => FK}).
```

## Requirements

- Erlang/OTP **28**
- rebar3 **3.25+**
- C++17 toolchain (Apple clang or recent gcc; `cmake` >= 3.20)
- Apple Silicon: Metal + Accelerate auto-detected.
- Linux: BLAS auto-detected; CUDA off by default (toggle via
  `ERLLAMA_OPTS=-DGGML_CUDA=ON`).
- FreeBSD: `erlang-runtime28` from ports, plus `cmake bash gmake`.

## Architecture at a glance

```
erllama_sup
├── erllama_cache_sup
│   ├── erllama_cache_meta_srv      sole writer; meta + LRU + reservations
│   ├── erllama_cache_ram           RAM tier (ETS slabs)
│   ├── erllama_cache_ramfile_srv   ram_file tier
│   ├── erllama_cache_disk_srv      disk tier (plain read/write I/O)
│   └── erllama_cache_writer        writer pool, leak-proof semaphore
├── erllama_model_sup               simple_one_for_one for dynamic models
└── erllama_scheduler               memory-pressure poller (off by default)
```

Inside a request:

1. `erllama:complete/2` enters the per-model `gen_statem`.
2. **prefilling** — tokenize, then either hit the cache and
   `kv_unpack` (warm) or run `llama_decode` over the prompt (cold).
   Cold path fires an async `cold` save at the trimmed-prefix
   boundary.
3. **generating** — token-by-token greedy `llama_decode`. Every
   `continued_interval` tokens, fire an async `continued` save.
4. **idle** — fire an async `finish` save for the full prompt +
   reply. The KV state becomes evictable.

For the publish protocol, the reservation state machine, and the
exception-safe NIF wrappers, see
[internals/publish-protocol.md](internals/publish-protocol.md) and
[internals/nif-safety.md](internals/nif-safety.md).

## Status

**Pre-release.** Core cache, scheduler, and NIF: 166 EUnit + 11
PropEr + 7 stub Common Test cases. End-to-end CT suite gated on
`LLAMA_TEST_MODEL` (6 cases, passing locally with TinyLlama 1.1B
Q4_K_M).

See [CHANGELOG.md](CHANGELOG.md) for the release notes.

## Contributing

The contributor guide is [AGENTS.md](AGENTS.md). The short version:

```bash
rebar3 fmt          # auto-format (always run first)
rebar3 compile      # warnings_as_errors
rebar3 eunit        # unit tests
rebar3 proper       # property tests
rebar3 ct           # Common Test (without a real model)
rebar3 lint         # Elvis
rebar3 dialyzer     # static analysis
rebar3 xref         # cross-reference
```

End-to-end against a real GGUF:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.gguf \
    rebar3 ct --suite=test/erllama_real_model_SUITE
```

Bumping the vendored llama.cpp: see [UPDATE_LLAMA.md](UPDATE_LLAMA.md).

## Coming next: erllama_cluster

A separate OTP application is in development to coordinate a fleet of
erllama nodes into a single inference cluster. Each node continues to
run erllama as a standalone library — local model loading, local KV
cache, local inference. The cluster layer sits on top and decides
which node serves which request.

Three distribution strategies, all in v1:

- **Request distribution** with pluggable load-balancing and
  cache-affinity routing — follow-up requests prefer the node that
  warmed the KV cache for the prefix.
- **Speculative decoding across nodes** — small draft model on one
  node, large verifier on another, coordinated per request.
- **Pipeline parallelism** — models too large for one node split by
  layer ranges across multiple nodes, hidden states passed between
  shards as Erlang binaries.

Transport is QUIC, via Erlang distribution carried over
[erlang_quic](https://github.com/benoitc/erlang_quic) — a pure Erlang
QUIC implementation, no C NIF in the protocol path. Circuit breakers
per `{Node, ModelId}` driven by `nodeup`/`nodedown` rather than
application-level pings. A globally registered scheduler handles
cluster-wide GPU budgeting and on-demand model placement, with local
fallback schedulers elected by `pg` quorum on network partition.

Repository: <https://github.com/erllama/erllama_cluster> (under
construction).

## Acknowledgements

Same idea as [antirez/ds4](https://github.com/antirez/ds4).

## License

MIT. Copyright (c) 2026 Benoit Chesneau. See [LICENSE](LICENSE).

The vendored `c_src/llama.cpp/` retains its upstream MIT license; see
`c_src/llama.cpp/LICENSE`.
