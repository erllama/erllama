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

5> {ok, Reply, _} = erllama:complete(M, <<"Once upon a time">>).
%% ~3 s on a CPU box. Prompt prefill, async cold save fired.

6> {ok, Reply2, _} = erllama:complete(M, <<"Once upon a time">>).
%% ~10 ms. Cache hit; KV state restored, one decode for fresh logits.

7> {ok, _, _} = erllama:complete(M, <<"Once upon a time, in a quiet village">>).
%% ~50 ms. Longest-prefix walk found the cached row even though
%% the new prompt is longer.
```

`load_model/1` returns a binary `model_id` that is also the registered
name for the underlying gen_statem. Pass it to `complete/2,3`,
`unload/1`, etc.

That is the whole pitch. The cache is on by default, runs under
its own supervisor, and never returns approximate matches.

## What you get

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
  Per-counter cost is ~10–20 ns; you cannot meaningfully turn them
  off.

## Installation

erllama targets Erlang/OTP **28** with rebar3 **3.25+**.

Add to `rebar.config`:

```erlang
{deps, [
    {erllama, "~> 0.1"}
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
    model_opts       => #{n_gpu_layers => 99},
    context_opts     => #{n_ctx => 8192, n_batch => 4096},
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
        session_resume_wait_ms => 500
    }
}).
```

Stateless OpenAI/Anthropic-shaped server:

```erlang
handle_completion(ModelId, Prompt) ->
    {ok, Reply, _Tokens} = erllama:complete(ModelId, Prompt),
    Reply.
```

No `parent_key`. The cache walks the new prompt backward by the
configured stride and finds the longest cached prefix. If the new
prompt is yesterday's conversation plus one fresh turn, the walk
hits.

Stateful Erlang-native multi-turn: the session layer threads
`parent_key` between turns. The previous turn's finish-save key is
the parent of the next call. It is held by the calling session
process, not retrieved from the cache.

```erlang
%% First turn: cold prefill. The model fires an async finish save
%% whose key is sha256(fingerprint || quant || ctx_params || tokens).
{ok, R1, Tokens1} = erllama:complete(M, Prompt1),
ParentKey = erllama_cache_key:make(#{
    fingerprint => Fp,
    quant_type  => q4_k_m,
    ctx_params_hash => CtxHash,
    tokens      => Tokens1
}),

%% Second turn: pass ParentKey to skip the longest-prefix walk.
{ok, R2, _} = erllama:complete(M, Prompt2, #{parent_key => ParentKey}).
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

## Acknowledgements

The on-disk `KVC` file format (48-byte header, `"KVC"` magic), the
save-reasons taxonomy, and the `boundary_trim_tokens` /
`boundary_align_tokens` defaults are direct ports from
[antirez/ds4](https://github.com/antirez/ds4). ds4 pioneered the
"disk KV cache as a first-class resume mechanism" idea for
DeepSeek V4; erllama generalises that pattern as an Erlang/OTP
library across any GGUF llama.cpp can load. The rest — multi-tier
storage, supervised writer pool, longest-prefix walk indexing,
memory-pressure scheduler, exception-safe NIF — is erllama's own.

## License

MIT. Copyright (c) 2026 Benoit Chesneau. See [LICENSE](LICENSE).

The vendored `c_src/llama.cpp/` retains its upstream MIT license; see
`c_src/llama.cpp/LICENSE`.
