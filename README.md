# erllama

[![CI](https://github.com/erllama/erllama/actions/workflows/ci.yml/badge.svg)](https://github.com/erllama/erllama/actions/workflows/ci.yml)
[![Hex.pm](https://img.shields.io/hexpm/v/erllama.svg)](https://hex.pm/packages/erllama)

Run `llama.cpp` from Erlang. Keep prompts warm. Stay inside OTP.

erllama is a native Erlang/OTP runtime for `llama.cpp` with supervised
model processes, OpenAI-shaped completion APIs, and a token-exact KV
cache that turns repeated prompt prefill from seconds into milliseconds.

If your app sends the same system prompt, agent scaffold, or conversation
prefix again and again, erllama saves the model state once and restores it
on the next request. No fuzzy matching. No hidden session server. Just
exact tokens, exact cache keys, and OTP supervision around the whole path.

## Why erllama?

- **Fast repeat prompts.** Cache hits restore KV state instead of
  recomputing prompt prefill.
- **Native OTP shape.** Each loaded model is a supervised process with a
  clear lifecycle: load, complete, stream, observe, unload.
- **Bigger-than-RAM warmth.** Hot prefixes can live in RAM, warm prefixes
  in tmpfs, and large working sets on disk.
- **Stateless-server friendly.** Resend the full conversation every turn
  and still get longest-prefix cache hits.
- **Multi-model safe.** Cache rows include the model fingerprint and
  context shape, so different models never collide on identical prompts.
- **Observable by default.** Hit/miss counters and per-model state probes
  are cheap enough to call from routers.
- **Built on `llama.cpp`.** Local GGUF inference with the platform support
  you expect: Metal, BLAS, CUDA toggles, and plain CPU fallback.

## Quick taste

```erlang
1> {ok, _} = application:ensure_all_started(erllama).
2> Path = "/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf".
3> {ok, Bin} = file:read_file(Path).
4> {ok, Model} = erllama:load_model(#{
       backend => erllama_model_llama,
       model_path => Path,
       fingerprint => crypto:hash(sha256, Bin)
   }).
{ok, <<"erllama_model_2375">>}

5> {ok, #{reply := Reply, finish_key := Key}} =
       erllama:complete(Model, <<"Once upon a time">>).
%% First call: cold prefill, async save.

6> {ok, #{reply := Reply2}} =
       erllama:complete(Model, <<"Once upon a time">>).
%% Same prompt: KV cache restore.

7> {ok, #{reply := Reply3}} =
       erllama:complete(Model,
                        <<"Once upon a time, in a quiet village">>).
%% Longer prompt: longest cached prefix wins.

8> {ok, #{reply := Reply4}} =
       erllama:complete(Model, <<"and they lived happily ever after">>,
                        #{parent_key => Key}).
%% Stateful resume from the previous finish save.
```

`load_model/1` returns a binary model id. Pass it to `complete/2,3`,
`infer/4`, `tokenize/2`, `unload/1`, and the rest of the public API.

## Install

erllama targets Erlang/OTP **28** and rebar3 **3.25+**.

Add it to `rebar.config`:

```erlang
{deps, [
    {erllama, "~> 0.5"}
]}.
```

Then start the application before loading models:

```erlang
{ok, _} = application:ensure_all_started(erllama).
```

The first compile builds the vendored `llama.cpp`. See
[Building](guides/building.md) for platform notes and CUDA/Metal options.

## Common patterns

### Stateless HTTP completion

OpenAI/Anthropic-shaped servers usually resend the whole conversation on
each turn. That is fine. erllama walks the prompt backward and restores
the longest exact prefix it has already saved.

```erlang
handle_completion(ModelId, Prompt) ->
    {ok, #{reply := Reply}} =
        erllama:complete(ModelId, Prompt, #{response_tokens => 256}),
    Reply.
```

### Stateful Erlang session

If your session process already tracks turns, keep the returned
`finish_key` and pass it as `parent_key` on the next request. That skips
the longest-prefix walk and resumes directly from the saved row.

```erlang
{ok, #{reply := R1, finish_key := K1}} =
    erllama:complete(ModelId, Prompt1),

{ok, #{reply := R2, finish_key := K2}} =
    erllama:complete(ModelId, Prompt2, #{parent_key => K1}).
```

### Many models in one BEAM

Each loaded model is its own supervised process. The cache is shared, but
rows are fingerprint-segregated.

```erlang
{ok, _} = erllama:load_model(<<"tiny">>, TinyConfig),
{ok, _} = erllama:load_model(<<"big">>, BigConfig),

{ok, #{reply := R1}} = erllama:complete(<<"tiny">>, <<"summarise: ...">>),
{ok, #{reply := R2}} = erllama:complete(<<"big">>, <<"deep analysis: ...">>),

ok = erllama:unload(<<"tiny">>).
```

### Inspect live state

```erlang
1> erllama_cache:get_counters().
#{hits_exact => 142, hits_resume => 17, hits_longest_prefix => 89,
  misses => 12, saves_cold => 12, saves_finish => 31, ...}

2> erllama:phase(<<"big">>).
generating
3> erllama:pending_len(<<"big">>).
3
4> erllama:last_cache_hit(<<"big">>).
#{kind => partial, prefix_len => 1024}
```

## Documentation

| Need | Read |
|---|---|
| Load a model | [Loading a model](guides/loading.md) |
| Configure cache tiers and save policy | [Caching](guides/caching.md) |
| Configure `sys.config` and per-model options | [Configuration](guides/configuration.md) |
| Build from source | [Building](guides/building.md) |
| Copy working snippets | [Examples](guides/examples.md) |
| Stream tool calls while preserving cache hits | [Tool calls](guides/tool-calls.md) |
| Understand cache design tradeoffs | [Cache design](internals/cache-design.md) |
| Understand crash-safe save publication | [Publish protocol](internals/publish-protocol.md) |
| Understand request admission and decode flow | [Request lifecycle](internals/request-lifecycle.md) |
| Understand NIF lifetime safety | [NIF safety](internals/nif-safety.md) |

API reference for `erllama`, `erllama_cache`, `erllama_scheduler`, and
`erllama_nif` is published on
[HexDocs](https://hexdocs.pm/erllama). You can also build it locally:

```bash
rebar3 ex_doc
```

## Architecture

```text
erllama_sup
├── erllama_cache_sup
│   ├── erllama_cache_meta_srv
│   ├── erllama_cache_ram
│   └── erllama_cache_writer
├── erllama_registry
├── erllama_inflight
├── erllama_model_sup
│   └── erllama_model      one supervised gen_statem per loaded model
└── erllama_scheduler      memory-pressure poller, off by default
```

Disk and `ram_file` tier servers are started by the operator, one per
root directory, then referenced by loaded models through `tier_srv` and
`tier`.

The important invariant is simple: cache hits are token-exact. A key is
derived from the model fingerprint, quantization, context shape, and full
token list. erllama may find a shorter saved prefix for a longer prompt,
but it never returns an approximate match.

## Requirements

- Erlang/OTP **28**
- rebar3 **3.25+**
- C++17 toolchain
- `cmake` >= 3.20
- Apple Silicon: Metal + Accelerate are auto-detected
- Linux: BLAS is auto-detected; CUDA is enabled with
  `ERLLAMA_OPTS=-DGGML_CUDA=ON`
- FreeBSD: `erlang-runtime28` plus `cmake bash gmake`

## Status

erllama is pre-release. The cache, scheduler, and NIF safety wrappers have
unit, property, and Common Test coverage. The real-model Common Test suite
is gated by `LLAMA_TEST_MODEL` so normal CI can run without a GGUF file.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Contributing

The contributor guide is [AGENTS.md](AGENTS.md). The short version:

```bash
rebar3 fmt
rebar3 compile
rebar3 eunit
rebar3 proper
rebar3 ct
rebar3 lint
rebar3 dialyzer
rebar3 xref
```

Run the real-model suite when you have a GGUF available:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.Q4_K_M.gguf \
    rebar3 ct --suite=test/erllama_real_model_SUITE
```

Bumping the vendored `llama.cpp` is covered in
[UPDATE_LLAMA.md](UPDATE_LLAMA.md).

## Related projects

`erllama_cluster` is planned as a separate OTP application for routing,
cache-aware placement, speculative decoding, and distributed inference
across erllama nodes.

Repository: <https://github.com/erllama/erllama_cluster>

## Acknowledgements

Same idea as [antirez/ds4](https://github.com/antirez/ds4).

## License

MIT. Copyright (c) 2026 Benoit Chesneau. See [LICENSE](LICENSE).

The vendored `c_src/llama.cpp/` retains its upstream MIT license; see
`c_src/llama.cpp/LICENSE`.
