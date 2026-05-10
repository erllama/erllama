# Examples

Drop-in patterns for common erllama workflows. Each block is a
self-contained snippet that runs from `rebar3 shell` after
`{ok, _} = application:ensure_all_started(erllama)` (the boot is
omitted from the snippets; assume it's there).

## 10-second smoke test (no model required)

The cache subsystem is independently usable. From `rebar3 shell`:

```erlang
1> {ok, _} = application:ensure_all_started(erllama).
2> ok = filelib:ensure_path("/tmp/edemo").
3> {ok, _} = erllama_cache_disk_srv:start_link(d, "/tmp/edemo").
4> Meta = #{save_reason => cold, quant_bits => 16,
           fingerprint => binary:copy(<<170>>, 32),
           fingerprint_mode => safe, quant_type => f16,
           ctx_params_hash => binary:copy(<<187>>, 32),
           tokens => [1,2,3], context_size => 4096,
           prompt_text => <<>>, hostname => <<"d">>,
           erllama_version => <<"0.1.0">>}.
5> {ok, K, _Header, Size} =
       erllama_cache_disk_srv:save(d, Meta, <<"hi">>).
6> {ok, _Info, <<"hi">>} = erllama_cache_disk_srv:load(d, K).
7> erllama_cache:get_counters().
```

Verified end-to-end: a published .kvc file, the meta-server
registers it, the load path round-trips the payload, and the
counters reflect one cold save plus one cache miss followed by one
exact hit. **No model loaded — `llama_backend_init` doesn't run.**

## 1. Load a model and run a one-shot completion

```erlang
{ok, Bin} = file:read_file("/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf"),
Fp = crypto:hash(sha256, Bin),

{ok, M} = erllama:load_model(#{
    backend          => erllama_model_llama,
    model_path       => "/srv/models/tinyllama-1.1b-chat.Q4_K_M.gguf",
    fingerprint      => Fp,
    fingerprint_mode => safe,
    quant_type       => q4_k_m,
    quant_bits       => 4,
    ctx_params_hash  => crypto:hash(sha256, term_to_binary({2048, 512})),
    context_size     => 2048,
    tier_srv         => default_disk,
    tier             => disk
}),

{ok, Reply, _Tokens} =
    erllama:complete(M, <<"Once upon a time, in a quiet village">>),

io:format("~s~n", [Reply]),
ok = erllama:unload(M).
```

First call cold-prefills the prompt and async-saves cold + finish
rows. Repeating the same call hits the cache via the exact-key path.

## 2. Stateless HTTP server (resends full conversation per turn)

```erlang
%% Inside your request handler. The cache walks the new prompt's
%% tokens backward by `boundary_align_tokens` and resumes from the
%% longest published prefix automatically — no parent_key needed.
handle_chat(Prompt) ->
    {ok, Reply, _Tokens} =
        erllama:complete(model_a3b1, Prompt, #{response_tokens => 256}),
    {200, [{"Content-Type", "text/plain"}], Reply}.
```

Hits show up as `hits_longest_prefix` in `erllama_cache:get_counters/0`.

## 3. Multi-turn Erlang-native session (tracks parent_key)

```erlang
chat(Model, Prompt, undefined) ->
    {ok, Reply, _} = erllama:complete(Model, Prompt, #{}),
    {ok, K} = erllama_cache:last_finish_key(Model),
    {Reply, K};
chat(Model, Prompt, ParentKey) ->
    {ok, Reply, _} =
        erllama:complete(Model, Prompt, #{parent_key => ParentKey}),
    {ok, K} = erllama_cache:last_finish_key(Model),
    {Reply, K}.

%% Driver:
{R1, K1} = chat(M, <<"User: hello\nAssistant:">>, undefined),
{R2, K2} = chat(M,
    <<"User: hello\nAssistant: ", R1/binary,
      "\nUser: tell me a joke\nAssistant:">>,
    K1),
ok.
```

Passing `parent_key` skips the longest-prefix walk and resumes
directly from the previous turn's finish save.

## 4. Multiple loaded models

```erlang
{ok, _} = erllama:load_model(tiny, TinyConfig),
{ok, _} = erllama:load_model(big,  BigConfig),

{ok, R1, _} = erllama:complete(tiny, <<"summarise: ...">>),
{ok, R2, _} = erllama:complete(big,  <<"deep analysis of: ...">>),

ok = erllama:unload(tiny),
ok = erllama:unload(big).
```

Both share one `erllama_cache` instance. Cache rows are scoped by
fingerprint, so the two models never collide.

## 5. Concurrent agents on a shared system prompt

```erlang
SharedPrefix = <<"You are a helpful assistant.\n">>,

%% Spawn N workers; each appends a different user query but they
%% all start with the same prefix. After the first agent's cold
%% prefill saves, every subsequent agent gets a longest-prefix hit
%% on the shared part and only prefills its tail.
Workers = [
    spawn(fun() ->
        Q = list_to_binary(io_lib:format("Worker ~p question.", [N])),
        Prompt = <<SharedPrefix/binary, Q/binary>>,
        {ok, Reply, _} = erllama:complete(model_a3b1, Prompt),
        Parent ! {N, Reply}
    end) || N <- lists:seq(1, 8)
],

%% Collect.
Replies = [receive {N, R} -> {N, R} end || N <- lists:seq(1, 8)],
Replies.
```

## 6. Inspecting cache state

```erlang
%% Hit/miss/save counters and per-path latency totals.
Counters = erllama_cache:get_counters(),
io:format("~p~n", [Counters]).

%% Every row in the index, with tier, size, last_used_ns, refcount.
Dump = erllama_cache_meta_srv:dump(),
[io:format("~s ~p ~p~n",
    [erllama_cache_key:to_hex(K), Tier, Size])
 || #{key := K, tier := Tier, size := Size} <- Dump].

%% Free at least 256 MiB, oldest LRU first, RAM tiers only.
erllama_cache:evict_bytes(256 * 1024 * 1024, [ram, ram_file]).

%% Synchronous full eviction pass.
erllama_cache:gc().
```

## 7. Memory-pressure-driven eviction (in `sys.config`)

```erlang
{erllama, [
  {scheduler, #{
    enabled         => true,
    pressure_source => system,        %% memsup-backed, portable
    interval_ms     => 5000,
    high_watermark  => 0.85,
    low_watermark   => 0.75,
    evict_tiers     => [ram, ram_file] %% disk fills to its own quota
  }}
]}.
```

Sources shipped: `noop`, `system`, `nvidia_smi`, `{module, M}`. Roll
your own with `-behaviour(erllama_pressure)` and pass
`{module, M}` as the source.

## 8. Cache-only tests (no model required)

The cache subsystem is independently usable. eunit tests that
exercise save/load round-trips never touch llama.cpp:

```erlang
%% test/my_cache_test.erl
-module(my_cache_test).
-include_lib("eunit/include/eunit.hrl").
-include_lib("erllama/include/erllama_cache.hrl").

with_disk(Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    {ok, Dir} = file:make_dir("/tmp/my_cache_test"), %% ensure exists
    {ok, _} = erllama_cache_disk_srv:start_link(t, "/tmp/my_cache_test"),
    try Body() after
        catch gen_server:stop(t),
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv)
    end.

round_trip_test() ->
    with_disk(fun() ->
        Meta = #{
            save_reason => cold,
            quant_bits => 16,
            fingerprint => binary:copy(<<16#AA>>, 32),
            fingerprint_mode => safe,
            quant_type => f16,
            ctx_params_hash => binary:copy(<<16#BB>>, 32),
            tokens => [1, 2, 3],
            context_size => 4096
        },
        {ok, Key, _, _} = erllama_cache_disk_srv:save(t, Meta, <<"data">>),
        ?assertMatch({ok, _Info, <<"data">>},
                     erllama_cache_disk_srv:load(t, Key))
    end).
```

The lazy `llama_backend_init` means cache-only tests never trigger
`ggml_backend_load_all` — no Metal/CUDA discovery cost.

## 9. End-to-end against a real GGUF

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.Q4_K_M.gguf \
    rebar3 ct --suite=test/erllama_real_model_SUITE
```

The 6-case suite covers cold prefill, warm restore, multi-turn
parent-key resume, longest-prefix walk, eviction, and a multi-model
concurrent run. Without the env var it skips so default
`rebar3 ct` stays green.

## 10. Microbench: cold vs. warm

```bash
bench/run.sh tiny    # TinyLlama 1.1B Q4_K_M
bench/run.sh large   # LLaMA-3 8B Q4_K_M (needs the file)
```

`bench/run.sh` drives a `cold_vs_warm` matrix plus a 4-agent
shared-prefix scenario; see `bench/README.md`.

## See also

- [Loading a model](loading.md) — every option to
  `erllama:load_model/1,2`, with examples and pitfalls.
- [Caching](caching.md) — tiers, save reasons, lookup paths,
  watermarks.
- [Configuration](configuration.md) — full `sys.config` reference.
- [Building](building.md) — platform-specific build notes.
