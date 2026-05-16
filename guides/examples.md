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
{ok, _} = erllama_cache_disk_srv:start_link(my_disk, "/var/lib/erllama/kvc"),
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
    tier_srv         => my_disk,
    tier             => disk
}),

{ok, #{reply := Reply}} =
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
handle_chat(ModelId, Prompt) ->
    {ok, #{reply := Reply}} =
        erllama:complete(ModelId, Prompt, #{response_tokens => 256}),
    {200, [{"Content-Type", "text/plain"}], Reply}.
```

Hits show up as `hits_longest_prefix` in `erllama_cache:get_counters/0`.

## 3. Multi-turn Erlang-native session (tracks parent_key)

The session layer threads `parent_key` between turns. `complete/2,3`
returns the `finish_key` for the full context — pass it as
`parent_key` on the next turn.

```erlang
chat(Model, Prompt, undefined) ->
    {ok, #{reply := Reply, finish_key := K}} =
        erllama:complete(Model, Prompt, #{}),
    {Reply, K};
chat(Model, Prompt, ParentKey) ->
    {ok, #{reply := Reply, finish_key := K}} =
        erllama:complete(Model, Prompt, #{parent_key => ParentKey}),
    {Reply, K}.

%% Driver.
{R1, K1} = chat(M, <<"User: hello\nAssistant:">>, undefined),
{R2, _K2} = chat(M,
    <<"User: hello\nAssistant: ", R1/binary,
      "\nUser: tell me a joke\nAssistant:">>,
    K1),
ok.
```

Passing `parent_key` skips the longest-prefix walk and resumes
directly from the previous turn's finish save. `finish_key` is
`undefined` if the finish save was suppressed (token count below
`min_tokens`); guard with a pattern match if your sessions can be
that short.

## 4. Multiple loaded models

Model ids are `binary()` (the registered name).

```erlang
{ok, _} = erllama:load_model(<<"tiny">>, TinyConfig),
{ok, _} = erllama:load_model(<<"big">>,  BigConfig),

{ok, #{reply := R1}} = erllama:complete(<<"tiny">>, <<"summarise: ...">>),
{ok, #{reply := R2}} = erllama:complete(<<"big">>,  <<"deep analysis of: ...">>),

ok = erllama:unload(<<"tiny">>),
ok = erllama:unload(<<"big">>).
```

Both share one `erllama_cache` instance. Cache rows are scoped by
fingerprint, so the two models never collide.

## 5. Concurrent agents on a shared system prompt

```erlang
ModelId = <<"assistant">>,
SharedPrefix = <<"You are a helpful assistant.\n">>,
Parent = self(),

%% Spawn N workers; each appends a different user query but they
%% all start with the same prefix. After the first agent's cold
%% prefill saves, every subsequent agent gets a longest-prefix hit
%% on the shared part and only prefills its tail.
Workers = [
    spawn(fun() ->
        Q = list_to_binary(io_lib:format("Worker ~p question.", [N])),
        Prompt = <<SharedPrefix/binary, Q/binary>>,
        {ok, #{reply := Reply}} = erllama:complete(ModelId, Prompt),
        Parent ! {N, Reply}
    end) || N <- lists:seq(1, 8)
],

%% Collect.
Replies = [receive {N, R} -> {N, R} end || N <- lists:seq(1, 8)],
Replies.
```

## 6. Streaming tokens (`infer/4`) with cancellation

```erlang
{ok, Tokens} = erllama:tokenize(ModelId, <<"Once upon a time">>),
{ok, Ref} = erllama:infer(ModelId, Tokens,
                           #{response_tokens => 200}, self()),

loop(Ref) ->
    receive
        {erllama_token, Ref, Fragment} ->
            io:put_chars(Fragment),
            loop(Ref);
        {erllama_done, Ref, _Stats} ->
            io:nl(),
            ok;
        {erllama_error, Ref, Reason} ->
            {error, Reason}
    after 30000 ->
        erllama:cancel(Ref),
        %% Still drain the final done message after cancel.
        loop(Ref)
    end.
```

`cancel/1` is observed at the next inter-token boundary; the model
always emits a final `{erllama_done, Ref, Stats}` with
`#{cancelled => true}`.

## 7. Chat template + embeddings

```erlang
%% Render a chat request through the model's built-in GGUF template
%% and tokenise it in one shot. Backed by llama_chat_apply_template.
{ok, ChatTokens} = erllama:apply_chat_template(ModelId, #{
    messages => [
        #{role => system,    content => <<"You are concise.">>},
        #{role => user,      content => <<"What's 2+2?">>}
    ]
}),
{ok, Ref} = erllama:infer(ModelId, ChatTokens, #{response_tokens => 8}, self()).
```

```erlang
%% Pooled sentence embedding via llama_get_embeddings_seq.
%% The model must have been loaded with embedding-friendly settings
%% (see guides/loading.md). Returns a list of floats.
{ok, Toks}      = erllama:tokenize(ModelId, <<"The quick brown fox.">>),
{ok, Embedding} = erllama:embed(ModelId, Toks).
```

## 8. Grammar-constrained sampling (GBNF)

```erlang
%% Force the model to emit a JSON-shaped string with a digit value.
Grammar = <<
    "root ::= \"{\" ws \"\\\"n\\\":\" ws digit \"}\"\n"
    "digit ::= [0-9]\n"
    "ws    ::= [ \\t\\n]*"
>>,
{ok, Toks} = erllama:tokenize(ModelId, <<"Reply with JSON:">>),
{ok, Ref}  = erllama:infer(ModelId, Toks,
                           #{response_tokens => 32, grammar => Grammar},
                           self()).
```

The grammar is per-request: the sampler chain is reset to grammar →
greedy for the duration of the request and cleared on completion or
cancellation.

## 9. Stop sequences

Halt generation as soon as one of the caller-supplied strings
appears in the detokenised output. The match is trimmed from the
streamed text and from the synchronous `reply`; the matched binary
is reported as `stop_sequence` in the result map and in the
streaming `Stats`.

```erlang
{ok, #{reply := Reply, finish_reason := FinishReason} = Result} =
    erllama:complete(ModelId, <<"Write a short list, then say END.">>,
                     #{response_tokens => 64,
                       stop_sequences => [<<"END">>, <<"\n\nUser:">>]}),
case maps:find(stop_sequence, Result) of
    {ok, S} ->
        %% Generation halted on a caller-supplied stop string.
        %% FinishReason is `stop`; `reply` has been trimmed at the
        %% first occurrence of S.
        io:format("stopped at ~p, reply=~p~n", [S, Reply]);
    error ->
        %% No stop string fired: FinishReason is `length`, `stop`
        %% (natural EOG), or `cancelled`.
        io:format("finished ~p, reply=~p~n", [FinishReason, Reply])
end.
```

Streaming works the same way: the matched string never appears in
any `{erllama_token, _, _}` chunk, and the final
`{erllama_done, _, Stats}` carries the matched value:

```erlang
{ok, Tokens} = erllama:tokenize(ModelId, Prompt),
{ok, Ref} = erllama:infer(ModelId, Tokens,
                          #{response_tokens => 200,
                            stop_sequences => [<<"\n\nUser:">>]},
                          self()),

receive
    {erllama_done, Ref, #{stop_sequence := <<"\n\nUser:">>}} ->
        stopped_at_user_turn;
    {erllama_done, Ref, _Stats} ->
        finished_without_match
end.
```

The first occurrence by **text position** wins; the scanner holds
back the last `(max_stop_len - 1)` bytes of each chunk so a match
crossing a chunk boundary is still detected, then flushes the tail
on terminal end when no match fired.

## 10. Extended thinking (`thinking_delta` / `thinking_end`)

When `Params` carries `thinking => enabled` and the backend
supports extended thinking, streaming requests receive
thinking-phase fragments as `{erllama_token, Ref, {thinking_delta,
Bin}}` and a single `{erllama_thinking_end, Ref, Sig}` close marker
before any non-thinking token. `Sig` is an opaque integrity
signature (or `<<>>` when none is available) that thinking-capable
SDKs verify on the next turn.

```erlang
{ok, Tokens} = erllama:tokenize(ModelId, <<"Solve: 23 * 17.">>),
{ok, Ref} = erllama:infer(ModelId, Tokens,
                          #{response_tokens => 256,
                            thinking => enabled},
                          self()),

loop(Ref, _Thinking = <<>>, _Answer = <<>>, _Sig = <<>>) ->
    receive
        {erllama_token, Ref, {thinking_delta, Bin}} ->
            loop(Ref, <<Thinking/binary, Bin/binary>>, Answer, Sig);
        {erllama_thinking_end, Ref, NewSig} ->
            loop(Ref, Thinking, Answer, NewSig);
        {erllama_token, Ref, Bin} when is_binary(Bin) ->
            loop(Ref, Thinking, <<Answer/binary, Bin/binary>>, Sig);
        {erllama_done, Ref, _Stats} ->
            #{thinking => Thinking, answer => Answer, signature => Sig}
    end.
```

With `thinking => disabled` (the default), backends without
thinking support, or when no thinking phase fires, no
`{thinking_delta, _}` payloads and no `erllama_thinking_end`
messages arrive — the stream is identical to a plain text-only
request.

## 11. Warm a session without sampling (`prefill_only/2`)

Useful for priming the cache before a burst of short follow-ups,
or for holding a warm session across long pauses without consuming
generation budget.

```erlang
{ok, SysToks} = erllama:apply_chat_template(ModelId, #{
    messages => [
        #{role => system, content => SystemPrompt},
        #{role => user,   content => FirstUserTurn}
    ]
}),
{ok, #{finish_key := FK,
       committed_tokens := N,
       cache_hit_kind := cold}} =
    erllama:prefill_only(ModelId, SysToks),

%% Later turn: pass FK as parent_key so the next complete/3 skips
%% the longest-prefix walk and resumes from the warm row.
{ok, #{reply := R}} =
    erllama:complete(ModelId, NextUserTurn, #{parent_key => FK}).
```

`finish_key` is `undefined` if the prompt was shorter than
`min_tokens` and the finish save was suppressed.

## 12. Multi-tenant concurrent decoding (`n_seq_max`)

Default is single-tenant. Opt in with `context_opts.n_seq_max > 1`
and up to N requests prefill and decode concurrently through one
`llama_decode` per tick.

```erlang
{ok, _} = erllama:load_model(<<"chat">>, Config#{
    context_opts => #{n_ctx => 8192, n_batch => 4096, n_seq_max => 4}
}),

Parent = self(),
Workers = [
    spawn(fun() ->
        Q = list_to_binary(io_lib:format("Worker ~p question.", [N])),
        {ok, #{reply := R}} = erllama:complete(<<"chat">>, Q),
        Parent ! {N, R}
    end) || N <- lists:seq(1, 4)
],
[receive {N, _R} -> ok end || N <- lists:seq(1, 4)].
```

Per-request samplers are isolated: each worker can pass its own
`temperature`, `seed`, or `grammar` in the `Opts` map without
spilling sampler state across the other in-flight requests.
Admissions past `n_seq_max` queue FIFO in the model's `pending`
list - observable via `erllama:pending_len/1`.

Long prompts are sliced by `prefill_chunk_size` (default
`max(64, n_batch div 4)`) so a single heavy prompt cannot
monopolise the batch. See [caching](caching.md) for warm-prefix
behaviour across concurrent workers.

## 13. Lock-free observability for routers

A cluster router that bin-packs requests onto the least-loaded node
should not serialise behind the work it is probing. These accessors
read a public ETS row, never cross the model gen_statem:

```erlang
1> erllama:phase(<<"chat">>).
generating
2> erllama:pending_len(<<"chat">>).
3
3> erllama:queue_depth(<<"chat">>).
1
4> erllama:last_cache_hit(<<"chat">>).
#{kind => partial, prefix_len => 1024}
5> erllama:queue_depth().
%% global: total admitted streaming infer/4 rows across all loaded models.
4
```

`phase/1` returns `idle | prefilling | generating` and falls back to
`idle` for unknown ids. `last_cache_hit/1` returns `undefined` if
the model has not admitted any request yet. All four are O(1) ETS
reads and safe to call from a hot path or via `erpc` from another
node.

## 14. Inspecting cache state

```erlang
%% Hit/miss/save counters and per-path latency totals.
Counters = erllama_cache:get_counters(),
io:format("~p~n", [Counters]).

%% Every row in the index. dump/0 returns raw ETS tuples; the layout
%% is documented in include/erllama_cache.hrl:
%%   {Key, Tier, Size, LastUsedNs, Refcount, Status, Header,
%%    Location, TokensRef, Hits}
Dump = erllama_cache_meta_srv:dump(),
[io:format("tier=~p size=~p refs=~p~n", [Tier, Size, Refs])
 || {_Key, Tier, Size, _Lru, Refs, _Status, _Hdr, _Loc, _Tok, _Hits} <- Dump].

%% Free at least 256 MiB, oldest LRU first, RAM tiers only.
erllama_cache:evict_bytes(256 * 1024 * 1024, [ram, ram_file]).

%% Synchronous full eviction pass.
erllama_cache:gc().
```

## 15. Memory-pressure-driven eviction (in `sys.config`)

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

## 16. Cache-only tests (no model required)

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

## 17. End-to-end against a real GGUF

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.Q4_K_M.gguf \
    rebar3 ct --suite=test/erllama_real_model_SUITE
```

The 6-case suite covers cold prefill, warm restore, multi-turn
parent-key resume, longest-prefix walk, eviction, and a multi-model
concurrent run. Without the env var it skips so default
`rebar3 ct` stays green.

## 18. Microbench: cold vs. warm

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
