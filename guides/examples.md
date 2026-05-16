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

Some LLMs emit a "thinking" phase before their actual answer —
typically text wrapped in template-defined markers like
`<think>...</think>`. erllama surfaces those thoughts as a
separate stream so Anthropic-Messages SDKs can render and verify
them.

### Enabling thinking on a real model

Tell the backend which token strings open and close a thinking
block for *this* GGUF. Each model has its own — qwen3 uses
`<think>`/`</think>`, deepseek-r1 uses different markers. The
strings are tokenised through the model's own vocabulary at load
time.

```erlang
erllama:load_model(<<"qwen3-thinking">>, #{
    backend => erllama_model_llama,
    model_path => "/models/qwen3-7b.gguf",
    context_size => 8192,
    policy => default_policy(),
    thinking_markers => #{
        start => <<"<think>">>,
        end   => <<"</think>">>
    }
}).
```

Omitting `thinking_markers` (or passing `#{}`) keeps the backend
on the non-thinking path — requests with `thinking => enabled`
behave identically to a normal text request, no thinking messages
arrive.

For tamper-proof signatures on each thinking block, set a
node-wide HMAC key once at boot:

```erlang
{erllama, [
    {thinking_signing_key, <<"a-32-byte-secret-or-longer">>}
]}.
```

`erllama_model_llama:thinking_signature/3` HMACs the observed
thinking text with this key. Leaving it unset returns `<<>>`, the
documented "no signature" path — the downstream forwards no
`signature_delta` event.

### Per-request usage

```erlang
{ok, Tokens} = erllama:tokenize(<<"qwen3-thinking">>, <<"Solve: 23 * 17.">>),
{ok, Ref} = erllama:infer(<<"qwen3-thinking">>, Tokens,
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

### Capping the thinking phase (`thinking_budget_tokens`)

Anthropic's API takes a `thinking.budget_tokens` hint that caps
the thinking phase. erllama honours it as the maximum number of
`{thinking_delta, _}` payloads to deliver. Once the budget is
reached, the scheduler synthesises the `erllama_thinking_end`
close immediately and routes any further model-emitted thinking
tokens through the normal post-thinking pipeline so generation
still progresses.

```erlang
{ok, Ref} = erllama:infer(<<"qwen3-thinking">>, Tokens,
                          #{response_tokens => 256,
                            thinking => enabled,
                            thinking_budget_tokens => 64},
                          self()).
```

Non-positive values and a missing key both mean "no cap".

## 11. Tool-call streaming (`tool_call_markers`)

Models whose chat template emits structured tool-call markers
(qwen3 with `<tool_call>...</tool_call>`, DSML, OpenAI-style XML,
etc.) can surface those boundaries on the streaming wire so an
HTTP front end can capture the exact bytes the model produced
under a tool id, and splice them back verbatim on the next turn.

Declare the markers per model:

```erlang
erllama:load_model(<<"qwen3-tool">>, #{
    backend => erllama_model_llama,
    model_path => "/models/qwen3-7b.gguf",
    tool_call_markers => #{
        start => <<"<tool_call>">>,
        end   => <<"</tool_call>">>
    }
}).
```

A streaming `infer/4` against this model receives:

```erlang
{erllama_token, Ref, {tool_call_delta, Bin}}    %% one per chunk of the call body
{erllama_tool_call_end, Ref, FullBin :: binary()} %% concatenated bytes of every delta
```

`FullBin` is what to persist under a minted tool id — no need to
re-buffer the deltas yourself.

When the template also delimits string-typed argument bodies
(DSML's `string=true ... string=false`, XML's `<arg>...</arg>`),
declare payload markers too:

```erlang
tool_call_markers => #{
    start         => <<"<tool_call>">>,
    end           => <<"</tool_call>">>,
    payload_start => <<"<arg>">>,
    payload_end   => <<"</arg>">>
}
```

With payload markers set, erllama swaps to a greedy
(`temperature=0`) sampler for tool-call syntax tokens so the
syntax stays byte-deterministic, then back to the request's normal
sampler for payload bytes so long string arguments stay diverse.
Without payload markers the entire tool-call span uses the greedy
sampler.

Models loaded without `tool_call_markers` behave identically to
v0.4: no tool-call messages, no sampler swap.

## 12. Warm a session without sampling (`prefill_only/2,3`)

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

`prefill_only/3` accepts `Opts` with `parent_key`, useful when
chaining cache-warming calls across turns deterministically rather
than relying on the implicit longest-prefix walk:

```erlang
%% Materialise turn-1 bytes into cache.
{ok, #{finish_key := K1}} = erllama:prefill_only(ModelId, Turn1Tokens),

%% Extend the same prefix with turn 2's tokens; only the suffix
%% runs through prefill, returning K2 for the longer prompt.
{ok, #{finish_key := K2,
       cache_hit_kind := partial,
       cache_delta := #{read := R, created := C}}} =
    erllama:prefill_only(ModelId,
                         Turn1Tokens ++ Turn2Tokens,
                         #{parent_key => K1}).
```

## 13. Sticky sessions (`session_id` + `end_session/2`)

A `session_id` (any term) pins the request's seq_id across turns
so the next turn whose prompt continues the stored tokens
truncates-and-prefills in place on the already-live KV cells —
no `kv_unpack` from disk. The HTTP server's conversation id is
the natural value; nothing here is HTTP-specific.

```erlang
SessionId = ConvId,  %% your auth layer's conversation identifier

%% Turn 1: cold or partial via longest-prefix; on finish the
%% seq_id stays pinned to SessionId.
{ok, #{generated := Gen1}} =
    erllama:complete(ModelId, Turn1Prompt,
                     #{session_id => SessionId,
                       response_tokens => 32}),

%% Turn 2: the prompt extends the stored prefix (prior prompt +
%% generated). cache_hit_kind comes back as `sticky`: no disk
%% read, just suffix prefill on the live cells.
{ok, #{cache_hit_kind := sticky}} =
    erllama:complete(ModelId, Turn2Prompt,
                     #{session_id => SessionId,
                       response_tokens => 32}),

%% End the conversation explicitly when the user logs out / TTL
%% expires. Idempotent; unknown ids are a no-op.
ok = erllama:end_session(ModelId, SessionId).
```

Constraints:

- Concurrent admits on the same `session_id` return `{error,
  sticky_busy}`; serialise per session (typical HTTP per-user
  request pipelines do this naturally).
- A new prompt that *diverges* from the stored tokens evicts the
  sticky seq (KV cleared, seq returned to idle) and falls back to
  the normal cold/warm path — useful when a user starts a fresh
  conversation while a prior one was still active.
- The sticky map is bounded by `n_seq_max`. Sessions that overrun
  the pool will see `{error, sticky_busy}` until older sessions
  end (or you can pre-`end_session` LRU ones from your own
  scheduler).

## 14. Multi-tenant concurrent decoding (`n_seq_max`)

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

## 15. Lock-free observability for routers

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

## 16. Per-request cache delta (`cache_creation_input_tokens` / `cache_read_input_tokens`)

Every result and `Stats` map carries `cache_delta => #{read := N,
created := N}`. `read` counts tokens served from the warm prefix
at admission, `created` counts tokens this request added to the
cache beyond that prefix. Both default to `0`; together they
populate Anthropic's `cache_read_input_tokens` and
`cache_creation_input_tokens` accurately.

```erlang
{ok, #{cache_delta := #{read := R, created := C},
       cache_hit_kind := Hit}} =
    erllama:complete(ModelId, Prompt, #{response_tokens => 64}),
io:format("hit=~p read=~p created=~p~n", [Hit, R, C]).
```

What each combination tells you:

| `cache_hit_kind` | `read`         | `created`        |
| ---------------- | -------------- | ---------------- |
| `cold`           | `0`            | `prompt + generated` |
| `partial`        | warm prefix    | `committed - warm prefix` |
| `exact`          | full prompt    | `generated` |
| any, save below `min_tokens` | warm prefix | `0` (no save fired) |

Streaming consumers read the same map from the final
`{erllama_done, Ref, Stats}` message; `prefill_only/2` exposes it
on its result map too.

## 17. Inspecting cache state

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

## 18. Memory-pressure-driven eviction (in `sys.config`)

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

## 19. Cache-only tests (no model required)

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

## 20. End-to-end against a real GGUF

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.Q4_K_M.gguf \
    rebar3 ct --suite=test/erllama_real_model_SUITE
```

The 6-case suite covers cold prefill, warm restore, multi-turn
parent-key resume, longest-prefix walk, eviction, and a multi-model
concurrent run. Without the env var it skips so default
`rebar3 ct` stays green.

## 21. Microbench: cold vs. warm

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
- [Tool calls](tool-calls.md) — marker configuration, deterministic
  syntax, and exact byte replay across turns.
- [Configuration](configuration.md) — full `sys.config` reference.
- [Building](building.md) — platform-specific build notes.
