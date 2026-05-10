%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model).
-moduledoc """
Per-model gen_statem that drives the request flow and wires the
cache subsystem into the model lifecycle.

State machine (v0.1):

```
  idle ──complete──▶ prefilling ──prefill_done──▶ generating ──finish──▶ idle
```

On the `prefilling → generating` transition the model fires a
**cold** save (boundary-trimmed prefix, async). Inside `generating`
it fires a **finish** save (full live token list, async) just
before returning to `idle`.

The `continued` save reason (every N tokens of new generation),
the `evict` save reason (driven by an external scheduler), and the
`shutdown` save reason (driven by `application:prep_stop`) are
defined in `erllama_cache_policy` but not yet wired here; they
land in follow-up steps.

Model operations (tokenize, prefill, decode, kv_pack, kv_unpack)
are stubbed — the gen_statem's `tokens` field IS the "context".
When step 2b lands the real `erllama_nif` for llama.cpp, those
stubs get replaced; the cache integration is unaffected.
""".
-behaviour(gen_statem).

-include("erllama_cache.hrl").

-export([
    start_link/2,
    stop/1,
    complete/2,
    complete/3,
    infer/4,
    cancel/1,
    status/1,
    evict/1,
    shutdown/1,
    model_info/1,
    tokenize/2,
    detokenize/2,
    apply_chat_template/2,
    embed/2
]).

-export_type([model/0, model_info/0, stats/0, cache_hit_kind/0,
              finish_reason/0, infer_params/0]).

-type model() :: erllama_registry:model_id() | pid().
-type model_info() :: #{
    id            := binary(),
    pid           := pid(),
    status        := idle | prefilling | generating,
    backend       := module(),
    context_size  := non_neg_integer(),
    quant_type    := atom(),
    quant_bits    := non_neg_integer(),
    tier          := disk | ram_file,
    fingerprint   := binary()
}.

-type cache_hit_kind() :: exact | partial | cold.
-type finish_reason() :: stop | length | cancelled.
-type stats() :: #{
    prompt_tokens     := non_neg_integer(),
    completion_tokens := non_neg_integer(),
    prefill_ms        := non_neg_integer(),
    generation_ms     := non_neg_integer(),
    cache_hit_kind    := cache_hit_kind(),
    finish_reason     := finish_reason(),
    cancelled         := boolean()
}.

%% Optional fields the caller may set on `infer/4`. `parent_key` and
%% `response_tokens` mirror the existing `complete/3` opts; the rest
%% are placeholders for sampling settings that bucket C will wire to
%% the llama.cpp sampler.
-type infer_params() :: #{
    response_tokens => pos_integer(),
    parent_key      => term(),
    temperature     => float(),
    top_p           => float(),
    top_k           => pos_integer(),
    min_p           => float(),
    seed            => integer(),
    stop            => [binary()],
    grammar         => binary(),
    _               => _
}.

-export([
    init/1,
    callback_mode/0,
    terminate/3
]).

%% State callbacks
-export([idle/3, prefilling/3, generating/3]).

-record(data, {
    model_id :: binary(),
    tier_srv :: atom(),
    tier :: disk | ram_file,
    fingerprint :: <<_:256>>,
    fingerprint_mode :: safe | gguf_chunked | fast_unsafe,
    quant_type :: erllama_cache_key:quant_type(),
    quant_bits :: non_neg_integer(),
    ctx_params_hash :: <<_:256>>,
    context_size :: non_neg_integer(),
    policy :: erllama_cache_policy:config(),
    %% Inference backend: erllama_model_stub | erllama_model_llama.
    backend :: module(),
    backend_state :: term(),
    %% Per-request fields
    caller :: gen_statem:from() | undefined,
    prompt_tokens :: [non_neg_integer()],
    %% Tokens currently in the "context".
    context_tokens :: [non_neg_integer()],
    response_target :: non_neg_integer(),
    generated :: [non_neg_integer()],
    last_save_at :: non_neg_integer(),
    %% Streaming-mode fields (set by infer/4, unset by complete/2,3).
    mode = standard :: standard | streaming,
    caller_pid :: pid() | undefined,
    request_ref :: reference() | undefined,
    cancel_pending = false :: boolean(),
    prefill_started_at :: integer() | undefined,
    generation_started_at :: integer() | undefined,
    cache_hit_kind = cold :: cache_hit_kind()
}).

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link(binary(), map()) -> {ok, pid()} | {error, term()}.
start_link(ModelId, Config) when is_binary(ModelId) ->
    gen_statem:start_link(
        {via, erllama_registry, ModelId}, ?MODULE, [ModelId, Config], []
    ).

-spec stop(model()) -> ok.
stop(Model) ->
    gen_statem:stop(via(Model)).

-spec complete(model(), binary()) ->
    {ok, binary(), [non_neg_integer()]} | {error, term()}.
complete(Model, Prompt) ->
    complete(Model, Prompt, #{}).

-spec complete(model(), binary(), map()) ->
    {ok, binary(), [non_neg_integer()]} | {error, term()}.
complete(Model, Prompt, Opts) ->
    gen_statem:call(via(Model), {complete, Prompt, Opts}, infinity).

-doc """
Streaming inference. Admits a request and immediately returns a
unique `reference()`; tokens are delivered to `CallerPid` via
asynchronous messages:

- `{erllama_token, Ref, binary()}` per generated token (text fragment)
- `{erllama_done, Ref, stats()}` on normal completion
- `{erllama_error, Ref, term()}` on failure

`Tokens` is the prompt as a list of token ids - tokenisation is the
caller's responsibility (use `tokenize/2` or apply a chat template
first). `Params` is an `infer_params()` map; only `response_tokens`
and `parent_key` are honoured in v0.1, the rest are reserved for
bucket C.

Returns `{error, busy}` if the model is already serving another
request.
""".
-spec infer(model(), [non_neg_integer()], infer_params(), pid()) ->
    {ok, reference()} | {error, term()}.
infer(Model, Tokens, Params, CallerPid)
  when is_list(Tokens), is_map(Params), is_pid(CallerPid) ->
    gen_statem:call(via(Model), {infer, Tokens, Params, CallerPid}, infinity).

-doc """
Cancel an in-flight streaming inference. Idempotent and fire-and-
forget: returns `ok` even if the ref is unknown (already finished or
never existed). The cancellation is observed at the next
inter-token boundary; the model emits a final `{erllama_done, Ref,
Stats}` with `cancelled => true` after the running decode step
completes.
""".
-spec cancel(reference()) -> ok.
cancel(Ref) when is_reference(Ref) ->
    case erllama_inflight:lookup(Ref) of
        {ok, ModelPid} ->
            gen_statem:cast(ModelPid, {cancel, Ref}),
            ok;
        {error, not_found} ->
            ok
    end.

-spec status(model()) -> idle | prefilling | generating.
status(Model) ->
    gen_statem:call(via(Model), status).

-doc """
Request that the model evict its current state. Fires an `evict`
save synchronously if there is anything in the context. Called by
`erllama_scheduler` (future) when GPU memory pressure requires this
model to release its context handle. No-op when the model is idle
with no live context.
""".
-spec evict(model()) -> ok.
evict(Model) ->
    gen_statem:call(via(Model), evict).

-doc """
Fire a `shutdown` save synchronously and return. Called from the
application's `prep_stop` hook so live state survives a graceful
restart.
""".
-spec shutdown(model()) -> ok.
shutdown(Model) ->
    gen_statem:call(via(Model), shutdown).

-doc """
Snapshot of the model's metadata.

Returns a `model_info()` map with status, context size, quantisation,
backend, fingerprint, and tier. Safe to call from any state - the
gen_statem handles it as a common event without disrupting in-flight
inference.
""".
-spec model_info(model()) -> model_info().
model_info(Model) ->
    gen_statem:call(via(Model), model_info).

-doc """
Tokenise a string using the model's tokenizer. Returns a list of
token IDs. Safe to call concurrently with `complete/2,3`; tokenisation
runs against the model's static vocabulary, not the live KV cache.
""".
-spec tokenize(model(), binary()) ->
    {ok, [non_neg_integer()]} | {error, term()}.
tokenize(Model, Text) when is_binary(Text) ->
    gen_statem:call(via(Model), {tokenize, Text}).

-doc """
Detokenise a list of token IDs back to a string. Safe to call
concurrently with `complete/2,3`.
""".
-spec detokenize(model(), [non_neg_integer()]) ->
    {ok, binary()} | {error, term()}.
detokenize(Model, Tokens) when is_list(Tokens) ->
    gen_statem:call(via(Model), {detokenize, Tokens}).

-doc """
Render a normalised chat request through the model's chat template
and tokenise in one step. The Request map carries `messages`,
`system`, and `tools`; the per-model template decides where each
field lands in the prompt.

Returns `{error, not_supported}` if the backend does not implement
chat templating.
""".
-spec apply_chat_template(model(), erllama_model_backend:chat_request()) ->
    {ok, [non_neg_integer()]} | {error, term()}.
apply_chat_template(Model, Request) when is_map(Request) ->
    gen_statem:call(via(Model), {apply_chat_template, Request}).

-doc """
Compute an embedding vector for the given prompt tokens.
""".
-spec embed(model(), [non_neg_integer()]) ->
    {ok, [float()]} | {error, term()}.
embed(Model, Tokens) when is_list(Tokens) ->
    gen_statem:call(via(Model), {embed, Tokens}).

%% =============================================================================
%% gen_statem callbacks
%% =============================================================================

callback_mode() -> state_functions.

init([ModelId, Config]) ->
    Backend = maps:get(backend, Config, erllama_model_stub),
    case Backend:init(Config) of
        {ok, BState} ->
            Data = #data{
                model_id = ModelId,
                tier_srv = maps:get(tier_srv, Config),
                tier = maps:get(tier, Config),
                fingerprint = maps:get(fingerprint, Config),
                fingerprint_mode = maps:get(fingerprint_mode, Config, safe),
                quant_type = maps:get(quant_type, Config, f16),
                quant_bits = maps:get(quant_bits, Config, 16),
                ctx_params_hash = maps:get(ctx_params_hash, Config),
                context_size = maps:get(context_size, Config, 4096),
                policy = maps:get(policy, Config),
                backend = Backend,
                backend_state = BState,
                prompt_tokens = [],
                context_tokens = [],
                response_target = 0,
                generated = [],
                last_save_at = 0
            },
            {ok, idle, Data};
        {error, Reason} ->
            {stop, Reason}
    end.

terminate(_Reason, _State, #data{backend = B, backend_state = S}) ->
    B:terminate(S),
    ok;
terminate(_Reason, _State, _Data) ->
    ok.

%% =============================================================================
%% State: idle
%% =============================================================================

idle({call, From}, {complete, Prompt, Opts}, Data) ->
    PromptTokens = backend_call(Data, tokenize, [Prompt]),
    ResponseTarget = maps:get(response_tokens, Opts, 4),
    ParentKey = maps:get(parent_key, Opts, undefined),
    Data1 = Data#data{
        mode = standard,
        caller = From,
        caller_pid = undefined,
        request_ref = undefined,
        cancel_pending = false,
        prompt_tokens = PromptTokens,
        response_target = ResponseTarget,
        generated = [],
        prefill_started_at = erlang:monotonic_time(millisecond)
    },
    enter_after_lookup(ParentKey, Data1);
idle({call, From}, {infer, Tokens, Params, CallerPid}, Data) ->
    Ref = make_ref(),
    ResponseTarget = maps:get(response_tokens, Params, 64),
    ParentKey = maps:get(parent_key, Params, undefined),
    Grammar = maps:get(grammar, Params, undefined),
    case set_grammar(Grammar, Data) of
        {ok, Data0} ->
            ok = erllama_inflight:register(Ref, self()),
            Data1 = Data0#data{
                mode = streaming,
                caller = undefined,
                caller_pid = CallerPid,
                request_ref = Ref,
                cancel_pending = false,
                prompt_tokens = Tokens,
                response_target = ResponseTarget,
                generated = [],
                prefill_started_at = erlang:monotonic_time(millisecond)
            },
            %% Reply with the ref before kicking off prefill so the caller is
            %% guaranteed to have it before any erllama_token messages land.
            add_reply_action(From, {ok, Ref}, enter_after_lookup(ParentKey, Data1));
        {error, _} = E ->
            {keep_state, Data, [{reply, From, E}]}
    end;
idle({call, From}, status, Data) ->
    {keep_state, Data, [{reply, From, idle}]};
idle(EventType, EventContent, Data) ->
    handle_common(idle, EventType, EventContent, Data).

%% Tack a {reply, From, Reply} action onto a gen_statem transition
%% returned by enter_after_lookup, regardless of whether the original
%% transition carried an actions list or not.
add_reply_action(From, Reply, {next_state, NextState, NewData}) ->
    {next_state, NextState, NewData, [{reply, From, Reply}]};
add_reply_action(From, Reply, {next_state, NextState, NewData, Actions}) ->
    {next_state, NextState, NewData, [{reply, From, Reply} | Actions]};
add_reply_action(From, Reply, {keep_state, NewData}) ->
    {keep_state, NewData, [{reply, From, Reply}]};
add_reply_action(From, Reply, {keep_state, NewData, Actions}) ->
    {keep_state, NewData, [{reply, From, Reply} | Actions]}.

%% Branches the lookup result into the correct gen_statem transition.
%% Used by both complete/2,3 and infer/4 paths.
enter_after_lookup(ParentKey, Data) ->
    case lookup_or_resume(Data#data.prompt_tokens, ParentKey, Data) of
        {warm, ContextTokens, RemainingTokens, HitKind} ->
            ok = prime_logits(ContextTokens, RemainingTokens, Data),
            Data1 = Data#data{
                context_tokens = ContextTokens ++ RemainingTokens,
                cache_hit_kind = HitKind
            },
            enter_generating(Data1);
        cold ->
            Data1 = Data#data{cache_hit_kind = cold},
            enter_prefilling(Data1)
    end.

%% =============================================================================
%% State: prefilling
%% =============================================================================

prefilling({call, From}, status, Data) ->
    {keep_state, Data, [{reply, From, prefilling}]};
prefilling(EventType, EventContent, Data) ->
    handle_common(prefilling, EventType, EventContent, Data).

%% =============================================================================
%% State: generating
%% =============================================================================

generating({call, From}, status, Data) ->
    {keep_state, Data, [{reply, From, generating}]};
generating(cast, decode_step, Data) ->
    decode_step(Data);
generating(EventType, EventContent, Data) ->
    handle_common(generating, EventType, EventContent, Data).

%% =============================================================================
%% Common event handler
%% =============================================================================

handle_common(_State, {call, From}, {complete, _, _}, Data) ->
    %% Reject concurrent complete calls; only one in flight.
    %% In idle this clause is unreachable: idle/3 matches the event
    %% first. From prefilling/generating it surfaces here.
    {keep_state, Data, [{reply, From, {error, busy}}]};
handle_common(_State, {call, From}, {infer, _, _, _}, Data) ->
    %% Reject concurrent infer calls; only one in flight per model.
    {keep_state, Data, [{reply, From, {error, busy}}]};
handle_common(_State, cast, {cancel, Ref}, Data = #data{request_ref = Ref}) ->
    %% Set the cancel flag; the next decode_step honours it.
    {keep_state, Data#data{cancel_pending = true}};
handle_common(_State, cast, {cancel, _OtherRef}, Data) ->
    %% Stale cancel for a previous request. Ignore.
    {keep_state, Data};
handle_common(_State, {call, From}, evict, Data) ->
    {keep_state, fire_save_for_reason(evict, Data), [{reply, From, ok}]};
handle_common(_State, {call, From}, shutdown, Data) ->
    {keep_state, fire_save_for_reason(shutdown, Data), [{reply, From, ok}]};
handle_common(State, {call, From}, model_info, Data) ->
    Reply = build_model_info(State, Data),
    {keep_state, Data, [{reply, From, Reply}]};
handle_common(_State, {call, From}, {tokenize, Text}, Data) ->
    Reply = case backend_call(Data, tokenize, [Text]) of
                {error, _} = E -> E;
                Tokens when is_list(Tokens) -> {ok, Tokens}
            end,
    {keep_state, Data, [{reply, From, Reply}]};
handle_common(_State, {call, From}, {detokenize, Tokens}, Data) ->
    Reply = case backend_call(Data, detokenize, [Tokens]) of
                {error, _} = E -> E;
                Bin when is_binary(Bin) -> {ok, Bin}
            end,
    {keep_state, Data, [{reply, From, Reply}]};
handle_common(_State, {call, From}, {apply_chat_template, Request}, Data) ->
    Reply = optional_backend_call(Data, apply_chat_template, [Request]),
    {keep_state, Data, [{reply, From, Reply}]};
handle_common(_State, {call, From}, {embed, Tokens}, Data) ->
    Reply = optional_backend_call(Data, embed, [Tokens]),
    {keep_state, Data, [{reply, From, Reply}]};
handle_common(_State, _EventType, _EventContent, Data) ->
    {keep_state, Data}.

%% Like backend_call/3, but for callbacks declared optional in the
%% behaviour. If the backend module does not export the function
%% returns `{error, not_supported}` instead of crashing.
optional_backend_call(#data{backend = Mod, backend_state = S}, Fn, Args) ->
    Arity = length(Args) + 1,
    case erlang:function_exported(Mod, Fn, Arity) of
        true  -> apply(Mod, Fn, [S | Args]);
        false -> {error, not_supported}
    end.

build_model_info(State, Data) ->
    #{
        id           => Data#data.model_id,
        pid          => self(),
        status       => State,
        backend      => Data#data.backend,
        context_size => Data#data.context_size,
        quant_type   => Data#data.quant_type,
        quant_bits   => Data#data.quant_bits,
        tier         => Data#data.tier,
        fingerprint  => Data#data.fingerprint
    }.

%% =============================================================================
%% Internal: state transitions
%% =============================================================================

enter_prefilling(Data) ->
    Tokens = Data#data.prompt_tokens,
    case erllama_cache_policy:cold_save_split(Tokens, Data#data.policy) of
        {trim, TrimmedPrefix, RemainingTokens} ->
            ok = backend_call(Data, prefill, [TrimmedPrefix]),
            ok = fire_cold_save(TrimmedPrefix, Data),
            ok = backend_call(Data, prefill, [RemainingTokens]);
        no_save ->
            ok = backend_call(Data, prefill, [Tokens])
    end,
    Data1 = Data#data{context_tokens = Tokens},
    enter_generating(Data1).

enter_generating(Data) ->
    %% Decode token-by-token via cast-to-self events so continued
    %% saves can fire mid-stream AND external events (cancel, busy
    %% reject, evict, status) get a fair turn between tokens. We
    %% deliberately avoid `next_event` here because next_event has
    %% higher priority than mailbox messages, which would starve
    %% cancel.
    Data1 = Data#data{
        last_save_at = length(Data#data.context_tokens),
        generation_started_at = erlang:monotonic_time(millisecond)
    },
    gen_statem:cast(self(), decode_step),
    {next_state, generating, Data1}.

decode_step(Data = #data{cancel_pending = true}) ->
    %% Honoured between tokens. The current decode step has already
    %% returned; we finish with cancelled => true and never call
    %% decode_one again.
    finish_request_cancelled(Data);
decode_step(Data) ->
    case length(Data#data.generated) >= Data#data.response_target of
        true ->
            finish_request(Data, length);
        false ->
            case backend_call(Data, decode_one, [Data#data.context_tokens]) of
                {ok, Token} ->
                    advance_with(Token, Data);
                {eog, Token} ->
                    Data1 = append_token(Token, Data),
                    Data2 = stream_emit(Token, Data1),
                    finish_request(Data2, stop);
                {error, _} = E ->
                    Data1 = reset(Data),
                    case Data#data.mode of
                        standard ->
                            {next_state, idle, Data1, [{reply, Data#data.caller, E}]};
                        streaming ->
                            send_error(Data, E),
                            {next_state, idle, Data1}
                    end
            end
    end.

advance_with(Token, Data) ->
    Data1 = append_token(Token, Data),
    Data2 = stream_emit(Token, Data1),
    Data3 = maybe_fire_continued(Data2),
    gen_statem:cast(self(), decode_step),
    {keep_state, Data3}.

%% In streaming mode, send the just-appended token to the caller as a
%% text fragment. In standard mode this is a no-op; the full reply is
%% built in finish_request via detokenize on the whole `generated`
%% list.
stream_emit(Token, Data = #data{mode = streaming, caller_pid = Pid,
                                request_ref = Ref}) ->
    case backend_call(Data, detokenize, [[Token]]) of
        Bin when is_binary(Bin), Bin =/= <<>> ->
            Pid ! {erllama_token, Ref, Bin};
        _ ->
            ok
    end,
    Data;
stream_emit(_Token, Data = #data{mode = standard}) ->
    Data.

append_token(Token, Data) ->
    Data#data{
        context_tokens = Data#data.context_tokens ++ [Token],
        generated = Data#data.generated ++ [Token]
    }.

reset(Data) ->
    Data#data{
        mode = standard,
        caller = undefined,
        caller_pid = undefined,
        request_ref = undefined,
        cancel_pending = false,
        prefill_started_at = undefined,
        generation_started_at = undefined,
        cache_hit_kind = cold,
        context_tokens = [],
        prompt_tokens = [],
        generated = [],
        response_target = 0,
        last_save_at = 0
    }.

maybe_fire_continued(Data) ->
    LiveCount = length(Data#data.context_tokens),
    Should = erllama_cache_policy:should_continued_save(
        LiveCount, Data#data.last_save_at, Data#data.policy
    ),
    case Should of
        true ->
            ok = fire_save_if(true, continued, Data#data.context_tokens, Data),
            Data#data{last_save_at = LiveCount};
        false ->
            Data
    end.

finish_request(Data, FinishReason) ->
    ok = fire_finish_save(Data#data.context_tokens, Data),
    {ok, Data1} = clear_grammar(Data),
    case Data1#data.mode of
        standard ->
            Reply = backend_call(Data1, detokenize, [Data1#data.generated]),
            From = Data1#data.caller,
            Generated = Data1#data.generated,
            {next_state, idle, reset(Data1), [{reply, From, {ok, Reply, Generated}}]};
        streaming ->
            Stats = build_stats(FinishReason, false, Data1),
            send_done(Data1, Stats),
            {next_state, idle, reset(Data1)}
    end.

finish_request_cancelled(Data) ->
    %% Cancelled requests still fire a finish save: whatever live
    %% context exists is worth keeping for resume.
    ok = fire_finish_save(Data#data.context_tokens, Data),
    {ok, Data1} = clear_grammar(Data),
    Stats = build_stats(cancelled, true, Data1),
    send_done(Data1, Stats),
    {next_state, idle, reset(Data1)}.

send_done(#data{mode = streaming, caller_pid = Pid, request_ref = Ref}, Stats) ->
    erllama_inflight:unregister(Ref),
    Pid ! {erllama_done, Ref, Stats},
    ok;
send_done(_, _) ->
    ok.

send_error(#data{mode = streaming, caller_pid = Pid, request_ref = Ref}, Reason) ->
    erllama_inflight:unregister(Ref),
    Pid ! {erllama_error, Ref, Reason},
    ok;
send_error(_, _) ->
    ok.

%% Optional grammar setup. If the backend doesn't export set_grammar
%% the caller's grammar (if any) is silently ignored. Returns the
%% (possibly updated) Data on success.
set_grammar(undefined, Data) ->
    clear_grammar(Data);
set_grammar(Grammar, Data = #data{backend = Mod, backend_state = S}) when is_binary(Grammar) ->
    case erlang:function_exported(Mod, set_grammar, 2) of
        true ->
            case Mod:set_grammar(S, Grammar) of
                {ok, S1}    -> {ok, Data#data{backend_state = S1}};
                {error, _} = E -> E
            end;
        false ->
            {ok, Data}
    end.

clear_grammar(Data = #data{backend = Mod, backend_state = S}) ->
    case erlang:function_exported(Mod, clear_sampler, 1) of
        true ->
            case Mod:clear_sampler(S) of
                {ok, S1}    -> {ok, Data#data{backend_state = S1}};
                {error, _} = E -> E
            end;
        false ->
            {ok, Data}
    end.

build_stats(FinishReason, Cancelled, Data) ->
    Now = erlang:monotonic_time(millisecond),
    PrefillStart = Data#data.prefill_started_at,
    GenStart = Data#data.generation_started_at,
    PrefillMs = case {PrefillStart, GenStart} of
                    {undefined, _} -> 0;
                    {_, undefined} -> max(0, Now - PrefillStart);
                    _              -> max(0, GenStart - PrefillStart)
                end,
    GenMs = case GenStart of
                undefined -> 0;
                _         -> max(0, Now - GenStart)
            end,
    #{
        prompt_tokens     => length(Data#data.prompt_tokens),
        completion_tokens => length(Data#data.generated),
        prefill_ms        => PrefillMs,
        generation_ms     => GenMs,
        cache_hit_kind    => Data#data.cache_hit_kind,
        finish_reason     => FinishReason,
        cancelled         => Cancelled
    }.

%% =============================================================================
%% Internal: cache integration
%% =============================================================================

lookup_or_resume(PromptTokens, ParentKey, Data) ->
    %% Exact-key fast path. checkout pins the row so eviction can't
    %% delete the underlying file/slab between the lookup and the
    %% load. The pin is released right after kv_unpack copies the
    %% bytes into our live context (claim → unpack → checkin), so
    %% the slab stays evictable while the user reads the streamed
    %% response.
    Key = make_key(PromptTokens, Data),
    case pin_and_load(Key, Data) of
        {ok, ContextTokens} ->
            erllama_cache_counters:incr(?C_HITS_EXACT),
            {warm, ContextTokens, [], exact};
        miss when ParentKey =/= undefined ->
            try_session_resume(PromptTokens, ParentKey, Data);
        miss ->
            try_longest_prefix(PromptTokens, Data)
    end.

%% Stateless callers (HTTP front-end, agent loops that resend the
%% full conversation each turn) don't have a parent_key to thread.
%% Walk back through the prompt by stride and pick the longest
%% cached prefix; fall through to cold if nothing matches.
try_longest_prefix(PromptTokens, Data) ->
    KeyMeta = #{
        fingerprint => Data#data.fingerprint,
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash
    },
    Stride = maps:get(boundary_align_tokens, Data#data.policy, 2048),
    Min = maps:get(min_tokens, Data#data.policy, 512),
    case erllama_cache_meta_srv:lookup_longest_prefix(KeyMeta, PromptTokens, Stride, Min) of
        {ok, PrefixLen, Row} ->
            resume_at_prefix(element(?POS_KEY, Row), PrefixLen, PromptTokens, Data, partial);
        miss ->
            erllama_cache_counters:incr(?C_MISSES),
            cold
    end.

%% Pin + load the row, then verify the tokens really are the first
%% PrefixLen of PromptTokens. The key is sha256 of the tokens, so a
%% hit implies equality, but we belt-and-braces it here.
resume_at_prefix(Key, PrefixLen, PromptTokens, Data, HitKind) ->
    case pin_and_load(Key, Data) of
        {ok, ParentTokens} when length(ParentTokens) =:= PrefixLen ->
            case is_strict_prefix(ParentTokens, PromptTokens) of
                true ->
                    Remaining = lists:nthtail(PrefixLen, PromptTokens),
                    erllama_cache_counters:incr(?C_HITS_LONGEST_PREFIX),
                    {warm, ParentTokens, Remaining, HitKind};
                false ->
                    erllama_cache_counters:incr(?C_MISSES),
                    cold
            end;
        _ ->
            erllama_cache_counters:incr(?C_MISSES),
            cold
    end.

%% Note: the resume hit counter is bumped inside `try_session_resume`
%% only on a verified strict-prefix match.

try_session_resume(PromptTokens, ParentKey, Data) ->
    Wait = maps:get(session_resume_wait_ms, Data#data.policy, 500),
    %% First wait for the row to publish (the previous turn's
    %% finish-save may still be in flight), then pin via checkout.
    case erllama_cache_meta_srv:lookup_exact_or_wait(ParentKey, Wait) of
        {ok, _Row} ->
            case pin_and_load(ParentKey, Data) of
                {ok, ParentTokens} ->
                    case is_strict_prefix(ParentTokens, PromptTokens) of
                        true ->
                            Remaining = lists:nthtail(length(ParentTokens), PromptTokens),
                            erllama_cache_counters:incr(?C_HITS_RESUME),
                            {warm, ParentTokens, Remaining, partial};
                        false ->
                            erllama_cache_counters:incr(?C_MISSES),
                            cold
                    end;
                miss ->
                    erllama_cache_counters:incr(?C_MISSES),
                    cold
            end;
        miss ->
            erllama_cache_counters:incr(?C_MISSES),
            cold
    end.

%% checkout the row, load + unpack the payload under the pin, then
%% checkin. Returns the row's stored token list on success or `miss`
%% if the row was evicted between the prior lookup and our checkout.
%% Eviction never selects a refcount > 0 row, so the load itself is
%% safe; the only failure mode is the row already being gone.
pin_and_load(Key, Data) ->
    T0 = erlang:monotonic_time(nanosecond),
    Result =
        case erllama_cache_meta_srv:checkout(Key, self()) of
            {ok, HolderRef, Tier, Loc, _Header, TokensBin} ->
                Bin = load_payload(Tier, Loc, Key, Data),
                case Bin of
                    <<>> ->
                        ok = erllama_cache_meta_srv:checkin(HolderRef),
                        miss;
                    _ ->
                        ok = backend_call(Data, kv_unpack, [Bin]),
                        ok = erllama_cache_meta_srv:checkin(HolderRef),
                        Tokens =
                            case TokensBin of
                                undefined -> [];
                                _ -> erllama_cache_key:decode_tokens(TokensBin)
                            end,
                        {ok, Tokens}
                end;
            {error, busy} ->
                miss;
            miss ->
                miss
        end,
    Elapsed = erlang:monotonic_time(nanosecond) - T0,
    erllama_cache_counters:add(?C_LOAD_TOTAL_NS, max(Elapsed, 0)),
    Result.

load_payload(ram, _Loc, Key, _Data) ->
    case erllama_cache_ram:load(Key) of
        {ok, B} -> B;
        miss -> <<>>
    end;
load_payload(_Tier, _Loc, Key, Data) ->
    case erllama_cache_disk_srv:load(Data#data.tier_srv, Key) of
        {ok, _Info, Payload} -> Payload;
        _ -> <<>>
    end.

fire_cold_save(TrimmedPrefix, Data) ->
    Min = maps:get(min_tokens, Data#data.policy),
    fire_save_if(length(TrimmedPrefix) >= Min, cold, TrimmedPrefix, Data).

fire_finish_save(LiveTokens, Data) ->
    Should = erllama_cache_policy:should_finish_save(
        length(LiveTokens), Data#data.policy
    ),
    fire_save_if(Should, finish, LiveTokens, Data).

fire_save_if(false, _Reason, _Tokens, _Data) ->
    ok;
fire_save_if(true, Reason, Tokens, Data) ->
    BuildMeta = build_meta_for(Reason, Tokens, Data),
    T0 = erlang:monotonic_time(nanosecond),
    Payload = backend_call(Data, kv_pack, [Tokens]),
    Elapsed = erlang:monotonic_time(nanosecond) - T0,
    erllama_cache_counters:add(?C_PACK_TOTAL_NS, max(Elapsed, 0)),
    _ = erllama_cache_writer:save(
        Data#data.tier_srv, Data#data.tier, BuildMeta, Payload, 0
    ),
    ok.

%% Evict and shutdown saves: fire unconditionally if there is any
%% live context, regardless of `min_tokens`. The plan's policy
%% module gates only cold/continued/finish; evict and shutdown are
%% emergency saves that capture whatever state exists. Update
%% `last_save_at` so a follow-up continued save inside the same
%% generation does not double-save the same tokens.
fire_save_for_reason(_Reason, #data{context_tokens = []} = Data) ->
    Data;
fire_save_for_reason(Reason, Data) ->
    Tokens = Data#data.context_tokens,
    fire_save_if(true, Reason, Tokens, Data),
    Data#data{last_save_at = length(Tokens)}.

build_meta_for(SaveReason, Tokens, Data) ->
    #{
        save_reason => SaveReason,
        quant_bits => Data#data.quant_bits,
        fingerprint => Data#data.fingerprint,
        fingerprint_mode => Data#data.fingerprint_mode,
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash,
        tokens => Tokens,
        context_size => Data#data.context_size,
        prompt_text => <<>>
    }.

make_key(Tokens, Data) ->
    erllama_cache_key:make(#{
        fingerprint => Data#data.fingerprint,
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash,
        tokens => Tokens
    }).

is_strict_prefix([], _) -> true;
is_strict_prefix([H | T1], [H | T2]) -> is_strict_prefix(T1, T2);
is_strict_prefix(_, _) -> false.

%% =============================================================================
%% Internal: backend dispatch
%% =============================================================================

backend_call(#data{backend = Mod, backend_state = S}, Fn, Args) ->
    apply(Mod, Fn, [S | Args]).

%% Resolve a model() reference to a Pid that gen_statem:call/2,3
%% accepts. Binary IDs go through the registry; pids pass through.
%% Crashes with `{noproc, {erllama_model, not_found, ModelId}}` if
%% the model is not registered, so callers do not have to special-
%% case that path - they get a useful error tag rather than a bare
%% `noproc`.
via(Pid) when is_pid(Pid) ->
    Pid;
via(ModelId) when is_binary(ModelId) ->
    case erllama_registry:whereis_name(ModelId) of
        Pid when is_pid(Pid) -> Pid;
        undefined            -> exit({noproc, {?MODULE, not_found, ModelId}})
    end.

%% After kv_unpack, the per-context logits buffer is stale. To regenerate
%% it the model layer drops the last cell of the restored sequence and
%% prefills the corresponding token. If the warm hit also has remaining
%% tokens (parent_key resume), they are prefilled in the same single
%% prefill call: the last "context" token gets popped, prepended to the
%% remaining list, and the whole batch goes through one llama_decode.
prime_logits([], Remaining, Data) ->
    backend_call(Data, prefill, [Remaining]);
prime_logits(ContextTokens, Remaining, #data{backend = Mod} = Data) ->
    case erlang:function_exported(Mod, seq_rm_last, 2) of
        true ->
            N = length(ContextTokens),
            Last = lists:last(ContextTokens),
            ok = backend_call(Data, seq_rm_last, [N]),
            backend_call(Data, prefill, [[Last | Remaining]]);
        false ->
            %% Backends without a real KV cache (e.g. the stub) just
            %% prefill any remaining tokens; the saved "state" carries
            %% no logits anyway.
            backend_call(Data, prefill, [Remaining])
    end.
