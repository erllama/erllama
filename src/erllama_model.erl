%% @doc
%% Per-model gen_statem that drives the request flow and wires the
%% cache subsystem into the model lifecycle.
%%
%% State machine (v0.1):
%%
%% ```
%%   idle ──complete──▶ prefilling ──prefill_done──▶ generating ──finish──▶ idle
%% ```
%%
%% On the `prefilling → generating` transition the model fires a
%% **cold** save (boundary-trimmed prefix, async). Inside `generating`
%% it fires a **finish** save (full live token list, async) just
%% before returning to `idle`.
%%
%% The `continued` save reason (every N tokens of new generation),
%% the `evict` save reason (driven by an external scheduler), and the
%% `shutdown` save reason (driven by `application:prep_stop`) are
%% defined in `erllama_cache_policy` but not yet wired here; they
%% land in follow-up steps.
%%
%% Model operations (tokenize, prefill, decode, kv_pack, kv_unpack)
%% are stubbed — the gen_statem's `tokens` field IS the "context".
%% When step 2b lands the real `erllama_nif` for llama.cpp, those
%% stubs get replaced; the cache integration is unaffected.
%% @end
-module(erllama_model).
-behaviour(gen_statem).

-include("erllama_cache.hrl").

-export([
    start_link/2,
    stop/1,
    complete/2,
    complete/3,
    status/1,
    evict/1,
    shutdown/1
]).

-export([
    init/1,
    callback_mode/0,
    terminate/3
]).

%% State callbacks
-export([idle/3, prefilling/3, generating/3]).

-record(data, {
    model_id :: atom(),
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
    response_target :: pos_integer(),
    generated :: [non_neg_integer()],
    last_save_at :: non_neg_integer()
}).

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link(atom(), map()) -> {ok, pid()} | {error, term()}.
start_link(ModelId, Config) ->
    gen_statem:start_link({local, ModelId}, ?MODULE, [ModelId, Config], []).

-spec stop(atom() | pid()) -> ok.
stop(Model) ->
    gen_statem:stop(Model).

-spec complete(atom() | pid(), binary()) ->
    {ok, binary(), [non_neg_integer()]} | {error, term()}.
complete(Model, Prompt) ->
    complete(Model, Prompt, #{}).

-spec complete(atom() | pid(), binary(), map()) ->
    {ok, binary(), [non_neg_integer()]} | {error, term()}.
complete(Model, Prompt, Opts) ->
    gen_statem:call(Model, {complete, Prompt, Opts}, infinity).

-spec status(atom() | pid()) -> idle | prefilling | generating.
status(Model) ->
    gen_statem:call(Model, status).

%% @doc Request that the model evict its current state. Fires an
%% `evict` save synchronously if there is anything in the context.
%% Called by `erllama_scheduler` (future) when GPU memory pressure
%% requires this model to release its context handle. No-op when
%% the model is idle with no live context.
-spec evict(atom() | pid()) -> ok.
evict(Model) ->
    gen_statem:call(Model, evict).

%% @doc Fire a `shutdown` save synchronously and return. Called from
%% the application's `prep_stop` hook so live state survives a
%% graceful restart.
-spec shutdown(atom() | pid()) -> ok.
shutdown(Model) ->
    gen_statem:call(Model, shutdown).

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
        caller = From,
        prompt_tokens = PromptTokens,
        response_target = ResponseTarget,
        generated = []
    },
    case lookup_or_resume(PromptTokens, ParentKey, Data1) of
        {warm, ContextTokens, RemainingTokens} ->
            ok = prime_logits(ContextTokens, RemainingTokens, Data1),
            Data2 = Data1#data{
                context_tokens = ContextTokens ++ RemainingTokens
            },
            enter_generating(Data2);
        cold ->
            enter_prefilling(Data1)
    end;
idle({call, From}, status, Data) ->
    {keep_state, Data, [{reply, From, idle}]};
idle(EventType, EventContent, Data) ->
    handle_common(idle, EventType, EventContent, Data).

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
generating(internal, decode_step, Data) ->
    decode_step(Data);
generating(EventType, EventContent, Data) ->
    handle_common(generating, EventType, EventContent, Data).

%% =============================================================================
%% Common event handler
%% =============================================================================

handle_common(_State, {call, From}, {complete, _, _}, Data) ->
    %% Reject concurrent complete calls; only one in flight.
    {keep_state, Data, [{reply, From, {error, busy}}]};
handle_common(_State, {call, From}, evict, Data) ->
    {keep_state, fire_save_for_reason(evict, Data), [{reply, From, ok}]};
handle_common(_State, {call, From}, shutdown, Data) ->
    {keep_state, fire_save_for_reason(shutdown, Data), [{reply, From, ok}]};
handle_common(_State, _EventType, _EventContent, Data) ->
    {keep_state, Data}.

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
    %% Decode token-by-token via internal events so continued saves
    %% can fire mid-stream. Each decode_step appends one token,
    %% checks the policy, and either continues or finishes.
    Data1 = Data#data{last_save_at = length(Data#data.context_tokens)},
    {next_state, generating, Data1, [{next_event, internal, decode_step}]}.

decode_step(Data) ->
    case length(Data#data.generated) >= Data#data.response_target of
        true ->
            finish_request(Data);
        false ->
            case backend_call(Data, decode_one, [Data#data.context_tokens]) of
                {ok, Token} ->
                    advance_with(Token, Data);
                {eog, Token} ->
                    Data1 = append_token(Token, Data),
                    finish_request(Data1);
                {error, _} = E ->
                    From = Data#data.caller,
                    Data1 = reset(Data),
                    {next_state, idle, Data1, [{reply, From, E}]}
            end
    end.

advance_with(Token, Data) ->
    Data1 = append_token(Token, Data),
    Data2 = maybe_fire_continued(Data1),
    {keep_state, Data2, [{next_event, internal, decode_step}]}.

append_token(Token, Data) ->
    Data#data{
        context_tokens = Data#data.context_tokens ++ [Token],
        generated = Data#data.generated ++ [Token]
    }.

reset(Data) ->
    Data#data{
        caller = undefined,
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

finish_request(Data) ->
    ok = fire_finish_save(Data#data.context_tokens, Data),
    Reply = backend_call(Data, detokenize, [Data#data.generated]),
    From = Data#data.caller,
    Generated = Data#data.generated,
    {next_state, idle, reset(Data), [{reply, From, {ok, Reply, Generated}}]}.

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
            {warm, ContextTokens, []};
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
            FoundKey = element(?POS_KEY, Row),
            case pin_and_load(FoundKey, Data) of
                {ok, ParentTokens} ->
                    %% Belt-and-braces: the cache key encodes the
                    %% tokens, so a hit at PrefixLen implies the
                    %% row's tokens equal the first PrefixLen of the
                    %% prompt. Verify before trusting.
                    case
                        length(ParentTokens) =:= PrefixLen andalso
                            is_strict_prefix(ParentTokens, PromptTokens)
                    of
                        true ->
                            Remaining = lists:nthtail(PrefixLen, PromptTokens),
                            erllama_cache_counters:incr(?C_HITS_RESUME),
                            {warm, ParentTokens, Remaining};
                        false ->
                            erllama_cache_counters:incr(?C_MISSES),
                            cold
                    end;
                miss ->
                    %% Race: row was evicted between the longest-prefix
                    %% scan and our checkout. Treat as cold.
                    erllama_cache_counters:incr(?C_MISSES),
                    cold
            end;
        miss ->
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
                            {warm, ParentTokens, Remaining};
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
    end.

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
    Payload = backend_call(Data, kv_pack, [Tokens]),
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
