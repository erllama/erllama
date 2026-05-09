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
    %% Per-request fields
    caller :: gen_statem:from() | undefined,
    prompt_tokens :: [non_neg_integer()],
    %% Tokens currently in the "context" (prompt prefix + generated).
    %% In v0.1 this is a plain list; the real llama_context* arrives
    %% with the kv_pack/kv_unpack NIFs.
    context_tokens :: [non_neg_integer()],
    %% Tokens that the requested response should produce. Set by
    %% complete/2 via Opts; defaults to a deterministic stub sequence.
    response_target :: pos_integer(),
    %% Generated tokens accumulated so far.
    generated :: [non_neg_integer()],
    %% Token count at the most recent save of any reason; used by the
    %% policy to decide when continued saves should fire.
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
        prompt_tokens = [],
        context_tokens = [],
        response_target = 0,
        generated = [],
        last_save_at = 0
    },
    {ok, idle, Data}.

terminate(_Reason, _State, _Data) ->
    ok.

%% =============================================================================
%% State: idle
%% =============================================================================

idle({call, From}, {complete, Prompt, Opts}, Data) ->
    PromptTokens = stub_tokenize(Prompt),
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
            %% Already-prefilled state was loaded; finish prefill on
            %% any tokens not covered, then jump to generating.
            ok = stub_prefill(RemainingTokens),
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
    %% Run "prefill" on the prompt, decide on cold save, advance to
    %% generating. In v0.1 this is synchronous; with real llama.cpp
    %% it would happen on a dirty CPU scheduler with the gen_statem
    %% receiving a `prefill_done` event.
    Tokens = Data#data.prompt_tokens,
    case erllama_cache_policy:cold_save_split(Tokens, Data#data.policy) of
        {trim, TrimmedPrefix, RemainingTokens} ->
            ok = stub_prefill(TrimmedPrefix),
            ok = fire_cold_save(TrimmedPrefix, Data),
            ok = stub_prefill(RemainingTokens);
        no_save ->
            ok = stub_prefill(Tokens)
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
            Token = stub_decode_one(Data#data.context_tokens),
            Data1 = Data#data{
                context_tokens = Data#data.context_tokens ++ [Token],
                generated = Data#data.generated ++ [Token]
            },
            Data2 = maybe_fire_continued(Data1),
            {keep_state, Data2, [{next_event, internal, decode_step}]}
    end.

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
    Reply = stub_detokenize(Data#data.generated),
    From = Data#data.caller,
    Generated = Data#data.generated,
    Data1 = Data#data{
        caller = undefined,
        context_tokens = [],
        prompt_tokens = [],
        generated = [],
        response_target = 0,
        last_save_at = 0
    },
    {next_state, idle, Data1, [{reply, From, {ok, Reply, Generated}}]}.

%% =============================================================================
%% Internal: cache integration
%% =============================================================================

lookup_or_resume(PromptTokens, ParentKey, Data) ->
    %% Exact-key fast path.
    Key = make_key(PromptTokens, Data),
    case erllama_cache_meta_srv:lookup_exact(Key) of
        {ok, Row} ->
            {warm, load_tokens_from_row(Row, Data), []};
        miss when ParentKey =/= undefined ->
            try_session_resume(PromptTokens, ParentKey, Data);
        miss ->
            cold
    end.

try_session_resume(PromptTokens, ParentKey, Data) ->
    Wait = maps:get(session_resume_wait_ms, Data#data.policy, 500),
    case erllama_cache_meta_srv:lookup_exact_or_wait(ParentKey, Wait) of
        {ok, Row} ->
            ParentTokens = load_tokens_from_row(Row, Data),
            case is_strict_prefix(ParentTokens, PromptTokens) of
                true ->
                    %% Slab covers the first length(ParentTokens) tokens;
                    %% prefill the remainder.
                    Remaining = lists:nthtail(length(ParentTokens), PromptTokens),
                    {warm, ParentTokens, Remaining};
                false ->
                    cold
            end;
        miss ->
            cold
    end.

load_tokens_from_row(Row, Data) ->
    %% In v0.1 the "kv state" is just the token list serialised via
    %% erllama_cache_key:encode_tokens. Real kv_unpack lands at step
    %% 2b and replaces this load path.
    Tier = element(?POS_TIER, Row),
    Key = element(?POS_KEY, Row),
    case Tier of
        ram ->
            case erllama_cache_ram:load(Key) of
                {ok, Bin} -> erllama_cache_key:decode_tokens(Bin);
                miss -> []
            end;
        _ ->
            case erllama_cache_disk_srv:load(Data#data.tier_srv, Key) of
                {ok, _Info, Payload} -> erllama_cache_key:decode_tokens(Payload);
                _ -> []
            end
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
    Payload = stub_kv_pack(Tokens),
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
%% Internal: model stubs (replaced by erllama_nif at step 2b)
%% =============================================================================

stub_tokenize(<<>>) ->
    [];
stub_tokenize(Prompt) when is_binary(Prompt) ->
    [
        erlang:phash2(W) rem (1 bsl 32)
     || W <- binary:split(Prompt, <<" ">>, [global, trim_all]),
        W =/= <<>>
    ].

stub_detokenize(Tokens) ->
    list_to_binary(
        lists:join(<<" ">>, [integer_to_binary(T) || T <- Tokens])
    ).

stub_prefill(_Tokens) ->
    ok.

stub_decode_one(ContextTokens) ->
    erlang:phash2({decode, ContextTokens}) rem (1 bsl 32).

stub_kv_pack(Tokens) ->
    erllama_cache_key:encode_tokens(Tokens).
