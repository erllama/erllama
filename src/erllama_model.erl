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
    status/1
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
    generated :: [non_neg_integer()]
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
        generated = []
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
generating(EventType, EventContent, Data) ->
    handle_common(generating, EventType, EventContent, Data).

%% =============================================================================
%% Common event handler
%% =============================================================================

handle_common(_State, {call, From}, {complete, _, _}, Data) ->
    %% Reject concurrent complete calls; only one in flight.
    {keep_state, Data, [{reply, From, {error, busy}}]};
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
    %% Run the decode loop (stubbed: produce N deterministic tokens).
    Generated = stub_decode(
        Data#data.context_tokens,
        Data#data.response_target
    ),
    LiveTokens = Data#data.context_tokens ++ Generated,
    ok = fire_finish_save(LiveTokens, Data),
    Reply = stub_detokenize(Generated),
    Data1 = Data#data{
        caller = undefined,
        context_tokens = [],
        prompt_tokens = [],
        generated = [],
        response_target = 0
    },
    {next_state, idle, Data1, [{reply, Data#data.caller, {ok, Reply, Generated}}]}.

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

stub_decode(ContextTokens, N) ->
    Seed = erlang:phash2(ContextTokens),
    [erlang:phash2({Seed, I}) rem (1 bsl 32) || I <- lists:seq(1, N)].

stub_kv_pack(Tokens) ->
    erllama_cache_key:encode_tokens(Tokens).
