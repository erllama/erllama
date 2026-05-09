%% @doc
%% File-tier save orchestrator.
%%
%% Wraps the meta-server reservation protocol around a tier-server
%% save call:
%%
%%   acquire (atomics semaphore)
%%   meta_srv:reserve_save -> Token
%%   disk_srv:save (returns once the file is linked + validated)
%%   meta_srv:announce_saved -> available
%%   release
%%
%% Concurrency control is a single lock-free atomics counter, NOT a
%% pool of worker processes. Each calling process runs the save
%% pipeline itself, gated by `acquire/0` returning `ok` or
%% `{error, busy}` depending on the current in-flight count vs
%% configured maximum. The cache façade (step 11+) decides whether
%% to drop, retry, or queue when busy.
%%
%% Caveat: if a writer process dies after `acquire/0` and before
%% `release/0` (e.g. SIGKILL), the slot leaks. The save path is
%% short (sub-second to a few seconds); in normal operation the
%% try/after release runs reliably. A periodic reconciler is a
%% future addition if leak rates become a problem.
%% @end
-module(erllama_cache_writer).

-export([
    init/1,
    save/4,
    inflight/0,
    max_concurrent/0
]).

-define(SEM_KEY, {?MODULE, sem}).
-define(SLOT_INFLIGHT, 1).
-define(SLOT_MAX, 2).

%% =============================================================================
%% Public API
%% =============================================================================

-spec init(pos_integer()) -> ok.
init(MaxConcurrent) when is_integer(MaxConcurrent), MaxConcurrent > 0 ->
    A = atomics:new(2, [{signed, false}]),
    atomics:put(A, ?SLOT_MAX, MaxConcurrent),
    persistent_term:put(?SEM_KEY, A),
    ok.

-spec inflight() -> non_neg_integer().
inflight() ->
    atomics:get(sem(), ?SLOT_INFLIGHT).

-spec max_concurrent() -> pos_integer().
max_concurrent() ->
    atomics:get(sem(), ?SLOT_MAX).

%% Run the full save pipeline for a file tier (`disk` or `ram_file`).
%% Returns `{ok, Key}` on success or `{error, Reason}` if the
%% semaphore is full, the meta server refuses the reservation, or
%% the tier server fails to write/link/validate.
-spec save(
    atom(),
    disk | ram_file,
    erllama_cache_kvc:build_meta(),
    binary()
) -> {ok, erllama_cache:cache_key()} | {error, term()}.
save(TierSrv, Tier, BuildMeta, Payload) when
    Tier =:= disk; Tier =:= ram_file
->
    case acquire() of
        ok ->
            try
                do_save(TierSrv, Tier, BuildMeta, Payload)
            after
                release()
            end;
        {error, _} = E ->
            E
    end.

%% =============================================================================
%% Internal: semaphore
%% =============================================================================

sem() ->
    persistent_term:get(?SEM_KEY).

acquire() ->
    A = sem(),
    Max = atomics:get(A, ?SLOT_MAX),
    case atomics:add_get(A, ?SLOT_INFLIGHT, 1) of
        N when N =< Max -> ok;
        _ ->
            atomics:sub(A, ?SLOT_INFLIGHT, 1),
            {error, busy}
    end.

release() ->
    A = sem(),
    atomics:sub(A, ?SLOT_INFLIGHT, 1),
    ok.

%% =============================================================================
%% Internal: pipeline
%% =============================================================================

do_save(TierSrv, Tier, BuildMeta, Payload) ->
    case derive_key(BuildMeta) of
        {ok, Key} ->
            case erllama_cache_meta_srv:reserve_save(Key, Tier, self()) of
                {ok, Token} ->
                    do_save_reserved(TierSrv, Key, Token, BuildMeta, Payload);
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            E
    end.

derive_key(BuildMeta) ->
    try
        {ok,
            erllama_cache_key:make(#{
                fingerprint => maps:get(fingerprint, BuildMeta),
                quant_type => maps:get(quant_type, BuildMeta),
                ctx_params_hash => maps:get(ctx_params_hash, BuildMeta),
                tokens => maps:get(tokens, BuildMeta)
            })}
    catch
        error:{badkey, K} -> {error, {missing_key, K}};
        error:Reason -> {error, Reason}
    end.

do_save_reserved(TierSrv, Key, Token, BuildMeta, Payload) ->
    case erllama_cache_disk_srv:save(TierSrv, BuildMeta, Payload) of
        {ok, _SameKey, Header, Size} ->
            case erllama_cache_meta_srv:announce_saved(Key, Token, Size, Header) of
                ok -> {ok, Key};
                {error, _} = E -> E
            end;
        {error, _} = E ->
            _ = erllama_cache_meta_srv:cancel_reservation(Key, Token),
            E
    end.
