%% @doc
%% File-tier save orchestrator with an ETS-backed counting semaphore.
%%
%% Modelled on `py_semaphore` from erlang-python (Discord pattern):
%% one ETS table holding `{running, N}` and `{max, M}` counters,
%% atomic `update_counter` for both directions, exponential backoff
%% on contention, saturating decrement on release so a buggy or
%% double-release cannot drive the counter negative.
%%
%% Save pipeline run by `save/4` in the caller's process:
%%
%%   acquire (ETS counter, blocking with backoff up to AcquireTimeoutMs)
%%   meta_srv:reserve_save -> Token
%%   disk_srv:save (returns once the file is linked + validated)
%%   meta_srv:announce_saved -> available
%%   release
%%
%% A `try/after` releases the slot on normal exits and exceptions.
%% A SIGKILL of the caller mid-save still leaks a slot, but the
%% saturating release means the leak does not corrupt subsequent
%% releases; counters can only drift upward, recovered by
%% `set_max_concurrent/1` or restart.
%% @end
-module(erllama_cache_writer).

-export([
    init/0,
    init/1,
    save/4,
    save/5,
    acquire/1,
    release/0,
    current/0,
    max_concurrent/0,
    set_max_concurrent/1
]).

-define(TABLE, erllama_cache_writer_sem).
-define(COUNTER_KEY, running).
-define(MAX_KEY, max).
-define(BACKOFF_MS, 5).
-define(MAX_BACKOFF_MS, 50).
-define(DEFAULT_TIMEOUT_MS, 30000).

%% =============================================================================
%% Public API: semaphore lifecycle
%% =============================================================================

%% @doc Initialise the semaphore with a sensible default
%% (`schedulers * 2 + 1`). Idempotent.
-spec init() -> ok.
init() ->
    init(default_max_concurrent()).

-spec init(pos_integer()) -> ok.
init(Max) when is_integer(Max), Max > 0 ->
    case ets:whereis(?TABLE) of
        undefined ->
            _ = ets:new(?TABLE, [
                named_table,
                public,
                {write_concurrency, true},
                {read_concurrency, true}
            ]),
            ets:insert(?TABLE, [{?COUNTER_KEY, 0}, {?MAX_KEY, Max}]),
            ok;
        _Tid ->
            ets:insert(?TABLE, {?MAX_KEY, Max}),
            ok
    end.

-spec current() -> non_neg_integer().
current() ->
    case ets:lookup(?TABLE, ?COUNTER_KEY) of
        [{_, N}] -> N;
        [] -> 0
    end.

-spec max_concurrent() -> pos_integer().
max_concurrent() ->
    case ets:lookup(?TABLE, ?MAX_KEY) of
        [{_, Max}] -> Max;
        [] -> default_max_concurrent()
    end.

-spec set_max_concurrent(pos_integer()) -> ok.
set_max_concurrent(Max) when is_integer(Max), Max > 0 ->
    ets:insert(?TABLE, {?MAX_KEY, Max}),
    ok.

%% @doc Try to acquire a slot. Blocks with exponential backoff up
%% to `Timeout` milliseconds. Returns `ok` or
%% `{error, max_concurrent}` on timeout.
-spec acquire(timeout()) -> ok | {error, max_concurrent}.
acquire(infinity) ->
    acquire_loop(infinity, 0, ?BACKOFF_MS);
acquire(Timeout) when is_integer(Timeout), Timeout >= 0 ->
    Start = erlang:monotonic_time(millisecond),
    acquire_loop(Timeout, Start, ?BACKOFF_MS).

-spec release() -> ok.
release() ->
    %% Saturating decrement: {Pos, Inc, Threshold, SetValue}.
    %% If a buggy double-release would push below 0, snap back to 0.
    _ = ets:update_counter(?TABLE, ?COUNTER_KEY, {2, -1, 0, 0}),
    ok.

%% =============================================================================
%% Public API: save pipeline
%% =============================================================================

-spec save(
    atom(),
    disk | ram_file,
    erllama_cache_kvc:build_meta(),
    binary()
) -> {ok, erllama_cache:cache_key()} | {error, term()}.
save(TierSrv, Tier, BuildMeta, Payload) ->
    save(TierSrv, Tier, BuildMeta, Payload, ?DEFAULT_TIMEOUT_MS).

-spec save(
    atom(),
    disk | ram_file,
    erllama_cache_kvc:build_meta(),
    binary(),
    timeout()
) -> {ok, erllama_cache:cache_key()} | {error, term()}.
save(TierSrv, Tier, BuildMeta, Payload, AcquireTimeout) when
    Tier =:= disk; Tier =:= ram_file
->
    case acquire(AcquireTimeout) of
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
%% Internal: semaphore loop
%% =============================================================================

-spec acquire_loop(timeout(), integer(), pos_integer()) ->
    ok | {error, max_concurrent}.
acquire_loop(Timeout, Start, Backoff) ->
    Max = max_concurrent(),
    N = ets:update_counter(?TABLE, ?COUNTER_KEY, {2, 1}),
    case N =< Max of
        true ->
            ok;
        false ->
            release(),
            case check_timeout(Timeout, Start) of
                continue ->
                    Jitter = rand:uniform(Backoff div 2 + 1),
                    timer:sleep(Backoff + Jitter),
                    acquire_loop(Timeout, Start, min(Backoff * 2, ?MAX_BACKOFF_MS));
                timeout ->
                    {error, max_concurrent}
            end
    end.

check_timeout(infinity, _) ->
    continue;
check_timeout(Timeout, Start) ->
    case erlang:monotonic_time(millisecond) - Start >= Timeout of
        true -> timeout;
        false -> continue
    end.

-spec default_max_concurrent() -> pos_integer().
default_max_concurrent() ->
    case application:get_env(erllama, writer_max_concurrent) of
        {ok, N} when is_integer(N), N > 0 -> N;
        _ -> erlang:system_info(schedulers) * 2 + 1
    end.

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
