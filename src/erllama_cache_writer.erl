%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% File-tier save orchestrator with a leak-proof ETS counting
%% semaphore.
%%
%% Modelled on `py_semaphore` from erlang-python (Discord pattern):
%% one ETS table holding `{running, N}` and `{max, M}` counters;
%% atomic `update_counter` for fast inspection from any process.
%%
%% Unlike the bare ETS pattern, this module also runs a small
%% gen_server that owns the holders map and monitors every active
%% acquirer. If a holder dies between acquire and release (SIGKILL,
%% process crash, etc.), the gen_server's `'DOWN'` handler releases
%% the slot. No leaks.
%%
%% Hot path:
%%
%%   `current/0` and `max_concurrent/0` read ETS directly
%%   (no server hop).
%%   `acquire/1` and `release/0` go through this gen_server, which
%%   serialises monitor bookkeeping. Per-call overhead is roughly
%%   one gen_server hop (~1 us); for save paths measured in
%%   seconds this is invisible.
%%
%% Save pipeline run by `save/4` in the caller's process:
%%
%%   acquire (with backoff up to AcquireTimeoutMs)
%%   meta_srv:reserve_save -> Token
%%   disk_srv:save (returns once the file is linked + validated)
%%   meta_srv:announce_saved -> available
%%   release (try/after)
%%
%% A `try/after` releases on normal exits and exceptions; the
%% gen_server's monitor catches everything else.
%% @end
-module(erllama_cache_writer).
-behaviour(gen_server).

-include("erllama_cache.hrl").

-export([
    start_link/0,
    start_link/1,
    save/4,
    save/5,
    acquire/1,
    release/0,
    current/0,
    max_concurrent/0,
    set_max_concurrent/1
]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(SERVER, ?MODULE).
-define(TABLE, erllama_cache_writer_sem).
-define(COUNTER_KEY, running).
-define(MAX_KEY, max).
-define(BACKOFF_MS, 5).
-define(MAX_BACKOFF_MS, 50).
-define(DEFAULT_TIMEOUT_MS, 30000).
-define(SLOT_PD_KEY, {?MODULE, slot}).

-record(state, {
    %% MonRef -> CallerPid; one entry per active acquire
    holders = #{} :: #{reference() => pid()}
}).

-type state() :: #state{}.

%% =============================================================================
%% Public API: lifecycle
%% =============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    start_link(default_max_concurrent()).

-spec start_link(pos_integer()) -> {ok, pid()} | {error, term()}.
start_link(Max) when is_integer(Max), Max > 0 ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [Max], []).

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

%% =============================================================================
%% Public API: semaphore
%% =============================================================================

%% @doc Try to acquire a slot. Blocks with exponential backoff up
%% to `Timeout` milliseconds. On success the holder ref is stashed
%% in the process dictionary so `release/0` finds it.
-spec acquire(timeout()) -> ok | {error, max_concurrent}.
acquire(infinity) ->
    acquire_loop(infinity, 0, ?BACKOFF_MS);
acquire(Timeout) when is_integer(Timeout), Timeout >= 0 ->
    acquire_loop(Timeout, erlang:monotonic_time(millisecond), ?BACKOFF_MS).

-spec release() -> ok.
release() ->
    case erase(?SLOT_PD_KEY) of
        Ref when is_reference(Ref) ->
            gen_server:call(?SERVER, {release, Ref});
        undefined ->
            ok
    end.

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
%% gen_server callbacks
%% =============================================================================

-spec init([pos_integer()]) -> {ok, state()}.
init([Max]) ->
    case ets:whereis(?TABLE) of
        undefined ->
            _ = ets:new(?TABLE, [
                named_table,
                public,
                {write_concurrency, true},
                {read_concurrency, true}
            ]),
            ets:insert(?TABLE, [{?COUNTER_KEY, 0}, {?MAX_KEY, Max}]);
        _Tid ->
            ets:insert(?TABLE, {?COUNTER_KEY, 0}),
            ets:insert(?TABLE, {?MAX_KEY, Max})
    end,
    {ok, #state{}}.

handle_call({try_acquire, Pid}, _From, S) ->
    Max = max_concurrent(),
    case ets:update_counter(?TABLE, ?COUNTER_KEY, {2, 1}) of
        N when N =< Max ->
            Ref = erlang:monitor(process, Pid),
            {reply, {ok, Ref}, S#state{holders = (S#state.holders)#{Ref => Pid}}};
        _ ->
            decr(),
            {reply, {error, busy}, S}
    end;
handle_call({release, Ref}, _From, S) ->
    case maps:take(Ref, S#state.holders) of
        {_Pid, Holders1} ->
            erlang:demonitor(Ref, [flush]),
            decr(),
            {reply, ok, S#state{holders = Holders1}};
        error ->
            %% Stale release (already cleaned up by 'DOWN').
            {reply, ok, S}
    end;
handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

-spec handle_cast(term(), state()) -> {noreply, state()}.
handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info({'DOWN', Ref, process, _DownPid, _Reason}, S) ->
    case maps:take(Ref, S#state.holders) of
        {_HolderPid, Holders1} ->
            decr(),
            {noreply, S#state{holders = Holders1}};
        error ->
            {noreply, S}
    end;
handle_info(_Msg, S) ->
    {noreply, S}.

%% =============================================================================
%% Internal: acquire loop
%% =============================================================================

acquire_loop(Timeout, Start, Backoff) ->
    case gen_server:call(?SERVER, {try_acquire, self()}) of
        {ok, Ref} ->
            put(?SLOT_PD_KEY, Ref),
            ok;
        {error, busy} ->
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

decr() ->
    %% Saturating decrement of the counter; clamps at 0.
    _ = ets:update_counter(?TABLE, ?COUNTER_KEY, {2, -1, 0, 0}),
    ok.

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
                {error, already_present} ->
                    erllama_cache_counters:incr(?C_DUPLICATE_DROPPED),
                    {error, already_present};
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

%% Stage-aware reservation flow:
%%   reserve_save (already done) → write_tmp → check_reservation
%%     → publish (link) → mark_published → announce_saved
%%
%% check_reservation right before link is the cancel-token check that
%% prevents a slow writer (whose reservation TTL expired and got
%% recycled to another writer) from publishing under stale authority.
%% mark_published advances the meta server's stage to post_link so a
%% writer that crashes between link and announce_saved triggers the
%% validate-and-adopt path rather than the placeholder-row delete.
do_save_reserved(TierSrv, Key, Token, BuildMeta, Payload) ->
    Root = erllama_cache_disk_srv:dir(TierSrv),
    case erllama_cache_disk_srv:write_tmp(Root, BuildMeta, Payload) of
        {ok, WriteOut} ->
            after_write_tmp(Key, Token, BuildMeta, WriteOut);
        {error, _} = E ->
            _ = erllama_cache_meta_srv:cancel_reservation(Key, Token),
            E
    end.

after_write_tmp(Key, Token, BuildMeta, WriteOut) ->
    case erllama_cache_meta_srv:check_reservation(Key, Token) of
        ok ->
            after_check_reservation(Key, Token, BuildMeta, WriteOut);
        {error, expired} = E ->
            erllama_cache_disk_srv:abort_tmp(WriteOut),
            E
    end.

after_check_reservation(Key, Token, BuildMeta, WriteOut) ->
    case erllama_cache_disk_srv:publish(WriteOut) of
        {ok, _SameKey, Header, Size} ->
            #{final := FinalPath} = WriteOut,
            after_publish(Key, Token, BuildMeta, FinalPath, Header, Size);
        {error, _} = E ->
            erllama_cache_disk_srv:abort_tmp(WriteOut),
            _ = erllama_cache_meta_srv:cancel_reservation(Key, Token),
            E
    end.

after_publish(Key, Token, BuildMeta, FinalPath, Header, Size) ->
    case erllama_cache_meta_srv:mark_published(Key, Token, FinalPath) of
        ok ->
            TokensBin = erllama_cache_key:encode_tokens(maps:get(tokens, BuildMeta)),
            case
                erllama_cache_meta_srv:announce_saved(
                    Key, Token, Size, Header, TokensBin
                )
            of
                ok ->
                    bump_save_counter(maps:get(save_reason, BuildMeta, unknown)),
                    {ok, Key};
                {error, _} = E ->
                    E
            end;
        {error, _} = E ->
            %% Token was recycled between check_reservation and
            %% mark_published. The file is published; the meta sweep
            %% will validate-and-adopt it on its next pass.
            E
    end.

bump_save_counter(cold) -> erllama_cache_counters:incr(?C_SAVES_COLD);
bump_save_counter(continued) -> erllama_cache_counters:incr(?C_SAVES_CONTINUED);
bump_save_counter(finish) -> erllama_cache_counters:incr(?C_SAVES_FINISH);
bump_save_counter(evict) -> erllama_cache_counters:incr(?C_SAVES_EVICT);
bump_save_counter(shutdown) -> erllama_cache_counters:incr(?C_SAVES_SHUTDOWN);
bump_save_counter(_) -> ok.
