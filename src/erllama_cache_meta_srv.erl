%% @doc
%% Sole writer for the cache meta and LRU ETS tables; arbitrates
%% claim/release and the reservation state machine for save
%% publication.
%%
%% Two read-mostly ETS tables, owned by this process and `protected`
%% so any caller can read them without a server hop:
%%
%%   erllama_cache_meta : set, key = cache_key, row layout per
%%                        include/erllama_cache.hrl ?POS_* constants
%%   erllama_cache_lru  : ordered_set, key = {LastUsedNs, cache_key},
%%                        value = []
%%
%% Two server-internal maps in process state:
%%
%%   holders      : MonRef -> {Pid, Key}; one entry per active claim
%%   reservations : Key -> #reservation{}; one entry per in-flight save
%%
%% Plus a waiters map for `lookup_exact_or_wait/2` which defers replies
%% until the in-flight save publishes (or the per-call deadline fires).
%%
%% The reservation state machine has two stages, `pre_link` and
%% `post_link`, to make crash cleanup correct: a writer that died
%% before `link/2` leaves no file; a writer that died after `link/2`
%% may have left a valid `.kvc` we can validate-and-adopt.
%% @end
-module(erllama_cache_meta_srv).
-behaviour(gen_server).

-include("erllama_cache.hrl").

-export([
    start_link/0,
    %% Read-only (no server hop)
    lookup_exact/1,
    lookup_longest_prefix/4,
    %% Read with bounded wait for an in-flight save
    lookup_exact_or_wait/2,
    %% Claim/release (active reader of a slab)
    checkout/2,
    checkin/1,
    %% Reservation state machine (writer)
    reserve_save/3,
    check_reservation/2,
    mark_published/3,
    announce_saved/4,
    announce_saved/5,
    cancel_reservation/2,
    %% Operator/test helpers
    gc/0,
    evict_bytes/1,
    evict_bytes/2,
    dump/0,
    dump/1,
    insert_available/5,
    insert_available/6
]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(SERVER, ?MODULE).
-define(TBL_META, erllama_cache_meta).
-define(TBL_LRU, erllama_cache_lru).

%% Default time-to-live for a reservation when the writer is silent.
%% Refreshed on each successful state transition (reserve / check /
%% mark_published / announce_saved).
-define(DEFAULT_TTL_NS, 60 * 1000 * 1000 * 1000).

%% Periodic sweep for stale reservations and expired waiters.
-define(SWEEP_INTERVAL_MS, 30 * 1000).

-record(reservation, {
    writer :: pid(),
    token :: reference(),
    monref :: reference(),
    expires_ns :: integer(),
    stage :: pre_link | post_link,
    tier :: erllama_cache:tier(),
    path :: file:name() | undefined
}).

-record(state, {
    holders :: #{reference() => {pid(), erllama_cache:cache_key()}},
    reservations :: #{erllama_cache:cache_key() => #reservation{}},
    waiters :: #{erllama_cache:cache_key() => [{gen_server:from(), integer()}]},
    sweep_timer :: reference() | undefined
}).

-type state() :: #state{}.

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [], []).

-spec lookup_exact(erllama_cache:cache_key()) ->
    {ok, tuple()} | miss.
lookup_exact(Key) ->
    try ets:lookup_element(?TBL_META, Key, ?POS_STATUS) of
        available ->
            [Row] = ets:lookup(?TBL_META, Key),
            {ok, Row};
        _ ->
            miss
    catch
        error:badarg -> miss
    end.

-spec lookup_exact_or_wait(erllama_cache:cache_key(), non_neg_integer()) ->
    {ok, tuple()} | miss.
lookup_exact_or_wait(Key, MaxWaitMs) ->
    gen_server:call(?SERVER, {lookup_or_wait, Key, MaxWaitMs}, infinity).

%% @doc Walk `Tokens` backward in `Stride` steps and return the row
%% for the longest cached prefix. Pure ETS reads, no server hop. Used
%% by stateless callers (HTTP front-end, agent loops) that resend a
%% full conversation each turn and don't have a `parent_key` to
%% thread.
%%
%% Stops at `MinTokens` floor and returns `miss` if nothing matches.
%% Walks at most `length(Tokens) / Stride` rows; with the default
%% 2048-token stride that's ~15 lookups for a 30k-token prompt.
%%
%% Encodes the full token list to a binary once at entry, then passes
%% `binary:part(TokensBin, 0, N*4)` sub-binaries (O(1) views) to
%% `erllama_cache_key:make/4` per probe. Avoids re-traversing the
%% list and re-allocating the binary on every step; the only
%% per-probe cost is the SHA-256 over N*4 bytes plus the ETS lookup.
-spec lookup_longest_prefix(map(), [erllama_nif:token_id()], pos_integer(), pos_integer()) ->
    {ok, pos_integer(), tuple()} | miss.
lookup_longest_prefix(KeyMeta, Tokens, Stride, MinTokens) when
    is_integer(Stride), Stride > 0, is_integer(MinTokens), MinTokens > 0
->
    Len = length(Tokens),
    Start = (Len div Stride) * Stride,
    Floor = max(MinTokens, Stride),
    AllTokensBin = erllama_cache_key:encode_tokens(Tokens),
    #{fingerprint := Fp, quant_type := QT, ctx_params_hash := CtxHash} = KeyMeta,
    walk_prefix(Fp, QT, CtxHash, AllTokensBin, Start, Stride, Floor).

walk_prefix(_Fp, _QT, _CtxHash, _Bin, N, _Stride, Floor) when N < Floor ->
    miss;
walk_prefix(Fp, QT, CtxHash, Bin, N, Stride, Floor) ->
    PrefixBin = binary:part(Bin, 0, N * 4),
    Key = erllama_cache_key:make(Fp, QT, CtxHash, PrefixBin),
    case lookup_exact(Key) of
        {ok, Row} -> {ok, N, Row};
        miss -> walk_prefix(Fp, QT, CtxHash, Bin, N - Stride, Stride, Floor)
    end.

-spec checkout(erllama_cache:cache_key(), pid()) ->
    {ok, reference(), erllama_cache:tier(), term(), binary(), term()}
    | {error, busy}
    | miss.
checkout(Key, Pid) when is_pid(Pid) ->
    gen_server:call(?SERVER, {checkout, Key, Pid}).

-spec checkin(reference()) -> ok.
checkin(MonRef) when is_reference(MonRef) ->
    gen_server:call(?SERVER, {checkin, MonRef}).

-spec reserve_save(erllama_cache:cache_key(), erllama_cache:tier(), pid()) ->
    {ok, reference()} | {error, already_present | conflict}.
reserve_save(Key, Tier, Pid) when is_pid(Pid) ->
    gen_server:call(?SERVER, {reserve_save, Key, Tier, Pid}).

-spec check_reservation(erllama_cache:cache_key(), reference()) ->
    ok | {error, expired}.
check_reservation(Key, Token) when is_reference(Token) ->
    gen_server:call(?SERVER, {check_reservation, Key, Token}).

-spec mark_published(erllama_cache:cache_key(), reference(), file:name()) ->
    ok | {error, expired}.
mark_published(Key, Token, Path) when is_reference(Token) ->
    gen_server:call(?SERVER, {mark_published, Key, Token, Path}).

-spec announce_saved(
    erllama_cache:cache_key(), reference(), non_neg_integer(), binary()
) -> ok | {error, expired}.
announce_saved(Key, Token, Size, Header) ->
    announce_saved(Key, Token, Size, Header, undefined).

-spec announce_saved(
    erllama_cache:cache_key(),
    reference(),
    non_neg_integer(),
    binary(),
    binary() | undefined
) -> ok | {error, expired}.
announce_saved(Key, Token, Size, Header, TokensBin) when
    is_reference(Token), is_binary(Header)
->
    gen_server:call(?SERVER, {announce_saved, Key, Token, Size, Header, TokensBin}).

-spec cancel_reservation(erllama_cache:cache_key(), reference()) -> ok.
cancel_reservation(Key, Token) when is_reference(Token) ->
    gen_server:call(?SERVER, {cancel_reservation, Key, Token}).

%% Direct insertion of an `available` row. Used by the RAM tier
%% (which has no on-disk publish step) and by the disk tier on-start
%% scan to register pre-existing valid files.
-spec insert_available(
    erllama_cache:cache_key(),
    erllama_cache:tier(),
    non_neg_integer(),
    binary(),
    term()
) -> ok.
insert_available(Key, Tier, Size, Header, Location) ->
    insert_available(Key, Tier, Size, Header, Location, undefined).

-spec insert_available(
    erllama_cache:cache_key(),
    erllama_cache:tier(),
    non_neg_integer(),
    binary(),
    term(),
    binary() | undefined
) -> ok.
insert_available(Key, Tier, Size, Header, Location, TokensBin) ->
    gen_server:call(
        ?SERVER, {insert_available, Key, Tier, Size, Header, Location, TokensBin}
    ).

-spec gc() -> {evicted, non_neg_integer()}.
gc() ->
    gen_server:call(?SERVER, gc).

%% @doc Evict oldest available rows until at least `TargetBytes` have
%% been freed (or no more candidates remain). Returns the number of
%% rows evicted and the bytes actually freed.
-spec evict_bytes(non_neg_integer()) ->
    {evicted, non_neg_integer(), non_neg_integer()}.
evict_bytes(TargetBytes) ->
    evict_bytes(TargetBytes, all).

%% @doc Evict oldest available rows whose tier is in `Tiers` until at
%% least `TargetBytes` have been freed. `Tiers = all` matches every
%% tier; otherwise it must be a list drawn from `[ram, ram_file, disk]`.
-spec evict_bytes(non_neg_integer(), all | [erllama_cache:tier()]) ->
    {evicted, non_neg_integer(), non_neg_integer()}.
evict_bytes(TargetBytes, Tiers) when is_integer(TargetBytes), TargetBytes >= 0 ->
    gen_server:call(?SERVER, {evict_bytes, TargetBytes, Tiers}).

-spec dump() -> [tuple()].
dump() ->
    ets:tab2list(?TBL_META).

-spec dump(erllama_cache:cache_key()) -> {ok, tuple()} | miss.
dump(Key) ->
    case ets:lookup(?TBL_META, Key) of
        [Row] -> {ok, Row};
        [] -> miss
    end.
%% Note: dump/1 wants the full row, so the lookup/2 form is appropriate
%% here. lookup_element + lookup would be two ETS ops on the hit path
%% with no benefit.

%% =============================================================================
%% gen_server callbacks
%% =============================================================================

-spec init([]) -> {ok, state()}.
init([]) ->
    EtsOpts = [
        named_table,
        protected,
        {keypos, 1},
        {read_concurrency, true}
    ],
    ets:new(?TBL_META, [set | EtsOpts]),
    ets:new(?TBL_LRU, [ordered_set | EtsOpts]),
    Timer = erlang:send_after(?SWEEP_INTERVAL_MS, self(), sweep),
    State = #state{
        holders = #{},
        reservations = #{},
        waiters = #{},
        sweep_timer = Timer
    },
    {ok, State}.

handle_call({checkout, Key, Pid}, _From, S) ->
    try ets:lookup_element(?TBL_META, Key, ?POS_STATUS) of
        available ->
            [Row] = ets:lookup(?TBL_META, Key),
            do_checkout(Key, Pid, Row, S);
        _ ->
            {reply, {error, busy}, S}
    catch
        error:badarg -> {reply, miss, S}
    end;
handle_call({checkin, MonRef}, _From, S) ->
    case maps:take(MonRef, S#state.holders) of
        {{_Pid, Key}, Holders1} ->
            erlang:demonitor(MonRef, [flush]),
            decrement_refcount(Key),
            {reply, ok, S#state{holders = Holders1}};
        error ->
            {reply, ok, S}
    end;
handle_call({lookup_or_wait, Key, MaxWaitMs}, From, S) ->
    try ets:lookup_element(?TBL_META, Key, ?POS_STATUS) of
        available ->
            [Row] = ets:lookup(?TBL_META, Key),
            {reply, {ok, Row}, S};
        writing when MaxWaitMs > 0 ->
            {noreply, add_waiter(Key, From, MaxWaitMs, S)};
        _ ->
            {reply, miss, S}
    catch
        error:badarg -> {reply, miss, S}
    end;
handle_call({reserve_save, Key, Tier, Pid}, _From, S) ->
    do_reserve_save(Key, Tier, Pid, S);
handle_call({check_reservation, Key, Token}, _From, S) ->
    case maps:get(Key, S#state.reservations, undefined) of
        #reservation{token = Token} = R ->
            R1 = R#reservation{expires_ns = monotonic_ns() + ?DEFAULT_TTL_NS},
            {reply, ok, S#state{reservations = (S#state.reservations)#{Key => R1}}};
        _ ->
            {reply, {error, expired}, S}
    end;
handle_call({mark_published, Key, Token, Path}, _From, S) ->
    case maps:get(Key, S#state.reservations, undefined) of
        #reservation{token = Token} = R ->
            R1 = R#reservation{
                stage = post_link,
                path = Path,
                expires_ns = monotonic_ns() + ?DEFAULT_TTL_NS
            },
            {reply, ok, S#state{reservations = (S#state.reservations)#{Key => R1}}};
        _ ->
            {reply, {error, expired}, S}
    end;
handle_call({announce_saved, Key, Token, Size, Header, TokensBin}, _From, S) ->
    case maps:get(Key, S#state.reservations, undefined) of
        #reservation{token = Token, monref = MonRef, tier = Tier, path = Path} ->
            erlang:demonitor(MonRef, [flush]),
            install_available_row(
                Key, Tier, Size, Header, location_for(Tier, Path), TokensBin
            ),
            S1 = S#state{reservations = maps:remove(Key, S#state.reservations)},
            S2 = notify_waiters(Key, S1),
            {reply, ok, S2};
        _ ->
            {reply, {error, expired}, S}
    end;
handle_call({cancel_reservation, Key, Token}, _From, S) ->
    case maps:get(Key, S#state.reservations, undefined) of
        #reservation{token = Token} = R ->
            erlang:demonitor(R#reservation.monref, [flush]),
            ets:delete(?TBL_META, Key),
            S1 = S#state{reservations = maps:remove(Key, S#state.reservations)},
            {reply, ok, S1};
        _ ->
            {reply, ok, S}
    end;
handle_call({insert_available, Key, Tier, Size, Header, Location, TokensBin}, _From, S) ->
    install_available_row(Key, Tier, Size, Header, Location, TokensBin),
    S1 = notify_waiters(Key, S),
    {reply, ok, S1};
handle_call(gc, _From, S) ->
    Evicted = run_eviction(),
    {reply, {evicted, Evicted}, S};
handle_call({evict_bytes, 0, _Tiers}, _From, S) ->
    %% "Evict at least 0 bytes" is a no-op. Use gc/0 for full GC.
    {reply, {evicted, 0, 0}, S};
handle_call({evict_bytes, Target, Tiers}, _From, S) when Target > 0 ->
    {N, Bytes} = run_eviction_bytes(Target, tier_pred(Tiers)),
    {reply, {evicted, N, Bytes}, S};
handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

-spec handle_cast(term(), state()) -> {noreply, state()}.
handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info({'DOWN', Ref, process, _DownPid, _Reason}, S) ->
    case maps:take(Ref, S#state.holders) of
        {{_HolderPid, Key}, Holders1} ->
            decrement_refcount(Key),
            {noreply, S#state{holders = Holders1}};
        error ->
            {noreply, on_writer_down(Ref, S)}
    end;
handle_info({waiter_expire, Key, From}, S) ->
    {noreply, expire_waiter(Key, From, S)};
handle_info(sweep, S) ->
    S1 = sweep_reservations(S),
    Timer = erlang:send_after(?SWEEP_INTERVAL_MS, self(), sweep),
    {noreply, S1#state{sweep_timer = Timer}};
handle_info(_Msg, S) ->
    {noreply, S}.

terminate(_Reason, _S) ->
    ok.

%% =============================================================================
%% Internal: checkout / refcount / LRU
%% =============================================================================

do_checkout(Key, Pid, Row, S) ->
    MonRef = erlang:monitor(process, Pid),
    NowNs = monotonic_ns(),
    OldLastUsed = element(?POS_LAST_USED, Row),
    ets:delete(?TBL_LRU, {OldLastUsed, Key}),
    ets:insert(?TBL_LRU, {{NowNs, Key}, []}),
    [NewHits] = ets:update_counter(?TBL_META, Key, [{?POS_HITS, +1}]),
    _ = ets:update_counter(?TBL_META, Key, {?POS_REFCOUNT, +1}),
    ets:update_element(?TBL_META, Key, {?POS_LAST_USED, NowNs}),
    Tier = element(?POS_TIER, Row),
    Loc = element(?POS_LOCATION, Row),
    Header = element(?POS_HEADER_BIN, Row),
    Tokens = element(?POS_TOKENS_REF, Row),
    %% Persist the bumped hit count for restart-survival. Best-effort:
    %% a failed write only loses the increment, with no behavioural
    %% impact this run. RAM tier has no persistent file.
    persist_hits(Loc, NewHits),
    Holders1 = (S#state.holders)#{MonRef => {Pid, Key}},
    {reply, {ok, MonRef, Tier, Loc, Header, Tokens}, S#state{holders = Holders1}}.

persist_hits({disk, Path}, Hits) ->
    erllama_cache_disk_srv:touch_hits(Path, Hits);
persist_hits({ram_file, Path}, Hits) ->
    erllama_cache_disk_srv:touch_hits(Path, Hits);
persist_hits(_, _) ->
    ok.

decrement_refcount(Key) ->
    try
        _ = ets:update_counter(?TBL_META, Key, {?POS_REFCOUNT, -1, 0, 0}),
        ok
    catch
        error:badarg -> ok
    end.

install_available_row(Key, Tier, Size, Header, Location, TokensBin) ->
    NowNs = monotonic_ns(),
    try ets:lookup_element(?TBL_META, Key, ?POS_LAST_USED) of
        OldLastUsed -> ets:delete(?TBL_LRU, {OldLastUsed, Key})
    catch
        error:badarg -> ok
    end,
    Hits = hits_from_header(Header),
    %% Bias last_used by accumulated hits so a high-hit row survives
    %% an LRU walk on a freshly-restarted server, where every row's
    %% natural last_used would otherwise collapse to ~NowNs and leave
    %% the order effectively random. Each accumulated hit pushes the
    %% row 1 second forward in the LRU. Once a row is actively
    %% checked out at runtime, recency takes over.
    LastUsed = NowNs + Hits * 1_000_000_000,
    Row =
        {Key, Tier, Size, LastUsed, 0, available, Header, Location, TokensBin, Hits},
    ets:insert(?TBL_META, Row),
    ets:insert(?TBL_LRU, {{LastUsed, Key}, []}),
    ok.

%% Extract the u32 hit_count from the on-disk header. RAM tier saves
%% pass a placeholder header so we treat a too-short header as "no
%% prior hits".
hits_from_header(<<_:?KVC_HEADER_HITS_OFFSET/binary, Hits:32/little, _/binary>>) ->
    Hits;
hits_from_header(_) ->
    0.

%% =============================================================================
%% Internal: reservation
%% =============================================================================

do_reserve_save(Key, Tier, Pid, S) ->
    try ets:lookup_element(?TBL_META, Key, ?POS_STATUS) of
        available ->
            {reply, {error, already_present}, S};
        writing ->
            handle_existing_writing_row(Key, Tier, Pid, S);
        evicting ->
            {reply, {error, conflict}, S}
    catch
        error:badarg -> create_reservation(Key, Tier, Pid, S)
    end.

handle_existing_writing_row(Key, Tier, Pid, S) ->
    case maps:get(Key, S#state.reservations, undefined) of
        undefined ->
            ets:delete(?TBL_META, Key),
            create_reservation(Key, Tier, Pid, S);
        #reservation{} = Old ->
            case reservation_is_live(Old) of
                true ->
                    {reply, {error, conflict}, S};
                false ->
                    S1 = cleanup_stale_reservation(Key, Old, S),
                    create_reservation(Key, Tier, Pid, S1)
            end
    end.

reservation_is_live(#reservation{writer = Pid, expires_ns = E}) ->
    is_process_alive(Pid) andalso E > monotonic_ns().

create_reservation(Key, Tier, Pid, S) ->
    MonRef = erlang:monitor(process, Pid),
    Token = erlang:make_ref(),
    NowNs = monotonic_ns(),
    R = #reservation{
        writer = Pid,
        token = Token,
        monref = MonRef,
        expires_ns = NowNs + ?DEFAULT_TTL_NS,
        stage = pre_link,
        tier = Tier,
        path = undefined
    },
    Placeholder = {Key, Tier, 0, NowNs, 0, writing, <<>>, undefined, undefined},
    ets:insert(?TBL_META, Placeholder),
    {reply, {ok, Token}, S#state{reservations = (S#state.reservations)#{Key => R}}}.

cleanup_stale_reservation(Key, Old, S) ->
    erlang:demonitor(Old#reservation.monref, [flush]),
    cleanup_by_stage(Key, Old, S).

cleanup_by_stage(Key, #reservation{stage = pre_link}, S) ->
    ets:delete(?TBL_META, Key),
    S#state{reservations = maps:remove(Key, S#state.reservations)};
cleanup_by_stage(Key, #reservation{stage = post_link, path = Path, tier = Tier}, S) ->
    case validate_and_adopt(Key, Path) of
        {ok, Size, Header, TokensBin} ->
            install_available_row(
                Key, Tier, Size, Header, location_for(Tier, Path), TokensBin
            ),
            S1 = S#state{reservations = maps:remove(Key, S#state.reservations)},
            notify_waiters(Key, S1);
        {error, _Reason} ->
            _ = file:delete(Path),
            ets:delete(?TBL_META, Key),
            S#state{reservations = maps:remove(Key, S#state.reservations)}
    end.

validate_and_adopt(Key, Path) ->
    case file:read_file(Path) of
        {ok, Bin} ->
            case erllama_cache_kvc:parse(Bin, Key) of
                {ok, Info, _Payload} ->
                    Tokens = maps:get(tokens, Info, []),
                    TokensBin = erllama_cache_key:encode_tokens(Tokens),
                    {ok, byte_size(Bin), header_slice(Bin), TokensBin};
                {error, R} ->
                    {error, R}
            end;
        {error, R} ->
            {error, R}
    end.

header_slice(Bin) ->
    HeaderSize = 48,
    binary:part(Bin, 0, HeaderSize).

location_for(ram, _) -> {ram};
location_for(ram_file, Path) -> {ram_file, Path};
location_for(disk, Path) -> {disk, Path}.

%% =============================================================================
%% Internal: writer DOWN handling
%% =============================================================================

on_writer_down(Ref, S) ->
    case find_reservation_by_monref(Ref, S#state.reservations) of
        {Key, R} -> cleanup_by_stage(Key, R, S);
        none -> S
    end.

find_reservation_by_monref(Ref, Reservations) ->
    Found = maps:fold(
        fun(K, R = #reservation{monref = M}, Acc) ->
            case M of
                Ref -> [{K, R} | Acc];
                _ -> Acc
            end
        end,
        [],
        Reservations
    ),
    case Found of
        [] -> none;
        [{K, R} | _] -> {K, R}
    end.

%% =============================================================================
%% Internal: waiters (lookup_exact_or_wait)
%% =============================================================================

add_waiter(Key, From, MaxWaitMs, S) ->
    Expires = monotonic_ns() + MaxWaitMs * 1_000_000,
    erlang:send_after(MaxWaitMs, self(), {waiter_expire, Key, From}),
    Existing = maps:get(Key, S#state.waiters, []),
    Waiters1 = (S#state.waiters)#{Key => [{From, Expires} | Existing]},
    S#state{waiters = Waiters1}.

notify_waiters(Key, S) ->
    case maps:take(Key, S#state.waiters) of
        {Waiters, W1} ->
            case ets:lookup(?TBL_META, Key) of
                [Row] -> [gen_server:reply(From, {ok, Row}) || {From, _} <- Waiters];
                [] -> [gen_server:reply(From, miss) || {From, _} <- Waiters]
            end,
            S#state{waiters = W1};
        error ->
            S
    end.

expire_waiter(Key, From, S) ->
    case maps:get(Key, S#state.waiters, []) of
        [] ->
            S;
        Waiters ->
            case lists:keytake(From, 1, Waiters) of
                {value, _, []} ->
                    gen_server:reply(From, miss),
                    S#state{waiters = maps:remove(Key, S#state.waiters)};
                {value, _, Rest} ->
                    gen_server:reply(From, miss),
                    S#state{waiters = (S#state.waiters)#{Key => Rest}};
                false ->
                    %% Already replied (e.g. a save published first).
                    S
            end
    end.

%% =============================================================================
%% Internal: sweep
%% =============================================================================

sweep_reservations(S) ->
    Now = monotonic_ns(),
    Stale = maps:fold(
        fun(K, R = #reservation{expires_ns = E, writer = Pid}, Acc) ->
            case E =< Now orelse not is_process_alive(Pid) of
                true -> [{K, R} | Acc];
                false -> Acc
            end
        end,
        [],
        S#state.reservations
    ),
    lists:foldl(
        fun({Key, R}, Acc) -> cleanup_stale_reservation(Key, R, Acc) end,
        S,
        Stale
    ).

%% =============================================================================
%% Internal: eviction
%% =============================================================================

run_eviction() ->
    run_eviction(ets:first(?TBL_LRU), 0).

run_eviction('$end_of_table', N) ->
    N;
run_eviction({_LastUsed, Key} = LruKey, N) ->
    Next = ets:next(?TBL_LRU, LruKey),
    case try_evict_one(LruKey, Key) of
        {ok, _Bytes} -> run_eviction(Next, N + 1);
        skip -> run_eviction(Next, N);
        gone -> run_eviction(Next, N)
    end.

run_eviction_bytes(Target, TierPred) ->
    run_eviction_bytes(ets:first(?TBL_LRU), Target, TierPred, 0, 0).

run_eviction_bytes('$end_of_table', _Target, _TierPred, N, Bytes) ->
    {N, Bytes};
run_eviction_bytes(_LruKey, Target, _TierPred, N, Bytes) when
    Bytes >= Target, Target > 0
->
    {N, Bytes};
run_eviction_bytes({_LastUsed, Key} = LruKey, Target, TierPred, N, Bytes) ->
    Next = ets:next(?TBL_LRU, LruKey),
    case try_evict_one(LruKey, Key, TierPred) of
        {ok, B} -> run_eviction_bytes(Next, Target, TierPred, N + 1, Bytes + B);
        skip -> run_eviction_bytes(Next, Target, TierPred, N, Bytes);
        gone -> run_eviction_bytes(Next, Target, TierPred, N, Bytes)
    end.

try_evict_one(LruKey, Key) ->
    try_evict_one(LruKey, Key, fun(_) -> true end).

try_evict_one(LruKey, Key, TierPred) ->
    case ets:lookup(?TBL_META, Key) of
        [Row] ->
            Tier = element(?POS_TIER, Row),
            case {element(?POS_STATUS, Row), element(?POS_REFCOUNT, Row), TierPred(Tier)} of
                {available, 0, true} ->
                    Location = element(?POS_LOCATION, Row),
                    Size = element(?POS_SIZE, Row),
                    delete_from_tier(Tier, Key, Location),
                    ets:delete(?TBL_LRU, LruKey),
                    ets:delete(?TBL_META, Key),
                    erllama_cache_counters:incr(?C_EVICTIONS),
                    {ok, Size};
                _ ->
                    skip
            end;
        [] ->
            ets:delete(?TBL_LRU, LruKey),
            gone
    end.

tier_pred(all) ->
    fun(_) -> true end;
tier_pred(Tiers) when is_list(Tiers) ->
    fun(T) -> lists:member(T, Tiers) end.

delete_from_tier(ram, Key, _Loc) ->
    erllama_cache_ram:delete(Key);
delete_from_tier(disk, _Key, {disk, Path}) ->
    _ = file:delete(Path),
    ok;
delete_from_tier(ram_file, _Key, {ram_file, Path}) ->
    _ = file:delete(Path),
    ok;
delete_from_tier(_, _, _) ->
    ok.

%% =============================================================================
%% Internal: helpers
%% =============================================================================

-spec monotonic_ns() -> integer().
monotonic_ns() ->
    erlang:monotonic_time(nanosecond).
