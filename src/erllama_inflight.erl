%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_inflight).
-moduledoc """
Tracks in-flight streaming inference requests.

Each `erllama:infer/4` admit produces a unique `reference()` and an
entry in a public ETS table mapping that reference back to the
serving `erllama_model` gen_statem. `erllama:cancel/1` looks up the
ref here to find which model owns the request, then casts a
`{cancel, Ref}` event at it.

The table is a fixed-name public ETS so lookups are lock-free from
any process. The owning gen_server (this module) is here only to
keep the table alive across releases and to clean up entries when a
model dies unexpectedly.
""".
-behaviour(gen_server).

-export([
    start_link/0,
    register/2,
    unregister/1,
    lookup/1,
    all/0,
    queue_depth/0,
    queue_depth/1,
    obs_put/2,
    obs_get/1,
    obs_delete/1
]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(TABLE, ?MODULE).
-define(OBS_TABLE, erllama_model_obs).
-define(COUNTER_KEY, {?MODULE, counter}).

%% =============================================================================
%% Public API
%% =============================================================================

-spec register(reference(), pid()) -> ok.
register(Ref, ModelPid) when is_reference(Ref), is_pid(ModelPid) ->
    %% ets:insert returns true even when the key already existed,
    %% so use ets:insert_new to bump the counter only on the row's
    %% true admission. A duplicate Ref is a programming error
    %% (references are unique per `make_ref/0`); we treat the
    %% second insert as a no-op for the counter.
    case ets:insert_new(?TABLE, {Ref, ModelPid}) of
        true ->
            counter_add(1);
        false ->
            ok
    end,
    %% Monitor the model so a crash purges its inflight entries.
    %% The monitor is held by the gen_server, not by callers.
    gen_server:cast(?MODULE, {monitor_model, ModelPid}),
    ok.

-spec unregister(reference()) -> ok.
unregister(Ref) when is_reference(Ref) ->
    %% ets:take returns the deleted row(s), so we decrement only
    %% when something was actually removed. A double-unregister
    %% then becomes a no-op and the counter stays in lockstep
    %% with the table.
    case ets:take(?TABLE, Ref) of
        [_] -> counter_add(-1);
        [] -> ok
    end,
    ok.

-spec lookup(reference()) -> {ok, pid()} | {error, not_found}.
lookup(Ref) when is_reference(Ref) ->
    case ets:lookup(?TABLE, Ref) of
        [{Ref, Pid}] -> {ok, Pid};
        [] -> {error, not_found}
    end.

-spec all() -> [{reference(), pid()}].
all() ->
    ets:tab2list(?TABLE).

-doc """
O(1) snapshot of currently-registered inflight rows. Reads the
atomics counter parked in `persistent_term`; returns 0 when the
gen_server has not been started yet (the counter does not exist
in that case).

Counts only admitted streaming requests. Pending FIFO requests
queued inside an individual model gen_statem (admitted to the
mailbox but not yet streaming) are not visible here.
""".
-spec queue_depth() -> non_neg_integer().
queue_depth() ->
    case persistent_term:get(?COUNTER_KEY, undefined) of
        undefined -> 0;
        Ref -> counters:get(Ref, 1)
    end.

-doc """
Per-model inflight count. Walks the inflight table filtering on
the model's pid; O(N) in the table size but accurate for a single
model. The global `queue_depth/0` is still O(1) via the atomics
counter; use that for cross-model totals.
""".
-spec queue_depth(pid()) -> non_neg_integer().
queue_depth(Pid) when is_pid(Pid) ->
    case ets:whereis(?TABLE) of
        undefined ->
            0;
        _ ->
            ets:select_count(?TABLE, [{{'_', Pid}, [], [true]}])
    end.

counter_add(N) ->
    case persistent_term:get(?COUNTER_KEY, undefined) of
        undefined -> ok;
        Ref -> counters:add(Ref, 1, N)
    end.

%% =============================================================================
%% Per-model observability snapshot
%% =============================================================================
%%
%% A second named public ETS table keyed by `model_id()` carrying the
%% per-model phase, pending FIFO depth, and last cache-hit summary.
%% Each model gen_statem is the sole writer for its own row; reads
%% are lock-free from any process (including remote nodes via erpc).
%%
%% Row shape: `{ModelId, Phase, PendingLen, LastCacheHitKind, LastCacheHitPrefixLen}`
%% where Phase is `idle | prefilling | generating`, LastCacheHitKind
%% is `exact | partial | cold | undefined`, and LastCacheHitPrefixLen
%% is the warm prefix token count (0 for cold or undefined).

-spec obs_put(binary(), tuple()) -> true.
obs_put(ModelId, Row) when is_binary(ModelId), is_tuple(Row) ->
    %% Tolerant of a missing table: when `erllama_inflight` has not
    %% been started (some unit-test fixtures bring up the model
    %% gen_statem directly without the cache supervisor), writes
    %% are silently dropped rather than crashing the writer. The
    %% model still functions; the observability snapshot just
    %% isn't available.
    case ets:whereis(?OBS_TABLE) of
        undefined -> true;
        _ -> ets:insert(?OBS_TABLE, Row)
    end.

-spec obs_get(binary()) -> tuple() | undefined.
obs_get(ModelId) when is_binary(ModelId) ->
    case ets:whereis(?OBS_TABLE) of
        undefined ->
            undefined;
        _ ->
            case ets:lookup(?OBS_TABLE, ModelId) of
                [Row] -> Row;
                [] -> undefined
            end
    end.

-spec obs_delete(binary()) -> true.
obs_delete(ModelId) when is_binary(ModelId) ->
    case ets:whereis(?OBS_TABLE) of
        undefined -> true;
        _ -> ets:delete(?OBS_TABLE, ModelId)
    end.

%% =============================================================================
%% gen_server
%% =============================================================================

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    Opts = [
        named_table,
        public,
        set,
        {read_concurrency, true},
        {write_concurrency, true}
    ],
    _ = ets:new(?TABLE, Opts),
    %% Per-model observability snapshot. Same public-ETS pattern as
    %% the inflight table above; the model gen_statem is the sole
    %% writer for its own row.
    _ = ets:new(?OBS_TABLE, Opts),
    %% Park a fresh counter ref in persistent_term so any process
    %% (including remote nodes via erpc, used by erllama_cluster)
    %% can read queue_depth/0 without crossing this gen_server.
    %% Re-creating on init resets the depth to 0, matching the
    %% empty ETS table.
    Counter = counters:new(1, [atomics]),
    persistent_term:put(?COUNTER_KEY, Counter),
    {ok, #{monitors => #{}}}.

handle_call(_, _, S) -> {reply, {error, unknown_call}, S}.

handle_cast({monitor_model, Pid}, S = #{monitors := M}) ->
    case maps:is_key(Pid, M) of
        true ->
            {noreply, S};
        false ->
            Ref = monitor(process, Pid),
            {noreply, S#{monitors := M#{Pid => Ref}}}
    end;
handle_cast(_, S) ->
    {noreply, S}.

handle_info({'DOWN', _MonRef, process, Pid, _Reason}, S = #{monitors := M}) ->
    %% Sweep all inflight entries owned by this dead model and
    %% decrement the queue-depth counter by the number of rows
    %% removed. The counter must stay in lockstep with the ETS
    %% table across this path or queue_depth/0 drifts up
    %% permanently every time a model crashes.
    Refs = [R || {R, P} <- ets:tab2list(?TABLE), P =:= Pid],
    [ets:delete(?TABLE, R) || R <- Refs],
    case Refs of
        [] -> ok;
        _ -> counter_add(-length(Refs))
    end,
    {noreply, S#{monitors := maps:remove(Pid, M)}};
handle_info(_, S) ->
    {noreply, S}.

terminate(_, _) -> ok.
