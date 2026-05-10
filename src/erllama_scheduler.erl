%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Memory-pressure-driven cache eviction.
%%
%% Periodically polls a pluggable pressure source (`erllama_pressure`)
%% and, when the used/total ratio crosses `high_watermark`, asks the
%% cache to evict slabs until the ratio would drop below
%% `low_watermark`. The eviction call is `erllama_cache:evict_bytes/2`
%% with a target of `(high - low) * Total` bytes; the cache may free
%% less if no evictable slabs remain.
%%
%% Tier policy: by default the scheduler evicts only `ram` and
%% `ram_file` slabs. Disk-tier slabs are left in place — disk is the
%% cheap tier, and the deployment usually wants to keep as much warm
%% state as possible there. Disk eviction happens via the cache's own
%% per-tier quota or via an explicit `erllama_cache:gc/0` call.
%% Override with `evict_tiers => all` (or a custom list) to include
%% disk in scheduler-driven eviction.
%%
%% Disabled by default. Enable via the `erllama` app environment:
%%
%% ```
%% {erllama, [
%%   {scheduler, #{
%%     enabled         => true,
%%     pressure_source => system,    %% noop | system | nvidia_smi | {module, M}
%%     interval_ms     => 5000,
%%     high_watermark  => 0.85,
%%     low_watermark   => 0.75,
%%     min_evict_bytes => 1048576,   %% don't bother with sub-MB targets
%%     evict_tiers     => [ram, ram_file]
%%   }}
%% ]}
%% ```
%%
%% The scheduler always starts (so it can be enabled at runtime via
%% `enable/1`), but its timer only fires when `enabled = true`.
%% @end
-module(erllama_scheduler).
-behaviour(gen_server).

-export([
    start_link/0,
    start_link/1,
    enable/1,
    set_pressure_source/1,
    set_thresholds/2,
    sample/0,
    force_check/0,
    status/0
]).

-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(SERVER, ?MODULE).
-define(DEFAULT_INTERVAL_MS, 5000).
-define(DEFAULT_HIGH, 0.85).
-define(DEFAULT_LOW, 0.75).
-define(DEFAULT_MIN_EVICT, 1024 * 1024).
-define(DEFAULT_EVICT_TIERS, [ram, ram_file]).

-record(state, {
    enabled :: boolean(),
    pressure_source :: erllama_pressure:source(),
    interval_ms :: pos_integer(),
    high_watermark :: float(),
    low_watermark :: float(),
    min_evict_bytes :: non_neg_integer(),
    evict_tiers :: all | [erllama_cache:tier()],
    timer_ref :: reference() | undefined,
    last_used :: non_neg_integer(),
    last_total :: non_neg_integer(),
    last_evicted_bytes :: non_neg_integer(),
    last_evicted_at :: integer() | undefined,
    sampled_at :: integer() | undefined
}).

-type state() :: #state{}.
-type config() :: #{
    enabled => boolean(),
    pressure_source => erllama_pressure:source(),
    interval_ms => pos_integer(),
    high_watermark => float(),
    low_watermark => float(),
    min_evict_bytes => non_neg_integer(),
    evict_tiers => all | [erllama_cache:tier()]
}.

-export_type([config/0]).

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    start_link(env_config()).

-spec start_link(config()) -> {ok, pid()} | {error, term()}.
start_link(Config) ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [Config], []).

-spec enable(boolean()) -> ok.
enable(Bool) when is_boolean(Bool) ->
    gen_server:call(?SERVER, {enable, Bool}).

-spec set_pressure_source(erllama_pressure:source()) -> ok.
set_pressure_source(Source) ->
    gen_server:call(?SERVER, {set_source, Source}).

-spec set_thresholds(float(), float()) -> ok | {error, term()}.
set_thresholds(High, Low) when
    is_number(High),
    is_number(Low),
    High > Low,
    High =< 1.0,
    Low >= 0.0
->
    gen_server:call(?SERVER, {set_thresholds, High, Low});
set_thresholds(_, _) ->
    {error, bad_thresholds}.

%% @doc Take a single pressure sample without acting on it. Returns
%% the most recent reading.
-spec sample() -> erllama_pressure:reading().
sample() ->
    gen_server:call(?SERVER, sample).

%% @doc Force a check now (sample + maybe evict). Returns the eviction
%% result if one was triggered, `{skipped, Reason}` otherwise.
-spec force_check() ->
    {evicted, non_neg_integer(), non_neg_integer()}
    | {skipped, below_watermark | disabled | nothing_to_evict}.
force_check() ->
    gen_server:call(?SERVER, force_check).

-spec status() -> map().
status() ->
    gen_server:call(?SERVER, status).

%% =============================================================================
%% gen_server callbacks
%% =============================================================================

-spec init([config()]) -> {ok, state()} | {stop, term()}.
init([Config]) ->
    case validate_config(Config) of
        {error, Reason} ->
            {stop, Reason};
        ok ->
            Source = maps:get(pressure_source, Config, noop),
            maybe_start_os_mon(Source),
            S0 = #state{
                enabled = maps:get(enabled, Config, false),
                pressure_source = Source,
                interval_ms = maps:get(interval_ms, Config, ?DEFAULT_INTERVAL_MS),
                high_watermark = maps:get(high_watermark, Config, ?DEFAULT_HIGH),
                low_watermark = maps:get(low_watermark, Config, ?DEFAULT_LOW),
                min_evict_bytes = maps:get(min_evict_bytes, Config, ?DEFAULT_MIN_EVICT),
                evict_tiers = maps:get(evict_tiers, Config, ?DEFAULT_EVICT_TIERS),
                last_used = 0,
                last_total = 1,
                last_evicted_bytes = 0
            },
            {ok, schedule_next(S0)}
    end.

handle_call({enable, Bool}, _From, S) ->
    S1 = S#state{enabled = Bool},
    {reply, ok, schedule_next(cancel_timer(S1))};
handle_call({set_source, Source}, _From, S) ->
    maybe_start_os_mon(Source),
    {reply, ok, S#state{pressure_source = Source}};
handle_call({set_thresholds, High, Low}, _From, S) ->
    {reply, ok, S#state{high_watermark = High, low_watermark = Low}};
handle_call(sample, _From, S) ->
    {Used, Total} = erllama_pressure:sample(S#state.pressure_source),
    S1 = S#state{
        last_used = Used,
        last_total = Total,
        sampled_at = monotonic_ns()
    },
    {reply, {Used, Total}, S1};
handle_call(force_check, _From, S) ->
    {Result, S1} = check_once(S),
    {reply, Result, S1};
handle_call(status, _From, S) ->
    {reply, snapshot(S), S};
handle_call(_Msg, _From, S) ->
    {reply, {error, unknown_call}, S}.

handle_cast(_Msg, S) ->
    {noreply, S}.

handle_info(tick, S) ->
    {_Result, S1} = check_once(S),
    {noreply, schedule_next(S1)};
handle_info(_Msg, S) ->
    {noreply, S}.

terminate(_Reason, _S) ->
    ok.

%% =============================================================================
%% Internal
%% =============================================================================

env_config() ->
    case application:get_env(erllama, scheduler) of
        {ok, M} when is_map(M) -> M;
        _ -> #{}
    end.

%% Validate raw config map before constructing the state record.
%% Defaults pass through unchanged; only user-supplied values are
%% type-checked. We validate before record construction so dialyzer's
%% pos_integer()/float() field types stay accurate.
validate_config(Cfg) ->
    H = maps:get(high_watermark, Cfg, ?DEFAULT_HIGH),
    L = maps:get(low_watermark, Cfg, ?DEFAULT_LOW),
    I = maps:get(interval_ms, Cfg, ?DEFAULT_INTERVAL_MS),
    case watermarks_ok(H, L) of
        false ->
            {error, {invalid_config, {watermarks, "require 0.0 <= low < high <= 1.0"}}};
        true ->
            case is_integer(I) andalso I > 0 of
                true ->
                    ok;
                false ->
                    {error, {invalid_config, {interval_ms, "must be a positive integer"}}}
            end
    end.

watermarks_ok(H, L) when is_number(H), is_number(L), H > L, H =< 1.0, L >= 0.0 ->
    true;
watermarks_ok(_, _) ->
    false.

maybe_start_os_mon(system) ->
    _ = application:ensure_all_started(os_mon),
    ok;
maybe_start_os_mon(_) ->
    ok.

cancel_timer(#state{timer_ref = undefined} = S) ->
    S;
cancel_timer(#state{timer_ref = Ref} = S) ->
    _ = erlang:cancel_timer(Ref),
    S#state{timer_ref = undefined}.

schedule_next(#state{enabled = false} = S) ->
    cancel_timer(S);
schedule_next(#state{interval_ms = Ms} = S) ->
    S1 = cancel_timer(S),
    Ref = erlang:send_after(Ms, self(), tick),
    S1#state{timer_ref = Ref}.

check_once(#state{enabled = false} = S) ->
    {{skipped, disabled}, S};
check_once(#state{pressure_source = Src} = S) ->
    {Used, Total} = erllama_pressure:sample(Src),
    NowNs = monotonic_ns(),
    S1 = S#state{
        last_used = Used,
        last_total = Total,
        sampled_at = NowNs
    },
    case Total of
        0 ->
            {{skipped, below_watermark}, S1};
        _ ->
            maybe_evict(Used, Total, S1, NowNs)
    end.

maybe_evict(Used, Total, S, NowNs) ->
    Ratio = Used / Total,
    case Ratio >= S#state.high_watermark of
        false ->
            {{skipped, below_watermark}, S};
        true ->
            Target = trunc((S#state.high_watermark - S#state.low_watermark) * Total),
            do_evict(max(Target, S#state.min_evict_bytes), S, NowNs)
    end.

do_evict(Target, S, _NowNs) when Target =< 0 ->
    {{skipped, below_watermark}, S};
do_evict(Target, S, NowNs) ->
    case erllama_cache:evict_bytes(Target, S#state.evict_tiers) of
        {evicted, 0, 0} ->
            {{skipped, nothing_to_evict}, S};
        {evicted, _N, Bytes} = R ->
            S1 = S#state{
                last_evicted_bytes = Bytes,
                last_evicted_at = NowNs
            },
            {R, S1}
    end.

snapshot(S) ->
    Total = max(S#state.last_total, 1),
    Ratio = S#state.last_used / Total,
    #{
        enabled => S#state.enabled,
        pressure_source => S#state.pressure_source,
        interval_ms => S#state.interval_ms,
        high_watermark => S#state.high_watermark,
        low_watermark => S#state.low_watermark,
        min_evict_bytes => S#state.min_evict_bytes,
        evict_tiers => S#state.evict_tiers,
        last_used => S#state.last_used,
        last_total => S#state.last_total,
        last_ratio => Ratio,
        last_evicted_bytes => S#state.last_evicted_bytes,
        sampled_at => S#state.sampled_at,
        last_evicted_at => S#state.last_evicted_at
    }.

monotonic_ns() ->
    erlang:monotonic_time(nanosecond).
