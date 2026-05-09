-module(erllama_scheduler_tests).
-behaviour(erllama_pressure).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

-export([sample/0]).

%% =============================================================================
%% Fixtures
%% =============================================================================

%% A pluggable pressure source that reads from persistent_term so tests
%% can drive the scheduler deterministically without exec'ing real
%% commands.
-define(STUB_KEY, {?MODULE, stub_pressure}).

stub_set(Used, Total) ->
    persistent_term:put(?STUB_KEY, {Used, Total}).

stub_clear() ->
    catch persistent_term:erase(?STUB_KEY).

with_subsystem(Body) ->
    ok = erllama_cache_counters:init(),
    erllama_cache_counters:reset(),
    {ok, _Meta} = erllama_cache_meta_srv:start_link(),
    {ok, _Ram} = erllama_cache_ram:start_link(),
    try
        Body()
    after
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv),
        stub_clear()
    end.

with_scheduler(Config, Body) ->
    with_subsystem(fun() ->
        {ok, _Sch} = erllama_scheduler:start_link(Config),
        try
            Body()
        after
            catch gen_server:stop(erllama_scheduler)
        end
    end).

key(N) ->
    crypto:hash(sha256, <<"sched-test-", (integer_to_binary(N))/binary>>).

insert_slab(N, Size) ->
    K = key(N),
    ok = erllama_cache_meta_srv:insert_available(K, ram, Size, <<"H">>, {ram}),
    K.

%% =============================================================================
%% Pressure source dispatch
%% =============================================================================

sample_noop_test() ->
    ?assertEqual({0, 1}, erllama_pressure:sample(noop)).

sample_module_test() ->
    stub_set(75, 100),
    try
        ?assertEqual(
            {75, 100},
            erllama_pressure:sample({module, ?MODULE})
        )
    after
        stub_clear()
    end.

%% Behaviour callback so {module, ?MODULE} works in dispatch tests.
sample() ->
    persistent_term:get(?STUB_KEY).

%% =============================================================================
%% Scheduler basics
%% =============================================================================

starts_disabled_test() ->
    with_scheduler(#{}, fun() ->
        Status = erllama_scheduler:status(),
        ?assertEqual(false, maps:get(enabled, Status)),
        ?assertEqual(noop, maps:get(pressure_source, Status))
    end).

force_check_disabled_returns_skipped_test() ->
    with_scheduler(#{}, fun() ->
        ?assertEqual({skipped, disabled}, erllama_scheduler:force_check())
    end).

force_check_below_watermark_test() ->
    stub_set(10, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.9,
            low_watermark => 0.7
        },
        fun() ->
            ?assertEqual(
                {skipped, below_watermark},
                erllama_scheduler:force_check()
            )
        end
    ).

force_check_evicts_when_above_watermark_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75,
            min_evict_bytes => 1
        },
        fun() ->
            _K1 = insert_slab(1, 5),
            _K2 = insert_slab(2, 7),
            Result = erllama_scheduler:force_check(),
            case Result of
                {evicted, N, Bytes} ->
                    ?assert(N >= 1),
                    ?assert(Bytes >= 5);
                Other ->
                    ?assertMatch({evicted, _, _}, Other)
            end
        end
    ).

force_check_above_watermark_no_slabs_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75
        },
        fun() ->
            ?assertEqual(
                {skipped, nothing_to_evict},
                erllama_scheduler:force_check()
            )
        end
    ).

enable_disable_test() ->
    with_scheduler(#{}, fun() ->
        ok = erllama_scheduler:enable(true),
        #{enabled := true} = erllama_scheduler:status(),
        ok = erllama_scheduler:enable(false),
        #{enabled := false} = erllama_scheduler:status()
    end).

set_thresholds_test() ->
    with_scheduler(#{}, fun() ->
        ok = erllama_scheduler:set_thresholds(0.92, 0.5),
        #{high_watermark := 0.92, low_watermark := 0.5} =
            erllama_scheduler:status(),
        ?assertMatch({error, _}, erllama_scheduler:set_thresholds(0.5, 0.9))
    end).

invalid_watermarks_at_init_test() ->
    process_flag(trap_exit, true),
    Cfg = #{high_watermark => 0.5, low_watermark => 0.9},
    {error, {invalid_watermarks, _}} =
        gen_server:start(erllama_scheduler, [Cfg], []),
    process_flag(trap_exit, false),
    ok.

disk_tier_skipped_by_default_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75,
            min_evict_bytes => 1
        },
        fun() ->
            DiskKey = crypto:hash(sha256, <<"sched-disk-only">>),
            ok = erllama_cache_meta_srv:insert_available(
                DiskKey, disk, 1024, <<"H">>, {disk, "/tmp/x"}
            ),
            ?assertEqual(
                {skipped, nothing_to_evict},
                erllama_scheduler:force_check()
            ),
            {ok, _Row} = erllama_cache_meta_srv:lookup_exact(DiskKey)
        end
    ).

disk_tier_evicted_when_explicit_test() ->
    stub_set(95, 100),
    with_scheduler(
        #{
            enabled => true,
            pressure_source => {module, ?MODULE},
            high_watermark => 0.85,
            low_watermark => 0.75,
            min_evict_bytes => 1,
            evict_tiers => all
        },
        fun() ->
            DiskKey = crypto:hash(sha256, <<"sched-disk-evict">>),
            ok = erllama_cache_meta_srv:insert_available(
                DiskKey, disk, 1024, <<"H">>, {disk, "/tmp/y"}
            ),
            {evicted, 1, 1024} = erllama_scheduler:force_check()
        end
    ).

sample_records_reading_test() ->
    stub_set(42, 100),
    with_scheduler(
        #{pressure_source => {module, ?MODULE}},
        fun() ->
            {42, 100} = erllama_scheduler:sample(),
            #{
                last_used := 42,
                last_total := 100,
                last_ratio := R
            } = erllama_scheduler:status(),
            ?assert(R > 0.41 andalso R < 0.43)
        end
    ).
