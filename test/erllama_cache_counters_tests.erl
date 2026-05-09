-module(erllama_cache_counters_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

with_counters(Body) ->
    ok = erllama_cache_counters:init(),
    erllama_cache_counters:reset(),
    Body().

%% =============================================================================
%% Primitives
%% =============================================================================

init_creates_zeroed_array_test() ->
    with_counters(fun() ->
        Snapshot = erllama_cache_counters:snapshot(),
        ?assert(map_size(Snapshot) >= 16),
        ?assertEqual(0, maps:get(hits_exact, Snapshot))
    end).

incr_bumps_slot_test() ->
    with_counters(fun() ->
        ok = erllama_cache_counters:incr(?C_HITS_EXACT),
        ok = erllama_cache_counters:incr(?C_HITS_EXACT),
        ?assertEqual(2, erllama_cache_counters:get(?C_HITS_EXACT))
    end).

add_increments_by_amount_test() ->
    with_counters(fun() ->
        ok = erllama_cache_counters:add(?C_PACK_TOTAL_NS, 1234),
        ok = erllama_cache_counters:add(?C_PACK_TOTAL_NS, 5678),
        ?assertEqual(6912, erllama_cache_counters:get(?C_PACK_TOTAL_NS))
    end).

snapshot_returns_named_map_test() ->
    with_counters(fun() ->
        ok = erllama_cache_counters:incr(?C_MISSES),
        ok = erllama_cache_counters:incr(?C_SAVES_COLD),
        ok = erllama_cache_counters:incr(?C_SAVES_COLD),
        Snap = erllama_cache_counters:snapshot(),
        ?assertEqual(1, maps:get(misses, Snap)),
        ?assertEqual(2, maps:get(saves_cold, Snap)),
        ?assertEqual(0, maps:get(saves_continued, Snap))
    end).

reset_zeroes_all_slots_test() ->
    with_counters(fun() ->
        ok = erllama_cache_counters:incr(?C_HITS_EXACT),
        ok = erllama_cache_counters:incr(?C_SAVES_FINISH),
        ok = erllama_cache_counters:reset(),
        ?assertEqual(0, erllama_cache_counters:get(?C_HITS_EXACT)),
        ?assertEqual(0, erllama_cache_counters:get(?C_SAVES_FINISH))
    end).

%% =============================================================================
%% Public façade
%% =============================================================================

facade_get_counters_returns_snapshot_test() ->
    with_counters(fun() ->
        ok = erllama_cache_counters:incr(?C_HITS_EXACT),
        Snap = erllama_cache:get_counters(),
        ?assertEqual(1, maps:get(hits_exact, Snap))
    end).

facade_reset_counters_zeroes_test() ->
    with_counters(fun() ->
        ok = erllama_cache_counters:incr(?C_HITS_EXACT),
        ok = erllama_cache:reset_counters(),
        ?assertEqual(0, erllama_cache_counters:get(?C_HITS_EXACT))
    end).
