%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_ram_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

with_cache(Body) ->
    {ok, _M} = erllama_cache_meta_srv:start_link(),
    {ok, _R} = erllama_cache_ram:start_link(),
    try
        Body()
    after
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv)
    end.

key(N) ->
    crypto:hash(sha256, integer_to_binary(N)).

put_then_load_returns_slab_test() ->
    with_cache(fun() ->
        Slab = <<"hello slab">>,
        ok = erllama_cache_ram:put(key(1), Slab, <<"H">>),
        ?assertEqual({ok, Slab}, erllama_cache_ram:load(key(1)))
    end).

put_registers_meta_row_test() ->
    with_cache(fun() ->
        Slab = <<"x">>,
        ok = erllama_cache_ram:put(key(1), Slab, <<"H">>),
        {ok, Row} = erllama_cache_meta_srv:lookup_exact(key(1)),
        ?assertEqual(ram, element(?POS_TIER, Row)),
        ?assertEqual(1, element(?POS_SIZE, Row))
    end).

load_miss_unknown_key_test() ->
    with_cache(fun() ->
        ?assertEqual(miss, erllama_cache_ram:load(key(99)))
    end).

delete_removes_slab_test() ->
    with_cache(fun() ->
        ok = erllama_cache_ram:put(key(1), <<"x">>, <<"H">>),
        ok = erllama_cache_ram:delete(key(1)),
        ?assertEqual(miss, erllama_cache_ram:load(key(1)))
    end).

size_bytes_aggregates_test() ->
    with_cache(fun() ->
        ok = erllama_cache_ram:put(key(1), <<"abc">>, <<"H">>),
        ok = erllama_cache_ram:put(key(2), <<"defgh">>, <<"H">>),
        ?assertEqual(8, erllama_cache_ram:size_bytes())
    end).

size_bytes_decrements_after_delete_test() ->
    with_cache(fun() ->
        ok = erllama_cache_ram:put(key(1), <<"abc">>, <<"H">>),
        ok = erllama_cache_ram:put(key(2), <<"defgh">>, <<"H">>),
        ok = erllama_cache_ram:delete(key(1)),
        ?assertEqual(5, erllama_cache_ram:size_bytes())
    end).

%% =============================================================================
%% End-to-end: meta_srv eviction also clears the slab from RAM
%% =============================================================================

eviction_removes_slab_too_test() ->
    with_cache(fun() ->
        ok = erllama_cache_ram:put(key(1), <<"abc">>, <<"H">>),
        ok = erllama_cache_ram:put(key(2), <<"defgh">>, <<"H">>),
        {evicted, 2} = erllama_cache_meta_srv:gc(),
        ?assertEqual(miss, erllama_cache_ram:load(key(1))),
        ?assertEqual(miss, erllama_cache_ram:load(key(2))),
        ?assertEqual(0, erllama_cache_ram:size_bytes())
    end).

eviction_skips_referenced_slab_test() ->
    with_cache(fun() ->
        ok = erllama_cache_ram:put(key(1), <<"abc">>, <<"H">>),
        ok = erllama_cache_ram:put(key(2), <<"defgh">>, <<"H">>),
        {ok, _Ref, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
        {evicted, 1} = erllama_cache_meta_srv:gc(),
        ?assertMatch({ok, _}, erllama_cache_ram:load(key(1))),
        ?assertEqual(miss, erllama_cache_ram:load(key(2)))
    end).
