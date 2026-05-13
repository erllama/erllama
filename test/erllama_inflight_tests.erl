%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_inflight_tests).
-include_lib("eunit/include/eunit.hrl").

%% Each test starts the gen_server itself so it is not coupled to
%% the rest of the application's supervision tree. Stops cleanly so
%% subsequent tests in the suite see a fresh table + counter.
with_inflight(Body) ->
    {ok, Pid} = erllama_inflight:start_link(),
    try
        Body()
    after
        unlink(Pid),
        gen_server:stop(Pid)
    end.

queue_depth_zero_when_idle_test() ->
    with_inflight(fun() ->
        ?assertEqual(0, erllama_inflight:queue_depth())
    end).

queue_depth_bumps_on_register_test() ->
    with_inflight(fun() ->
        Ref1 = make_ref(),
        Ref2 = make_ref(),
        ok = erllama_inflight:register(Ref1, self()),
        ?assertEqual(1, erllama_inflight:queue_depth()),
        ok = erllama_inflight:register(Ref2, self()),
        ?assertEqual(2, erllama_inflight:queue_depth()),
        ok = erllama_inflight:unregister(Ref1),
        ?assertEqual(1, erllama_inflight:queue_depth()),
        ok = erllama_inflight:unregister(Ref2),
        ?assertEqual(0, erllama_inflight:queue_depth())
    end).

queue_depth_double_unregister_is_noop_test() ->
    with_inflight(fun() ->
        Ref = make_ref(),
        ok = erllama_inflight:register(Ref, self()),
        ?assertEqual(1, erllama_inflight:queue_depth()),
        ok = erllama_inflight:unregister(Ref),
        ?assertEqual(0, erllama_inflight:queue_depth()),
        ok = erllama_inflight:unregister(Ref),
        ?assertEqual(0, erllama_inflight:queue_depth())
    end).

queue_depth_decrements_on_owner_crash_test() ->
    with_inflight(fun() ->
        Owner = spawn(fun() ->
            receive
                stop -> ok
            end
        end),
        Ref1 = make_ref(),
        Ref2 = make_ref(),
        ok = erllama_inflight:register(Ref1, Owner),
        ok = erllama_inflight:register(Ref2, Owner),
        ?assertEqual(2, erllama_inflight:queue_depth()),
        %% Synchronous handoff so the gen_server has its monitor
        %% installed before we kill the owner.
        sys:get_state(erllama_inflight),
        MonRef = monitor(process, Owner),
        Owner ! stop,
        receive
            {'DOWN', MonRef, process, Owner, _} -> ok
        after 1000 ->
            error(owner_did_not_die)
        end,
        %% After the inflight gen_server processes the DOWN message,
        %% the counter must be back to 0.
        sys:get_state(erllama_inflight),
        ?assertEqual(0, erllama_inflight:queue_depth())
    end).

%% =============================================================================
%% PR2: per-model queue_depth/1
%% =============================================================================

queue_depth_per_model_returns_zero_when_idle_test() ->
    with_inflight(fun() ->
        ?assertEqual(0, erllama_inflight:queue_depth(self()))
    end).

queue_depth_per_model_counts_only_matching_pid_test() ->
    with_inflight(fun() ->
        FakeModel1 = spawn(fun() ->
            receive
                stop -> ok
            end
        end),
        FakeModel2 = spawn(fun() ->
            receive
                stop -> ok
            end
        end),
        try
            ok = erllama_inflight:register(make_ref(), FakeModel1),
            ok = erllama_inflight:register(make_ref(), FakeModel1),
            ok = erllama_inflight:register(make_ref(), FakeModel2),
            ?assertEqual(2, erllama_inflight:queue_depth(FakeModel1)),
            ?assertEqual(1, erllama_inflight:queue_depth(FakeModel2)),
            ?assertEqual(0, erllama_inflight:queue_depth(self())),
            ?assertEqual(3, erllama_inflight:queue_depth())
        after
            FakeModel1 ! stop,
            FakeModel2 ! stop
        end
    end).

%% =============================================================================
%% PR2: obs_put / obs_get / obs_delete round-trip
%% =============================================================================

obs_put_then_get_round_trip_test() ->
    with_inflight(fun() ->
        Row = {<<"m1">>, generating, 3, exact, 42},
        true = erllama_inflight:obs_put(<<"m1">>, Row),
        ?assertEqual(Row, erllama_inflight:obs_get(<<"m1">>))
    end).

obs_get_missing_returns_undefined_test() ->
    with_inflight(fun() ->
        ?assertEqual(undefined, erllama_inflight:obs_get(<<"never_put">>))
    end).

obs_delete_removes_row_test() ->
    with_inflight(fun() ->
        true = erllama_inflight:obs_put(<<"m2">>, {<<"m2">>, idle, 0, undefined, 0}),
        true = erllama_inflight:obs_delete(<<"m2">>),
        ?assertEqual(undefined, erllama_inflight:obs_get(<<"m2">>))
    end).
