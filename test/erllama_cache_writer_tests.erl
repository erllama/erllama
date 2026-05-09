-module(erllama_cache_writer_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

with_writer(Max, Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(test_disk, Dir),
    {ok, _} = erllama_cache_writer:start_link(Max),
    try
        Body(Dir)
    after
        catch gen_server:stop(erllama_cache_writer),
        catch gen_server:stop(test_disk),
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv),
        rm_rf(Dir)
    end.

base_meta(Tokens) ->
    #{
        save_reason => cold,
        quant_bits => 16,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => Tokens,
        context_size => 4096,
        creation_time => 1000,
        last_used_time => 1000,
        hit_count => 0,
        prompt_text => <<>>,
        hostname => <<"test">>,
        erllama_version => <<"0.1.0">>
    }.

key_for(Meta) ->
    erllama_cache_key:make(#{
        fingerprint => maps:get(fingerprint, Meta),
        quant_type => maps:get(quant_type, Meta),
        ctx_params_hash => maps:get(ctx_params_hash, Meta),
        tokens => maps:get(tokens, Meta)
    }).

%% =============================================================================
%% Reservation protocol round-trip
%% =============================================================================

save_publishes_via_reservation_protocol_test() ->
    with_writer(4, fun(_Dir) ->
        Meta = base_meta([1, 2, 3]),
        Payload = <<"first save">>,
        {ok, Key} = erllama_cache_writer:save(test_disk, disk, Meta, Payload),
        ?assertEqual(Key, key_for(Meta)),
        %% After the writer returns the row should be `available` and
        %% load should succeed.
        {ok, Row} = erllama_cache_meta_srv:lookup_exact(Key),
        ?assertEqual(disk, element(?POS_TIER, Row)),
        ?assertMatch({disk, _Path}, element(?POS_LOCATION, Row)),
        {ok, _Info, P} = erllama_cache_disk_srv:load(test_disk, Key),
        ?assertEqual(Payload, P)
    end).

save_already_present_returns_error_test() ->
    with_writer(4, fun(_Dir) ->
        Meta = base_meta([1, 2, 3]),
        {ok, _Key} = erllama_cache_writer:save(test_disk, disk, Meta, <<"x">>),
        %% Second save for the same Key: meta_srv:reserve_save returns
        %% {error, already_present}; writer propagates it.
        ?assertEqual(
            {error, already_present},
            erllama_cache_writer:save(test_disk, disk, Meta, <<"x">>)
        )
    end).

%% =============================================================================
%% Semaphore: capacity, busy, inflight
%% =============================================================================

current_and_max_reflect_init_test() ->
    with_writer(7, fun(_Dir) ->
        ?assertEqual(7, erllama_cache_writer:max_concurrent()),
        ?assertEqual(0, erllama_cache_writer:current())
    end).

semaphore_returns_max_concurrent_on_timeout_test() ->
    %% Cap of 1, deterministic: pin the slot via a direct acquire,
    %% then try a save with a short acquire timeout.
    with_writer(1, fun(_Dir) ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                ok = erllama_cache_writer:acquire(infinity),
                Parent ! pinned,
                receive
                    release -> erllama_cache_writer:release()
                end
            end),
        receive
            pinned -> ok
        end,
        Key = key_for(base_meta([1, 2, 3])),
        ?assertEqual(
            {error, max_concurrent},
            erllama_cache_writer:save(
                test_disk, disk, base_meta([1, 2, 3]), <<"x">>, 30
            )
        ),
        Pid ! release,
        wait_until(fun() -> erllama_cache_writer:current() =:= 0 end, 1000),
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(Key))
    end).

release_saturates_at_zero_test() ->
    with_writer(2, fun(_Dir) ->
        %% Double-release: the saturating decrement clamps at 0.
        ok = erllama_cache_writer:release(),
        ok = erllama_cache_writer:release(),
        ?assertEqual(0, erllama_cache_writer:current())
    end).

down_releases_slot_no_leak_test() ->
    with_writer(1, fun(_Dir) ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                ok = erllama_cache_writer:acquire(infinity),
                Parent ! pinned,
                receive
                    forever -> ok
                end
            end),
        receive
            pinned -> ok
        end,
        ?assertEqual(1, erllama_cache_writer:current()),
        %% Kill the holder without releasing.
        exit(Pid, kill),
        wait_until(fun() -> erllama_cache_writer:current() =:= 0 end, 1000),
        %% A subsequent acquire should succeed.
        ok = erllama_cache_writer:acquire(100),
        erllama_cache_writer:release()
    end).

set_max_concurrent_takes_effect_test() ->
    with_writer(2, fun(_Dir) ->
        ok = erllama_cache_writer:set_max_concurrent(5),
        ?assertEqual(5, erllama_cache_writer:max_concurrent())
    end).

%% =============================================================================
%% Failure paths cancel the reservation
%% =============================================================================

bad_meta_cancels_reservation_test() ->
    with_writer(2, fun(_Dir) ->
        %% Missing required key in the build meta: kvc:build returns
        %% {error, {missing_key, _}}, which the writer treats as a
        %% disk_srv save failure. The reservation must be cancelled
        %% so a subsequent save with valid meta succeeds.
        BadMeta = maps:remove(fingerprint, base_meta([1, 2, 3])),
        ?assertMatch(
            {error, _},
            erllama_cache_writer:save(test_disk, disk, BadMeta, <<"x">>)
        ),
        %% Verify the reservation was cancelled (not stuck `writing`).
        BadKey = erllama_cache_key:make(#{
            fingerprint => binary:copy(<<16#AA>>, 32),
            quant_type => f16,
            ctx_params_hash => binary:copy(<<16#BB>>, 32),
            tokens => [1, 2, 3]
        }),
        ?assertEqual(miss, erllama_cache_meta_srv:dump(BadKey))
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

wait_until(Pred, TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_until_loop(Pred, Deadline).

wait_until_loop(Pred, Deadline) ->
    case Pred() of
        true ->
            ok;
        false ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    erlang:error(timeout);
                false ->
                    timer:sleep(10),
                    wait_until_loop(Pred, Deadline)
            end
    end.

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_cache_writer_tests_" ++
            integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = file:make_dir(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} -> [file:delete(filename:join(Dir, E)) || E <- Entries];
        _ -> ok
    end,
    file:del_dir(Dir).
