-module(erllama_cache_writer_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

with_writer(Max, Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(test_disk, Dir),
    ok = erllama_cache_writer:init(Max),
    try
        Body(Dir)
    after
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

inflight_and_max_reflect_init_test() ->
    with_writer(7, fun(_Dir) ->
        ?assertEqual(7, erllama_cache_writer:max_concurrent()),
        ?assertEqual(0, erllama_cache_writer:inflight())
    end).

semaphore_returns_busy_at_capacity_test() ->
    %% Cap of 1: one save in flight blocks the next.
    with_writer(1, fun(_Dir) ->
        Parent = self(),
        Key = key_for(base_meta([1, 2, 3])),
        %% Spawn a writer that holds a reservation manually so we
        %% know exactly when one slot is in use. We can't easily
        %% pause an in-flight save at the exact "after acquire,
        %% before release" point without surgery, so we test the
        %% semaphore primitives directly via inflight() at a
        %% boundary: kick off a save in a paused worker and check
        %% the second writer:save returns busy.
        Pid =
            spawn(fun() ->
                %% Inline the same atomics path the writer uses so we
                %% can deterministically pin a slot.
                Sem = persistent_term:get({erllama_cache_writer, sem}),
                _ = atomics:add_get(Sem, 1, 1),
                Parent ! pinned,
                receive
                    release -> atomics:sub(Sem, 1, 1)
                end
            end),
        receive
            pinned -> ok
        end,
        %% Now try a real save: the semaphore is full.
        ?assertEqual(
            {error, busy},
            erllama_cache_writer:save(test_disk, disk, base_meta([1]), <<"x">>)
        ),
        Pid ! release,
        timer:sleep(20),
        %% After the pinned slot is released, the row for Key still
        %% should not exist (the busy save did not run).
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(Key))
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
