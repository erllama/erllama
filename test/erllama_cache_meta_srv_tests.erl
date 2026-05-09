-module(erllama_cache_meta_srv_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_srv(Body) ->
    {ok, _Pid} = erllama_cache_meta_srv:start_link(),
    try
        Body()
    after
        catch gen_server:stop(erllama_cache_meta_srv)
    end.

key(N) ->
    crypto:hash(sha256, integer_to_binary(N)).

base_meta(Key, Tokens) ->
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
        erllama_version => <<"0.1.0">>,
        cache_key => Key
    }.

build_kvc_file(Path, Key, Tokens, Payload) ->
    Meta = base_meta(Key, Tokens),
    %% The cache_key the file actually carries is derived from its
    %% contents, not the Key argument; pass through both so the
    %% caller can choose whether to mismatch them in negative tests.
    {ok, Prefix} = erllama_cache_kvc:build(maps:remove(cache_key, Meta), Payload),
    file:write_file(Path, <<Prefix/binary, Payload/binary>>).

%% =============================================================================
%% Read-only lookup
%% =============================================================================

lookup_exact_miss_test() ->
    with_srv(fun() ->
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(key(1)))
    end).

insert_available_makes_lookup_hit_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H">>),
        {ok, Row} = erllama_cache_meta_srv:lookup_exact(key(1)),
        ?assertEqual(key(1), element(?POS_KEY, Row)),
        ?assertEqual(ram, element(?POS_TIER, Row)),
        ?assertEqual(available, element(?POS_STATUS, Row)),
        ?assertEqual(0, element(?POS_REFCOUNT, Row))
    end).

%% =============================================================================
%% Checkout / checkin
%% =============================================================================

checkout_increments_refcount_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H">>),
        {ok, Ref, ram, _Loc, <<"H">>, _Tokens} =
            erllama_cache_meta_srv:checkout(key(1), self()),
        {ok, Row} = erllama_cache_meta_srv:dump(key(1)),
        ?assertEqual(1, element(?POS_REFCOUNT, Row)),
        ok = erllama_cache_meta_srv:checkin(Ref),
        {ok, Row2} = erllama_cache_meta_srv:dump(key(1)),
        ?assertEqual(0, element(?POS_REFCOUNT, Row2))
    end).

checkout_misses_unknown_key_test() ->
    with_srv(fun() ->
        ?assertEqual(miss, erllama_cache_meta_srv:checkout(key(999), self()))
    end).

checkout_two_holders_independent_refcounts_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H">>),
        Parent = self(),
        Pid =
            spawn(fun() ->
                {ok, Ref, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
                Parent ! {claimed, Ref},
                receive
                    release -> ok = erllama_cache_meta_srv:checkin(Ref)
                end,
                Parent ! released
            end),
        {ok, Ref1, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
        receive
            {claimed, _} -> ok
        end,
        {ok, Row} = erllama_cache_meta_srv:dump(key(1)),
        ?assertEqual(2, element(?POS_REFCOUNT, Row)),
        Pid ! release,
        receive
            released -> ok
        end,
        {ok, Row2} = erllama_cache_meta_srv:dump(key(1)),
        ?assertEqual(1, element(?POS_REFCOUNT, Row2)),
        ok = erllama_cache_meta_srv:checkin(Ref1)
    end).

checkin_unknown_ref_is_noop_test() ->
    with_srv(fun() ->
        ?assertEqual(ok, erllama_cache_meta_srv:checkin(make_ref()))
    end).

down_decrements_refcount_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H">>),
        Parent = self(),
        _Pid =
            spawn(fun() ->
                {ok, _Ref, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
                Parent ! claimed,
                exit(normal)
            end),
        receive
            claimed -> ok
        end,
        wait_for_refcount(key(1), 0, 2000),
        ok
    end).

%% =============================================================================
%% Reservation state machine
%% =============================================================================

reserve_creates_writing_row_test() ->
    with_srv(fun() ->
        {ok, _Token} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
        {ok, Row} = erllama_cache_meta_srv:dump(key(1)),
        ?assertEqual(writing, element(?POS_STATUS, Row))
    end).

reserve_rejects_already_present_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H">>),
        ?assertEqual(
            {error, already_present},
            erllama_cache_meta_srv:reserve_save(key(1), disk, self())
        )
    end).

reserve_rejects_concurrent_writer_test() ->
    with_srv(fun() ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                {ok, _T} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
                Parent ! reserved,
                receive
                    stop -> ok
                end
            end),
        receive
            reserved -> ok
        end,
        ?assertEqual(
            {error, conflict},
            erllama_cache_meta_srv:reserve_save(key(1), disk, self())
        ),
        Pid ! stop
    end).

check_reservation_ok_with_valid_token_test() ->
    with_srv(fun() ->
        {ok, Token} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
        ?assertEqual(ok, erllama_cache_meta_srv:check_reservation(key(1), Token))
    end).

check_reservation_expired_for_unknown_token_test() ->
    with_srv(fun() ->
        {ok, _Token} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
        ?assertEqual(
            {error, expired},
            erllama_cache_meta_srv:check_reservation(key(1), make_ref())
        )
    end).

mark_published_advances_stage_test() ->
    with_srv(fun() ->
        {ok, Token} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
        ?assertEqual(
            ok,
            erllama_cache_meta_srv:mark_published(key(1), Token, "/tmp/whatever")
        )
    end).

announce_saved_promotes_to_available_test() ->
    with_srv(fun() ->
        {ok, Token} = erllama_cache_meta_srv:reserve_save(key(1), ram, self()),
        ok = erllama_cache_meta_srv:announce_saved(key(1), Token, 100, <<"H">>),
        {ok, Row} = erllama_cache_meta_srv:dump(key(1)),
        ?assertEqual(available, element(?POS_STATUS, Row)),
        ?assertEqual(<<"H">>, element(?POS_HEADER_BIN, Row))
    end).

cancel_reservation_clears_row_test() ->
    with_srv(fun() ->
        {ok, Token} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
        ok = erllama_cache_meta_srv:cancel_reservation(key(1), Token),
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(key(1))),
        ?assertEqual(miss, erllama_cache_meta_srv:dump(key(1)))
    end).

%% =============================================================================
%% Reservation cleanup on writer DOWN
%% =============================================================================

down_pre_link_clears_placeholder_test() ->
    with_srv(fun() ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                {ok, _T} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
                Parent ! reserved
            end),
        receive
            reserved -> ok
        end,
        wait_for_pid_dead(Pid, 1000),
        wait_for_row_gone(key(1), 2000)
    end).

down_post_link_with_valid_file_adopts_test() ->
    with_srv(fun() ->
        TmpDir = make_tmp_dir(),
        try
            Tokens = [1, 2, 3, 4, 5, 6, 7, 8],
            Payload = <<"x">>,
            CKey = erllama_cache_key:make(#{
                fingerprint => binary:copy(<<16#AA>>, 32),
                quant_type => f16,
                ctx_params_hash => binary:copy(<<16#BB>>, 32),
                tokens => Tokens
            }),
            Path = filename:join(TmpDir, "adopt.kvc"),
            ok = build_kvc_file(Path, CKey, Tokens, Payload),
            Parent = self(),
            Pid =
                spawn(fun() ->
                    {ok, T} = erllama_cache_meta_srv:reserve_save(CKey, disk, self()),
                    ok = erllama_cache_meta_srv:mark_published(CKey, T, Path),
                    Parent ! ready
                end),
            receive
                ready -> ok
            end,
            wait_for_pid_dead(Pid, 1000),
            wait_until(
                fun() ->
                    case erllama_cache_meta_srv:lookup_exact(CKey) of
                        {ok, _} -> true;
                        _ -> false
                    end
                end,
                2000
            )
        after
            rm_rf(TmpDir)
        end
    end).

down_post_link_with_invalid_file_deletes_and_clears_test() ->
    with_srv(fun() ->
        TmpDir = make_tmp_dir(),
        try
            Path = filename:join(TmpDir, "bad.kvc"),
            file:write_file(Path, <<"not a kvc file">>),
            Parent = self(),
            Pid =
                spawn(fun() ->
                    {ok, T} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
                    ok = erllama_cache_meta_srv:mark_published(key(1), T, Path),
                    Parent ! ready
                end),
            receive
                ready -> ok
            end,
            wait_for_pid_dead(Pid, 1000),
            wait_for_row_gone(key(1), 2000),
            ?assertEqual({error, enoent}, file:read_file_info(Path))
        after
            rm_rf(TmpDir)
        end
    end).

%% =============================================================================
%% lookup_exact_or_wait
%% =============================================================================

wait_returns_immediately_on_available_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H">>),
        {ok, _Row} = erllama_cache_meta_srv:lookup_exact_or_wait(key(1), 100)
    end).

wait_returns_miss_immediately_on_no_row_test() ->
    with_srv(fun() ->
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact_or_wait(key(1), 100))
    end).

wait_returns_miss_after_timeout_when_writing_test() ->
    with_srv(fun() ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                {ok, _T} = erllama_cache_meta_srv:reserve_save(key(1), ram, self()),
                Parent ! reserved,
                receive
                    stop -> ok
                end
            end),
        receive
            reserved -> ok
        end,
        Result = erllama_cache_meta_srv:lookup_exact_or_wait(key(1), 50),
        Pid ! stop,
        ?assertEqual(miss, Result)
    end).

wait_replies_when_save_publishes_test() ->
    with_srv(fun() ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                {ok, T} = erllama_cache_meta_srv:reserve_save(key(1), ram, self()),
                Parent ! reserved,
                receive
                    publish -> ok
                end,
                ok = erllama_cache_meta_srv:announce_saved(key(1), T, 100, <<"H">>)
            end),
        receive
            reserved -> ok
        end,
        Caller =
            spawn(fun() ->
                R = erllama_cache_meta_srv:lookup_exact_or_wait(key(1), 5000),
                Parent ! {result, R}
            end),
        timer:sleep(50),
        Pid ! publish,
        receive
            {result, Result} ->
                ?assertMatch({ok, _Row}, Result)
        after 5000 ->
            erlang:error(timeout)
        end,
        _ = Caller
    end).

%% =============================================================================
%% Eviction
%% =============================================================================

gc_evicts_unreferenced_rows_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H1">>),
        ok = erllama_cache_meta_srv:insert_available(key(2), ram, 200, <<"H2">>),
        {evicted, N} = erllama_cache_meta_srv:gc(),
        ?assertEqual(2, N),
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(key(1))),
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(key(2)))
    end).

gc_skips_referenced_rows_test() ->
    with_srv(fun() ->
        ok = erllama_cache_meta_srv:insert_available(key(1), ram, 100, <<"H1">>),
        ok = erllama_cache_meta_srv:insert_available(key(2), ram, 200, <<"H2">>),
        {ok, _Ref, _, _, _, _} = erllama_cache_meta_srv:checkout(key(1), self()),
        {evicted, N} = erllama_cache_meta_srv:gc(),
        ?assertEqual(1, N),
        ?assertMatch({ok, _Row}, erllama_cache_meta_srv:lookup_exact(key(1))),
        ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(key(2)))
    end).

gc_skips_writing_rows_test() ->
    with_srv(fun() ->
        Parent = self(),
        Pid =
            spawn(fun() ->
                {ok, _T} = erllama_cache_meta_srv:reserve_save(key(1), disk, self()),
                Parent ! reserved,
                receive
                    stop -> ok
                end
            end),
        receive
            reserved -> ok
        end,
        {evicted, N} = erllama_cache_meta_srv:gc(),
        ?assertEqual(0, N),
        Pid ! stop
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

wait_for_refcount(Key, Expected, TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_for_refcount_loop(Key, Expected, Deadline).

wait_for_refcount_loop(Key, Expected, Deadline) ->
    case erllama_cache_meta_srv:dump(Key) of
        {ok, Row} ->
            case element(?POS_REFCOUNT, Row) of
                Expected -> ok;
                _ -> retry_or_fail(Key, Expected, Deadline)
            end;
        miss when Expected =:= 0 ->
            ok;
        _ ->
            retry_or_fail(Key, Expected, Deadline)
    end.

retry_or_fail(Key, Expected, Deadline) ->
    case erlang:monotonic_time(millisecond) > Deadline of
        true ->
            erlang:error({refcount_not_reached, Key, Expected});
        false ->
            timer:sleep(10),
            wait_for_refcount_loop(Key, Expected, Deadline)
    end.

wait_for_pid_dead(Pid, TimeoutMs) ->
    Mon = erlang:monitor(process, Pid),
    receive
        {'DOWN', Mon, process, Pid, _} -> ok
    after TimeoutMs ->
        erlang:demonitor(Mon, [flush]),
        erlang:error(timeout)
    end.

wait_for_row_gone(Key, TimeoutMs) ->
    wait_until(fun() -> erllama_cache_meta_srv:dump(Key) =:= miss end, TimeoutMs).

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
        "erllama_cache_meta_srv_tests_" ++
            integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = file:make_dir(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} ->
            [file:delete(filename:join(Dir, E)) || E <- Entries];
        _ ->
            ok
    end,
    file:del_dir(Dir).
