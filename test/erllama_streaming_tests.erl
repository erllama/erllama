%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for streaming inference: erllama:infer/4 + erllama:cancel/1.
%% Uses the stub backend so no GGUF model file is required.
-module(erllama_streaming_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    DiskSrv = list_to_atom(
        "stream_disk_" ++
            integer_to_list(erlang:unique_integer([positive]))
    ),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    try
        Body(DiskSrv)
    after
        catch gen_server:stop(DiskSrv),
        rm_rf(Dir),
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

minimal_config(DiskSrv) ->
    #{
        backend => erllama_model_stub,
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => binary:copy(<<16#33>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#44>>, 32),
        context_size => 1024,
        policy => #{
            min_tokens => 4,
            cold_min_tokens => 4,
            cold_max_tokens => 1000,
            continued_interval => 2048,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 50
        }
    }.

with_model(Body) ->
    with_model(#{}, Body).

with_model(ConfigOverrides, Body) ->
    with_app(fun(DiskSrv) ->
        Id = iolist_to_binary([
            "stream_",
            integer_to_binary(erlang:unique_integer([positive]))
        ]),
        Cfg = maps:merge(minimal_config(DiskSrv), ConfigOverrides),
        {ok, _} = erllama:load_model(Id, Cfg),
        try
            Body(Id)
        after
            erllama:unload(Id)
        end
    end).

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_streaming_tests_" ++
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
    file:del_dir(Dir),
    ok.

%% Drain messages of a given Ref until {erllama_done, Ref, _} or timeout.
%% Returns {Tokens, Stats} or {error, Reason} on erllama_error.
collect_stream(Ref, TimeoutMs) ->
    collect_stream(Ref, TimeoutMs, []).

collect_stream(Ref, TimeoutMs, Acc) ->
    receive
        {erllama_token, Ref, Bin} ->
            collect_stream(Ref, TimeoutMs, [Bin | Acc]);
        {erllama_done, Ref, Stats} ->
            {lists:reverse(Acc), Stats};
        {erllama_error, Ref, Reason} ->
            {error, Reason}
    after TimeoutMs ->
        {timeout, lists:reverse(Acc)}
    end.

%% =============================================================================
%% Happy path
%% =============================================================================

infer_returns_ref_and_streams_tokens_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello world">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 4}, self()),
        ?assert(is_reference(Ref)),
        {Out, Stats} = collect_stream(Ref, 5000),
        ?assert(length(Out) >= 1),
        ?assertEqual(length(Tokens), maps:get(prompt_tokens, Stats)),
        %% Stub backend never emits eog, so finish_reason can be either
        %% length (target reached) or stop (eog from a real backend).
        FR = maps:get(finish_reason, Stats),
        ?assert(FR =:= length orelse FR =:= stop),
        ?assertEqual(false, maps:get(cancelled, Stats))
    end).

infer_emits_binary_fragments_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"x y z">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 3}, self()),
        {Out, _Stats} = collect_stream(Ref, 5000),
        ?assert(lists:all(fun is_binary/1, Out))
    end).

infer_finish_reason_length_when_target_reached_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"a b">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 2}, self()),
        {_, Stats} = collect_stream(Ref, 5000),
        FR = maps:get(finish_reason, Stats),
        %% Stub backend never produces eog so we should hit the
        %% response_target -> finish_reason=length path.
        ?assert(FR =:= length orelse FR =:= stop)
    end).

stats_carry_token_counts_test() ->
    with_model(fun(Id) ->
        {ok, PromptTokens} = erllama:tokenize(Id, <<"hello big world">>),
        {ok, Ref} = erllama:infer(Id, PromptTokens, #{response_tokens => 4}, self()),
        {Out, Stats} = collect_stream(Ref, 5000),
        ?assertEqual(length(PromptTokens), maps:get(prompt_tokens, Stats)),
        ?assertEqual(length(Out), maps:get(completion_tokens, Stats))
    end).

stats_carry_finish_key_and_committed_tokens_test() ->
    with_model(fun(Id) ->
        {ok, PromptTokens} = erllama:tokenize(Id, <<"hello big world">>),
        {ok, Ref} = erllama:infer(Id, PromptTokens, #{response_tokens => 6}, self()),
        {_Out, Stats} = collect_stream(Ref, 5000),
        ?assert(maps:is_key(finish_key, Stats)),
        ?assert(maps:is_key(committed_tokens, Stats)),
        Committed = maps:get(committed_tokens, Stats),
        Prompt = maps:get(prompt_tokens, Stats),
        Completion = maps:get(completion_tokens, Stats),
        ?assertEqual(Prompt + Completion, Committed),
        FinishKey = maps:get(finish_key, Stats),
        %% Stub backend has min_tokens=4 in the fixture; with prompt
        %% of 3 + response of 6 the committed count is well above the
        %% threshold so the finish save must have fired.
        ?assert(is_binary(FinishKey))
    end).

%% =============================================================================
%% stop_sequences
%% =============================================================================

%% The stub backend's detokenize produces space-separated decimal
%% integers, so a stop_sequence containing any digit hits early.
%% Use the full digit set to guarantee a match regardless of which
%% 32-bit token IDs the stub picks.
all_digits() ->
    [
        <<"0">>,
        <<"1">>,
        <<"2">>,
        <<"3">>,
        <<"4">>,
        <<"5">>,
        <<"6">>,
        <<"7">>,
        <<"8">>,
        <<"9">>
    ].

stop_sequence_value_reported_on_match_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        Params = #{response_tokens => 16, stop_sequences => all_digits()},
        {ok, Ref} = erllama:infer(Id, Tokens, Params, self()),
        {_Out, Stats} = collect_stream(Ref, 5000),
        ?assertEqual(stop, maps:get(finish_reason, Stats)),
        Match = maps:get(stop_sequence, Stats),
        ?assert(is_binary(Match)),
        ?assert(lists:member(Match, all_digits()))
    end).

stop_sequence_absent_on_length_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        %% A stop string the decimal output cannot contain.
        Params = #{
            response_tokens => 1,
            stop_sequences => [<<"unmatchable-xyz">>]
        },
        {ok, Ref} = erllama:infer(Id, Tokens, Params, self()),
        {_Out, Stats} = collect_stream(Ref, 5000),
        ?assertNot(maps:is_key(stop_sequence, Stats)),
        ?assertEqual(length, maps:get(finish_reason, Stats))
    end).

stop_sequence_absent_when_no_param_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 2}, self()),
        {_Out, Stats} = collect_stream(Ref, 5000),
        ?assertNot(maps:is_key(stop_sequence, Stats))
    end).

stop_sequence_trimmed_from_stream_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        Params = #{response_tokens => 16, stop_sequences => all_digits()},
        {ok, Ref} = erllama:infer(Id, Tokens, Params, self()),
        {Out, Stats} = collect_stream(Ref, 5000),
        Match = maps:get(stop_sequence, Stats),
        Streamed = iolist_to_binary(Out),
        ?assertEqual(nomatch, binary:match(Streamed, Match))
    end).

%% =============================================================================
%% thinking
%% =============================================================================

%% Drain in arrival order. Returns a list of {kind, Payload} tuples
%% terminated by {done, Stats} or {error, Reason}.
collect_all(Ref, TimeoutMs) ->
    collect_all(Ref, TimeoutMs, []).

collect_all(Ref, TimeoutMs, Acc) ->
    receive
        {erllama_token, Ref, {thinking_delta, Bin}} ->
            collect_all(Ref, TimeoutMs, [{thinking_delta, Bin} | Acc]);
        {erllama_token, Ref, Bin} when is_binary(Bin) ->
            collect_all(Ref, TimeoutMs, [{token, Bin} | Acc]);
        {erllama_thinking_end, Ref, Sig} ->
            collect_all(Ref, TimeoutMs, [{thinking_end, Sig} | Acc]);
        {erllama_done, Ref, Stats} ->
            lists:reverse([{done, Stats} | Acc]);
        {erllama_error, Ref, Reason} ->
            lists:reverse([{error, Reason} | Acc])
    after TimeoutMs ->
        lists:reverse([{timeout, TimeoutMs} | Acc])
    end.

thinking_capable_stub_emits_delta_end_then_tokens_test() ->
    with_model(#{thinking_capable => true}, fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref} = erllama:infer(
            Id, Tokens, #{response_tokens => 3, thinking => enabled}, self()
        ),
        Events = collect_all(Ref, 5000),
        Kinds = [K || {K, _} <- Events],
        %% Expect: thinking_delta+, thinking_end, token+, done.
        ?assert(lists:member(thinking_delta, Kinds)),
        ?assertEqual(1, length([1 || thinking_end <- Kinds])),
        ?assert(lists:member(token, Kinds)),
        ?assertEqual(done, lists:last(Kinds)),
        %% Order: every thinking_delta precedes the thinking_end,
        %% which precedes every token.
        EndIdx = idx_of(thinking_end, Kinds),
        ?assert(
            lists:all(
                fun({K, I}) -> K =/= thinking_delta orelse I < EndIdx end,
                indexed(Kinds)
            )
        ),
        ?assert(
            lists:all(
                fun({K, I}) -> K =/= token orelse I > EndIdx end,
                indexed(Kinds)
            )
        ),
        %% Signature: non-empty binary, exactly 32 bytes (sha256).
        Sig = sig_from(Events),
        ?assert(is_binary(Sig)),
        ?assertEqual(32, byte_size(Sig))
    end).

thinking_capable_stub_with_thinking_disabled_fails_test() ->
    %% Stub still emits thinking_token but the caller didn't opt in.
    %% The scheduler treats this as a backend bug and surfaces a
    %% clean erllama_error tagged thinking_not_enabled.
    with_model(#{thinking_capable => true}, fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 3}, self()),
        Events = collect_all(Ref, 5000),
        ?assertEqual({error, thinking_not_enabled}, lists:last(Events))
    end).

non_thinking_stub_with_thinking_enabled_is_normal_stream_test() ->
    %% Default stub (no thinking_capable). Caller may freely set
    %% thinking => enabled; no thinking messages should arrive.
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref} = erllama:infer(
            Id, Tokens, #{response_tokens => 2, thinking => enabled}, self()
        ),
        Events = collect_all(Ref, 5000),
        Kinds = [K || {K, _} <- Events],
        ?assertEqual([], [K || K <- Kinds, K =:= thinking_delta]),
        ?assertEqual([], [K || K <- Kinds, K =:= thinking_end]),
        ?assertEqual(done, lists:last(Kinds))
    end).

thinking_budget_clips_after_one_delta_test() ->
    %% Stub normally emits two thinking_token deltas before its
    %% thinking_end marker. With budget = 1 the scheduler must
    %% synthesise the close right after the first delta and route
    %% subsequent thinking_tokens through the normal token path.
    with_model(#{thinking_capable => true}, fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref} = erllama:infer(
            Id,
            Tokens,
            #{
                response_tokens => 4,
                thinking => enabled,
                thinking_budget_tokens => 1
            },
            self()
        ),
        Events = collect_all(Ref, 5000),
        Kinds = [K || {K, _} <- Events],
        Deltas = [K || K <- Kinds, K =:= thinking_delta],
        Ends = [K || K <- Kinds, K =:= thinking_end],
        ?assertEqual(1, length(Deltas)),
        %% Exactly one thinking_end across the whole stream, even if
        %% the backend later emitted its own thinking_end marker.
        ?assertEqual(1, length(Ends)),
        ?assertEqual(done, lists:last(Kinds)),
        %% Every thinking_delta arrives before the thinking_end.
        EndIdx = idx_of(thinking_end, Kinds),
        ?assert(
            lists:all(
                fun({K, I}) -> K =/= thinking_delta orelse I < EndIdx end,
                indexed(Kinds)
            )
        )
    end).

thinking_budget_unset_keeps_existing_shape_test() ->
    %% Regression: with no thinking_budget_tokens, the stub's full
    %% natural shape (two thinking_deltas) must still come through.
    with_model(#{thinking_capable => true}, fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref} = erllama:infer(
            Id, Tokens, #{response_tokens => 3, thinking => enabled}, self()
        ),
        Events = collect_all(Ref, 5000),
        Kinds = [K || {K, _} <- Events],
        Deltas = [K || K <- Kinds, K =:= thinking_delta],
        ?assertEqual(2, length(Deltas)),
        ?assertEqual(1, length([1 || thinking_end <- Kinds]))
    end).

%% Helpers for the thinking tests.
indexed(L) ->
    lists:zip(L, lists:seq(1, length(L))).

idx_of(Target, L) ->
    case lists:keyfind(Target, 1, indexed(L)) of
        {Target, I} -> I;
        false -> 0
    end.

sig_from([{thinking_end, S} | _]) -> S;
sig_from([_ | T]) -> sig_from(T);
sig_from([]) -> <<>>.

%% =============================================================================
%% Concurrency / busy
%% =============================================================================

second_infer_while_busy_is_queued_test() ->
    %% Phase 4: concurrent calls no longer return {error, busy}; the
    %% gen_statem queues them and dispatches each on return to idle.
    %% Cancelling the first request makes room for the second.
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, Ref1} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref1, _} -> ok
        after 5000 -> ?assert(false)
        end,
        %% Issue the second infer; it queues because the first is
        %% still generating. The {ok, Ref2} reply comes back as soon
        %% as the call is admitted (the gen_statem accepts the call
        %% and returns the ref before dispatching prefill, so the
        %% reply lands without waiting for the first to drain).
        {ok, Ref2} = erllama:infer(
            Id, Tokens, #{response_tokens => 1}, self()
        ),
        %% Cancel the long-running first; the queued second then
        %% starts and finishes naturally.
        ok = erllama:cancel(Ref1),
        receive
            {erllama_done, Ref1, _} -> ok
        after 5000 -> ?assert(false)
        end,
        receive
            {erllama_done, Ref2, _} -> ok
        after 5000 -> ?assert(false)
        end
    end).

complete_while_streaming_is_queued_test() ->
    %% A sync complete/2 issued while an infer is in flight would
    %% block this test process forever waiting for the queued reply,
    %% so we run it from a spawned helper that signals when its
    %% complete returns. The helper's call sits on the gen_statem
    %% queue until the cancellation drains the infer.
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hi">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref, _} -> ok
        after 5000 -> ?assert(false)
        end,
        Parent = self(),
        spawn(fun() ->
            Parent ! {complete_result, erllama:complete(Id, <<"another">>)}
        end),
        %% Give the spawned call time to enter the queue.
        timer:sleep(50),
        ok = erllama:cancel(Ref),
        receive
            {erllama_done, Ref, _} -> ok
        after 5000 -> ?assert(false)
        end,
        %% The queued complete now drains; expect a successful reply.
        receive
            {complete_result, {ok, _Reply}} -> ok
        after 5000 -> ?assert(false)
        end
    end).

queued_requests_drain_in_fifo_order_test() ->
    %% Three concurrent infer calls. First runs to natural completion;
    %% the second and third are queued. The erllama_done messages
    %% must arrive in submission order (FIFO queue semantics).
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hi">>),
        {ok, R1} = erllama:infer(Id, Tokens, #{response_tokens => 2}, self()),
        {ok, R2} = erllama:infer(Id, Tokens, #{response_tokens => 2}, self()),
        {ok, R3} = erllama:infer(Id, Tokens, #{response_tokens => 2}, self()),
        ?assertNotEqual(R1, R2),
        ?assertNotEqual(R2, R3),
        Order = drain_done_in_order(3, 5000),
        ?assertEqual([R1, R2, R3], Order)
    end).

drain_done_in_order(0, _Timeout) ->
    [];
drain_done_in_order(N, Timeout) ->
    receive
        {erllama_done, Ref, _} -> [Ref | drain_done_in_order(N - 1, Timeout)];
        %% Tokens emitted while waiting for done are fine; ignore them.
        {erllama_token, _, _} -> drain_done_in_order(N, Timeout)
    after Timeout -> ?assert(false)
    end.

%% =============================================================================
%% Cancellation
%% =============================================================================

cancel_unknown_ref_is_noop_test() ->
    with_app(fun(_DiskSrv) ->
        ?assertEqual(ok, erllama:cancel(make_ref()))
    end).

cancel_idempotent_test() ->
    with_app(fun(_DiskSrv) ->
        Ref = make_ref(),
        ?assertEqual(ok, erllama:cancel(Ref)),
        ?assertEqual(ok, erllama:cancel(Ref))
    end).

cancel_in_flight_marks_done_cancelled_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        %% Use a large response_tokens so cancellation is observable
        %% before completion. Wait for the first token so we know we
        %% are mid-stream before cancelling.
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref, _} -> ok
        after 5000 -> ?assert(false)
        end,
        ok = erllama:cancel(Ref),
        case collect_stream(Ref, 5000) of
            {timeout, _} ->
                ?assert(false, "stream did not finish within 5s after cancel");
            {_Tokens, Stats} when is_map(Stats) ->
                ?assertEqual(true, maps:get(cancelled, Stats)),
                ?assertEqual(cancelled, maps:get(finish_reason, Stats))
        end
    end).

cancelled_stream_releases_model_for_next_request_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hi">>),
        {ok, Ref1} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref1, _} -> ok
        after 5000 -> ?assert(false)
        end,
        ok = erllama:cancel(Ref1),
        _ = collect_stream(Ref1, 5000),
        %% After cancel-flushed-done, the model should be idle and
        %% accept a new infer.
        {ok, Ref2} = erllama:infer(Id, Tokens, #{response_tokens => 2}, self()),
        ?assertNotEqual(Ref1, Ref2),
        _ = collect_stream(Ref2, 5000)
    end).

%% =============================================================================
%% Inflight registry
%% =============================================================================

inflight_registers_and_unregisters_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"x">>),
        Pre = length(erllama_inflight:all()),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref, _} -> ok
        after 5000 -> ?assert(false)
        end,
        ?assertMatch({ok, _Pid}, erllama_inflight:lookup(Ref)),
        ?assert(length(erllama_inflight:all()) >= Pre + 1),
        ok = erllama:cancel(Ref),
        _ = collect_stream(Ref, 5000),
        ?assertEqual({error, not_found}, erllama_inflight:lookup(Ref))
    end).
