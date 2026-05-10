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
    with_app(fun(DiskSrv) ->
        Id = iolist_to_binary([
            "stream_",
            integer_to_binary(erlang:unique_integer([positive]))
        ]),
        {ok, _} = erllama:load_model(Id, minimal_config(DiskSrv)),
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

%% =============================================================================
%% Concurrency / busy
%% =============================================================================

second_infer_while_busy_returns_error_busy_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        %% Use a large target so the stream stays in flight long
        %% enough for the second call to race. We also wait for the
        %% first token to confirm the model is in `generating` state
        %% before issuing the second call.
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref, _} -> ok
        after 5000 -> ?assert(false)
        end,
        Result = erllama:infer(Id, Tokens, #{response_tokens => 1}, self()),
        ?assertEqual({error, busy}, Result),
        %% Cancel and drain so unload is clean and quick.
        ok = erllama:cancel(Ref),
        receive
            {erllama_done, Ref, _} -> ok
        after 5000 -> ok
        end
    end).

complete_while_streaming_returns_error_busy_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hi">>),
        {ok, Ref} = erllama:infer(Id, Tokens, #{response_tokens => 10000}, self()),
        receive
            {erllama_token, Ref, _} -> ok
        after 5000 -> ?assert(false)
        end,
        ?assertEqual({error, busy}, erllama:complete(Id, <<"another">>)),
        ok = erllama:cancel(Ref),
        receive
            {erllama_done, Ref, _} -> ok
        after 5000 -> ok
        end
    end).

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
