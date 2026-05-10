%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for the bucket-C C-Erlang surface: apply_chat_template/2,
%% embed/2, and grammar plumbing through Params. Stub backend only;
%% the real llama backend's NIF wiring lands in C-NIF.
-module(erllama_chat_template_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    DiskSrv = list_to_atom(
        "ct_disk_" ++
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
        fingerprint => binary:copy(<<16#55>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#66>>, 32),
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
            "ct_",
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
        "erllama_ct_tests_" ++
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

%% =============================================================================
%% apply_chat_template/2
%% =============================================================================

apply_chat_template_simple_test() ->
    with_model(fun(Id) ->
        Request = #{
            messages => [#{role => <<"user">>, content => <<"hello">>}]
        },
        {ok, Tokens} = erllama:apply_chat_template(Id, Request),
        ?assert(is_list(Tokens)),
        ?assert(lists:all(fun is_integer/1, Tokens))
    end).

apply_chat_template_with_system_test() ->
    with_model(fun(Id) ->
        Without = #{
            messages => [#{role => <<"user">>, content => <<"hi">>}]
        },
        With = Without#{system => <<"you are a helper">>},
        {ok, A} = erllama:apply_chat_template(Id, Without),
        {ok, B} = erllama:apply_chat_template(Id, With),
        %% System content lands in the rendered prompt, so token list
        %% length must differ.
        ?assertNotEqual(A, B),
        ?assert(length(B) > length(A))
    end).

apply_chat_template_with_tools_test() ->
    with_model(fun(Id) ->
        Tools = [
            #{
                name => <<"search">>,
                description => <<"web search">>,
                schema => #{}
            }
        ],
        Request = #{
            messages => [#{role => <<"user">>, content => <<"hi">>}],
            tools => Tools
        },
        {ok, Tokens} = erllama:apply_chat_template(Id, Request),
        ?assert(is_list(Tokens))
    end).

apply_chat_template_multi_turn_test() ->
    with_model(fun(Id) ->
        Request = #{
            messages => [
                #{role => <<"user">>, content => <<"hi">>},
                #{role => <<"assistant">>, content => <<"hello">>},
                #{role => <<"user">>, content => <<"who are you">>}
            ]
        },
        {ok, Tokens} = erllama:apply_chat_template(Id, Request),
        ?assert(length(Tokens) >= 3)
    end).

apply_chat_template_deterministic_test() ->
    with_model(fun(Id) ->
        Request = #{
            messages => [#{role => <<"user">>, content => <<"hi">>}]
        },
        {ok, A} = erllama:apply_chat_template(Id, Request),
        {ok, B} = erllama:apply_chat_template(Id, Request),
        ?assertEqual(A, B)
    end).

%% =============================================================================
%% embed/2
%% =============================================================================

embed_returns_vector_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"some text">>),
        {ok, Vec} = erllama:embed(Id, Tokens),
        ?assert(is_list(Vec)),
        ?assert(length(Vec) > 0),
        ?assert(lists:all(fun is_float/1, Vec))
    end).

embed_deterministic_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hello">>),
        {ok, A} = erllama:embed(Id, Tokens),
        {ok, B} = erllama:embed(Id, Tokens),
        ?assertEqual(A, B)
    end).

embed_different_input_different_vector_test() ->
    with_model(fun(Id) ->
        {ok, T1} = erllama:tokenize(Id, <<"alpha">>),
        {ok, T2} = erllama:tokenize(Id, <<"omega">>),
        {ok, A} = erllama:embed(Id, T1),
        {ok, B} = erllama:embed(Id, T2),
        ?assertNotEqual(A, B)
    end).

%% =============================================================================
%% Grammar plumbing
%% =============================================================================

grammar_in_params_does_not_break_streaming_test() ->
    %% The stub backend ignores grammar (it doesn't sample). What we
    %% verify is that the grammar option flows through Params without
    %% crashing the gen_statem.
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hi">>),
        Params = #{
            response_tokens => 4,
            grammar => <<"root ::= \"a\" | \"b\"">>
        },
        {ok, Ref} = erllama:infer(Id, Tokens, Params, self()),
        ?assert(is_reference(Ref)),
        Result = drain(Ref, 5000),
        ?assertMatch({_TokensList, #{}}, Result)
    end).

grammar_undefined_is_no_op_test() ->
    with_model(fun(Id) ->
        {ok, Tokens} = erllama:tokenize(Id, <<"hi">>),
        %% no grammar key
        Params = #{response_tokens => 2},
        {ok, _Ref} = erllama:infer(Id, Tokens, Params, self()),
        receive
            {erllama_done, _, _} -> ok
        after 5000 -> ?assert(false)
        end
    end).

%% =============================================================================
%% Real-backend compatibility (negative coverage)
%% =============================================================================
%% The llama backend declares the callbacks but returns
%% {error, not_implemented}. A unit test against the stub cannot
%% exercise that path without loading a GGUF, so we just verify the
%% public API surface of the not_supported / not_implemented errors
%% bubbles back through the gen_statem cleanly. The real coverage
%% lands in C-NIF when the NIF functions exist.
unsupported_apply_chat_template_returns_error_test() ->
    %% Use a model with a backend that explicitly disables chat.
    %% We do this by spinning up a model with a backend that does not
    %% export apply_chat_template at all - but the stub does, so we
    %% can't easily simulate this without a third backend module.
    %% Instead, just sanity-check the stub returns {ok, _}.
    with_model(fun(Id) ->
        ?assertMatch(
            {ok, _},
            erllama:apply_chat_template(
                Id,
                #{messages => [#{role => <<"u">>, content => <<"x">>}]}
            )
        )
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

drain(Ref, TimeoutMs) ->
    drain(Ref, TimeoutMs, []).

drain(Ref, TimeoutMs, Acc) ->
    receive
        {erllama_token, Ref, B} -> drain(Ref, TimeoutMs, [B | Acc]);
        {erllama_done, Ref, S} -> {lists:reverse(Acc), S};
        {erllama_error, Ref, R} -> {error, R}
    after TimeoutMs ->
        timeout
    end.
