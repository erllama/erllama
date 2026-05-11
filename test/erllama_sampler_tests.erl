%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Stub-backend coverage for the sampler-params plumbing added in
%% Phase 2: every sampler key supplied via complete/3's Opts (or
%% infer/4's Params) must land on the backend's configure_sampler/2
%% callback verbatim. Non-sampler keys must be stripped.
-module(erllama_sampler_tests).
-include_lib("eunit/include/eunit.hrl").

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(sampler_disk, Dir),
    try
        Body()
    after
        catch gen_server:stop(sampler_disk),
        rm_rf(Dir),
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

make_tmp_dir() ->
    Base = filename:basedir(user_cache, "erllama-sampler-tests"),
    Dir = filename:join(Base, integer_to_list(erlang:unique_integer([positive]))),
    ok = filelib:ensure_path(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Files} ->
            [
                begin
                    Full = filename:join(Dir, F),
                    case filelib:is_dir(Full) of
                        true -> rm_rf(Full);
                        false -> file:delete(Full)
                    end
                end
             || F <- Files
            ],
            file:del_dir(Dir);
        _ ->
            ok
    end.

minimal_config() ->
    #{
        backend => erllama_model_stub,
        tier_srv => sampler_disk,
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

%% =============================================================================
%% Tests
%% =============================================================================

empty_opts_no_sampler_cfg_test() ->
    %% No sampler keys in Opts -> stub receives configure_sampler(#{}).
    with_app(fun() ->
        Id = <<"sampler_a">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, _, _} = erllama:complete(Id, <<"hello world">>),
            ?assertEqual(#{}, get_stub_cfg(Id))
        after
            erllama:unload(Id)
        end
    end).

full_sampler_set_lands_on_backend_test() ->
    with_app(fun() ->
        Id = <<"sampler_b">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        Opts = #{
            response_tokens => 4,
            parent_key => <<"ignored">>,
            temperature => 0.7,
            top_k => 40,
            top_p => 0.9,
            min_p => 0.05,
            repetition_penalty => 1.1,
            seed => 42
        },
        try
            {ok, _, _} = erllama:complete(Id, <<"hello">>, Opts),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(0.7, maps:get(temperature, Cfg)),
            ?assertEqual(40, maps:get(top_k, Cfg)),
            ?assertEqual(0.9, maps:get(top_p, Cfg)),
            ?assertEqual(0.05, maps:get(min_p, Cfg)),
            ?assertEqual(1.1, maps:get(repetition_penalty, Cfg)),
            ?assertEqual(42, maps:get(seed, Cfg)),
            %% Non-sampler keys must not leak in.
            ?assertNot(maps:is_key(response_tokens, Cfg)),
            ?assertNot(maps:is_key(parent_key, Cfg))
        after
            erllama:unload(Id)
        end
    end).

grammar_opt_lands_on_backend_test() ->
    with_app(fun() ->
        Id = <<"sampler_c">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, _, _} = erllama:complete(
                Id, <<"hi">>, #{grammar => <<"root ::= [01]+">>}
            ),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(<<"root ::= [01]+">>, maps:get(grammar, Cfg))
        after
            erllama:unload(Id)
        end
    end).

infer_path_also_configures_sampler_test() ->
    %% Same plumbing must work via the streaming infer/4 entry.
    with_app(fun() ->
        Id = <<"sampler_d">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            {ok, Tokens} = erllama:tokenize(Id, <<"a b c">>),
            {ok, Ref} = erllama:infer(
                Id,
                Tokens,
                #{response_tokens => 2, temperature => 0.5, seed => 7},
                self()
            ),
            drain(Ref),
            Cfg = get_stub_cfg(Id),
            ?assertEqual(0.5, maps:get(temperature, Cfg)),
            ?assertEqual(7, maps:get(seed, Cfg))
        after
            erllama:unload(Id)
        end
    end).

drain(Ref) ->
    receive
        {erllama_token, Ref, _} -> drain(Ref);
        {erllama_done, Ref, _} -> ok;
        {erllama_error, Ref, R} -> ?assert({unexpected_error, R} =:= ok)
    after 5000 ->
        ?assert({timeout, drain} =:= ok)
    end.

%% =============================================================================
%% Helpers — read the stub's recorded config via the model's test
%% accessor.
%% =============================================================================

get_stub_cfg(ModelId) ->
    erllama_model_stub:last_sampler_cfg(
        erllama_model:get_backend_state(ModelId)
    ).
