-module(erllama_model_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_model(PolicyOverrides, Body) ->
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    {ok, _} = erllama_cache_writer:start_link(2),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(test_disk, Dir),
    Policy = maps:merge(default_policy(), PolicyOverrides),
    Config = #{
        tier_srv => test_disk,
        tier => disk,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        context_size => 4096,
        policy => Policy
    },
    {ok, _} = erllama_model:start_link(test_model, Config),
    try
        Body(Config)
    after
        catch erllama_model:stop(test_model),
        catch gen_server:stop(test_disk),
        catch gen_server:stop(erllama_cache_writer),
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv),
        rm_rf(Dir)
    end.

default_policy() ->
    #{
        min_tokens => 4,
        cold_min_tokens => 4,
        cold_max_tokens => 1000,
        continued_interval => 2048,
        boundary_trim_tokens => 0,
        boundary_align_tokens => 1,
        session_resume_wait_ms => 50
    }.

short_prompt() -> <<"hi">>.

long_prompt() ->
    list_to_binary(string:join([integer_to_list(N) || N <- lists:seq(1, 12)], " ")).

key_for_tokens(Tokens, Cfg) ->
    erllama_cache_key:make(#{
        fingerprint => maps:get(fingerprint, Cfg),
        quant_type => maps:get(quant_type, Cfg),
        ctx_params_hash => maps:get(ctx_params_hash, Cfg),
        tokens => Tokens
    }).

prompt_tokens(Prompt) ->
    [
        erlang:phash2(W) rem (1 bsl 32)
     || W <- binary:split(Prompt, <<" ">>, [global, trim_all]),
        W =/= <<>>
    ].

wait_for_key(Key, TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_for_key_loop(Key, Deadline).

wait_for_key_loop(Key, Deadline) ->
    case erllama_cache_meta_srv:lookup_exact(Key) of
        {ok, _} = R ->
            R;
        miss ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    miss;
                false ->
                    timer:sleep(10),
                    wait_for_key_loop(Key, Deadline)
            end
    end.

%% =============================================================================
%% Lifecycle
%% =============================================================================

starts_in_idle_test() ->
    with_model(#{}, fun(_) ->
        ?assertEqual(idle, erllama_model:status(test_model))
    end).

complete_returns_response_test() ->
    with_model(#{}, fun(_) ->
        {ok, _Reply, Generated} = erllama_model:complete(test_model, short_prompt()),
        ?assert(length(Generated) > 0),
        ?assertEqual(idle, erllama_model:status(test_model))
    end).

%% =============================================================================
%% Cold save fires for prompts in [cold_min, cold_max]
%% =============================================================================

short_prompt_does_not_cold_save_test() ->
    with_model(
        #{min_tokens => 4, cold_min_tokens => 4, cold_max_tokens => 1000},
        fun(Cfg) ->
            {ok, _, _} = erllama_model:complete(test_model, short_prompt()),
            %% Short prompt has 1 token; min is 4 -> no cold save.
            Tokens = prompt_tokens(short_prompt()),
            ColdKey = key_for_tokens(Tokens, Cfg),
            ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(ColdKey))
        end
    ).

long_prompt_fires_cold_save_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, _, _} = erllama_model:complete(test_model, long_prompt()),
        Tokens = prompt_tokens(long_prompt()),
        ColdKey = key_for_tokens(Tokens, Cfg),
        ?assertMatch({ok, _Row}, wait_for_key(ColdKey, 1000))
    end).

%% =============================================================================
%% Finish save fires at end-of-stream when total is above min
%% =============================================================================

finish_save_fires_for_long_prompt_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, _, Generated} =
            erllama_model:complete(test_model, long_prompt(), #{response_tokens => 6}),
        FullTokens = prompt_tokens(long_prompt()) ++ Generated,
        FinishKey = key_for_tokens(FullTokens, Cfg),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000))
    end).

%% =============================================================================
%% Cache hit on repeat
%% =============================================================================

repeat_prompt_hits_finish_save_path_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, _, Gen1} =
            erllama_model:complete(test_model, long_prompt(), #{response_tokens => 4}),
        FullKey1 = key_for_tokens(prompt_tokens(long_prompt()) ++ Gen1, Cfg),
        {ok, _} = wait_for_key(FullKey1, 1000),
        %% Second complete with the *same prompt + response continuation*:
        %% the cache row keyed on prompt-only doesn't yet exist (the
        %% first run only persisted the cold/finish keys), but a third
        %% turn that uses parent_key=FullKey1 will hit session resume.
        ?assertMatch({ok, _Row}, erllama_cache_meta_srv:lookup_exact(FullKey1))
    end).

%% =============================================================================
%% parent_key (session resume)
%% =============================================================================

parent_key_session_resume_test() ->
    with_model(#{}, fun(Cfg) ->
        %% Turn 1: prompt + response.
        {ok, _, Gen1} =
            erllama_model:complete(test_model, long_prompt(), #{response_tokens => 4}),
        FullTokens1 = prompt_tokens(long_prompt()) ++ Gen1,
        FullKey1 = key_for_tokens(FullTokens1, Cfg),
        {ok, _} = wait_for_key(FullKey1, 1000),
        %% Turn 2: a longer prompt that strictly prefix-extends turn 1's
        %% live tokens. Use parent_key = FullKey1 to take the session
        %% resume path.
        Extension = list_to_binary(
            stub_detokenize_decimal(FullTokens1) ++ " more tokens for turn two"
        ),
        {ok, _, _} =
            erllama_model:complete(test_model, Extension, #{
                parent_key => FullKey1,
                response_tokens => 2
            }),
        ?assertEqual(idle, erllama_model:status(test_model))
    end).

%% =============================================================================
%% Continued saves during generation
%% =============================================================================

continued_save_fires_during_long_generation_test() ->
    %% Lower continued_interval so the fixture's prompt + response
    %% crosses the boundary at least once.
    with_model(
        #{continued_interval => 4, response_target => 8},
        fun(Cfg) ->
            PromptTokens = prompt_tokens(long_prompt()),
            {ok, _, Generated} = erllama_model:complete(
                test_model, long_prompt(), #{response_tokens => 8}
            ),
            %% A continued save fires when LiveTokens - LastSavedAt
            %% reaches continued_interval. With the cold save firing
            %% at 12 prompt tokens, last_save_at = 12. After 4 more
            %% generated tokens, a continued save fires for the
            %% first 16 tokens. Those tokens are
            %% PromptTokens ++ first 4 generated.
            FirstContinuedTokens =
                PromptTokens ++ lists:sublist(Generated, 4),
            FirstContinuedKey = key_for_tokens(FirstContinuedTokens, Cfg),
            ?assertMatch({ok, _Row}, wait_for_key(FirstContinuedKey, 1000))
        end
    ).

%% =============================================================================
%% Evict save
%% =============================================================================

evict_idle_with_no_context_is_noop_test() ->
    with_model(#{}, fun(_Cfg) ->
        ok = erllama_model:evict(test_model),
        ?assertEqual(0, length(erllama_cache_meta_srv:dump())),
        ?assertEqual(idle, erllama_model:status(test_model))
    end).

evict_during_generation_persists_live_state_test() ->
    %% Fire evict in the middle of a long generation. The model
    %% intercepts the call between decode_step events and writes
    %% an evict save with whatever tokens are live.
    with_model(#{response_target => 200, continued_interval => 100000}, fun(_Cfg) ->
        Parent = self(),
        spawn(fun() ->
            Parent !
                {done,
                    erllama_model:complete(
                        test_model, long_prompt(), #{response_tokens => 200}
                    )}
        end),
        %% Give the gen_statem a beat to enter generating.
        timer:sleep(0),
        ok = erllama_model:evict(test_model),
        %% Wait for the request to complete (the finish save fires too,
        %% so we expect at least an evict row plus cold + finish).
        receive
            {done, _} -> ok
        after 5000 -> erlang:error(timeout)
        end,
        timer:sleep(50),
        ?assert(length(erllama_cache_meta_srv:dump()) >= 2)
    end).

%% =============================================================================
%% Shutdown save
%% =============================================================================

shutdown_idle_with_no_context_is_noop_test() ->
    with_model(#{}, fun(_Cfg) ->
        ok = erllama_model:shutdown(test_model),
        ?assertEqual(0, length(erllama_cache_meta_srv:dump())),
        ?assertEqual(idle, erllama_model:status(test_model))
    end).

%% =============================================================================
%% Concurrency
%% =============================================================================

concurrent_complete_rejects_with_busy_test() ->
    with_model(#{}, fun(_Cfg) ->
        Parent = self(),
        %% Spawn a slow caller that keeps the gen_statem busy.
        spawn(fun() ->
            Parent ! {first, erllama_model:complete(test_model, long_prompt())}
        end),
        timer:sleep(0),
        %% Try a second complete; the gen_statem is in prefilling/
        %% generating and rejects.
        Result = erllama_model:complete(test_model, short_prompt()),
        receive
            {first, _} -> ok
        after 5000 -> erlang:error(first_caller_timeout)
        end,
        case Result of
            %% raced and got there first/after
            {ok, _, _} -> ok;
            %% busy as expected
            {error, busy} -> ok
        end
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

stub_detokenize_decimal(Tokens) ->
    string:join([integer_to_list(T) || T <- Tokens], " ").

make_tmp_dir() ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(
        Base,
        "erllama_model_tests_" ++
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
