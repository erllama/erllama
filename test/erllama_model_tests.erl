%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model_tests).
-include_lib("eunit/include/eunit.hrl").
-include("erllama_cache.hrl").

%% =============================================================================
%% Fixtures
%% =============================================================================

with_model(PolicyOverrides, Body) ->
    with_model(PolicyOverrides, #{}, Body).

with_model(PolicyOverrides, ConfigOverrides, Body) ->
    ok = erllama_cache_counters:init(),
    erllama_cache_counters:reset(),
    {ok, _} = erllama_registry:start_link(),
    {ok, _} = erllama_inflight:start_link(),
    {ok, _} = erllama_cache_meta_srv:start_link(),
    {ok, _} = erllama_cache_ram:start_link(),
    {ok, _} = erllama_cache_writer:start_link(2),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(test_disk, Dir),
    Policy = maps:merge(default_policy(), PolicyOverrides),
    BaseConfig = #{
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
    Config = maps:merge(BaseConfig, ConfigOverrides),
    {ok, _} = erllama_model:start_link(<<"test_model">>, Config),
    try
        Body(Config)
    after
        catch erllama_model:stop(<<"test_model">>),
        catch gen_server:stop(test_disk),
        catch gen_server:stop(erllama_cache_writer),
        catch gen_server:stop(erllama_cache_ram),
        catch gen_server:stop(erllama_cache_meta_srv),
        catch gen_server:stop(erllama_inflight),
        catch gen_server:stop(erllama_registry),
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
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

model_info_returns_map_test() ->
    with_model(#{}, fun(_) ->
        Info = erllama_model:model_info(<<"test_model">>),
        ?assertEqual(<<"test_model">>, maps:get(id, Info)),
        ?assertEqual(idle, maps:get(status, Info)),
        ?assert(is_pid(maps:get(pid, Info))),
        ?assertEqual(erllama_model_stub, maps:get(backend, Info)),
        ?assertEqual(4096, maps:get(context_size, Info)),
        ?assertEqual(f16, maps:get(quant_type, Info)),
        ?assertEqual(16, maps:get(quant_bits, Info)),
        ?assertEqual(disk, maps:get(tier, Info)),
        ?assertEqual(32, byte_size(maps:get(fingerprint, Info)))
    end).

model_info_via_pid_test() ->
    with_model(#{}, fun(_) ->
        Pid = erllama_registry:whereis_name(<<"test_model">>),
        Info = erllama_model:model_info(Pid),
        ?assertEqual(<<"test_model">>, maps:get(id, Info))
    end).

tokenize_returns_list_test() ->
    with_model(#{}, fun(_) ->
        {ok, Tokens} = erllama_model:tokenize(<<"test_model">>, <<"hello world">>),
        ?assert(is_list(Tokens)),
        ?assert(lists:all(fun is_integer/1, Tokens))
    end).

tokenize_empty_string_test() ->
    with_model(#{}, fun(_) ->
        {ok, Tokens} = erllama_model:tokenize(<<"test_model">>, <<>>),
        ?assertEqual([], Tokens)
    end).

detokenize_roundtrip_test() ->
    %% The stub backend is not roundtrippable (phash2-based), but the
    %% types should line up: tokenize -> [int], detokenize -> binary.
    with_model(#{}, fun(_) ->
        {ok, Tokens} = erllama_model:tokenize(<<"test_model">>, <<"hi there">>),
        {ok, Bin} = erllama_model:detokenize(<<"test_model">>, Tokens),
        ?assert(is_binary(Bin))
    end).

tokenize_concurrent_with_idle_test() ->
    with_model(#{}, fun(_) ->
        ?assertEqual(idle, erllama_model:status(<<"test_model">>)),
        {ok, _} = erllama_model:tokenize(<<"test_model">>, <<"x">>),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

via_unknown_model_crashes_test() ->
    with_model(#{}, fun(_) ->
        ?assertExit(
            {noproc, {erllama_model, not_found, <<"unknown">>}},
            erllama_model:status(<<"unknown">>)
        )
    end).

complete_returns_response_test() ->
    with_model(#{}, fun(_) ->
        {ok, Result} = erllama_model:complete(<<"test_model">>, short_prompt()),
        ?assert(is_map(Result)),
        ?assert(is_binary(maps:get(reply, Result))),
        Generated = maps:get(generated, Result),
        ?assert(length(Generated) > 0),
        ?assertEqual(length(Generated), maps:get(completion_tokens, maps:get(stats, Result))),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

%% =============================================================================
%% Cold save fires for prompts in [cold_min, cold_max]
%% =============================================================================

short_prompt_does_not_cold_save_test() ->
    with_model(
        #{min_tokens => 4, cold_min_tokens => 4, cold_max_tokens => 1000},
        fun(Cfg) ->
            {ok, _} = erllama_model:complete(<<"test_model">>, short_prompt()),
            %% Short prompt has 1 token; min is 4 -> no cold save.
            Tokens = prompt_tokens(short_prompt()),
            ColdKey = key_for_tokens(Tokens, Cfg),
            ?assertEqual(miss, erllama_cache_meta_srv:lookup_exact(ColdKey))
        end
    ).

long_prompt_fires_cold_save_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, _} = erllama_model:complete(<<"test_model">>, long_prompt()),
        Tokens = prompt_tokens(long_prompt()),
        ColdKey = key_for_tokens(Tokens, Cfg),
        ?assertMatch({ok, _Row}, wait_for_key(ColdKey, 1000))
    end).

%% =============================================================================
%% Finish save fires at end-of-stream when total is above min
%% =============================================================================

finish_save_fires_for_long_prompt_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, #{generated := Generated, finish_key := ReportedKey}} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 6}),
        FullTokens = prompt_tokens(long_prompt()) ++ Generated,
        FinishKey = key_for_tokens(FullTokens, Cfg),
        ?assertEqual(FinishKey, ReportedKey),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000))
    end).

%% =============================================================================
%% Cache hit on repeat
%% =============================================================================

repeat_prompt_hits_finish_save_path_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, #{generated := Gen1}} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
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
        {ok, #{generated := Gen1, finish_key := ReportedKey1}} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        FullTokens1 = prompt_tokens(long_prompt()) ++ Gen1,
        FullKey1 = key_for_tokens(FullTokens1, Cfg),
        ?assertEqual(FullKey1, ReportedKey1),
        {ok, _} = wait_for_key(FullKey1, 1000),
        %% Turn 2: a longer prompt that strictly prefix-extends turn 1's
        %% live tokens. Use parent_key = FullKey1 to take the session
        %% resume path.
        Extension = list_to_binary(
            stub_detokenize_decimal(FullTokens1) ++ " more tokens for turn two"
        ),
        {ok, _} =
            erllama_model:complete(<<"test_model">>, Extension, #{
                parent_key => FullKey1,
                response_tokens => 2
            }),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
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
            {ok, #{generated := Generated}} = erllama_model:complete(
                <<"test_model">>, long_prompt(), #{response_tokens => 8}
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
        ok = erllama_model:evict(<<"test_model">>),
        ?assertEqual(0, length(erllama_cache_meta_srv:dump())),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
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
                        <<"test_model">>, long_prompt(), #{response_tokens => 200}
                    )}
        end),
        %% The reply binding (above) is now a map; downstream just
        %% receives the whole {ok, Map} tuple as `_` since this test
        %% only cares about save side-effects.
        %% Give the gen_statem a beat to enter generating.
        timer:sleep(0),
        ok = erllama_model:evict(<<"test_model">>),
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
        ok = erllama_model:shutdown(<<"test_model">>),
        ?assertEqual(0, length(erllama_cache_meta_srv:dump())),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

%% =============================================================================
%% Counters move under traffic
%% =============================================================================

counters_track_misses_and_saves_test() ->
    with_model(#{}, fun(_Cfg) ->
        Before = erllama_cache:get_counters(),
        {ok, _} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        timer:sleep(50),
        After = erllama_cache:get_counters(),
        %% A fresh prompt is a miss, fires cold + finish saves.
        ?assert(maps:get(misses, After) > maps:get(misses, Before)),
        ?assert(maps:get(saves_cold, After) > maps:get(saves_cold, Before)),
        ?assert(maps:get(saves_finish, After) > maps:get(saves_finish, Before))
    end).

%% =============================================================================
%% Concurrency
%% =============================================================================

concurrent_complete_rejects_with_busy_test() ->
    with_model(#{}, fun(_Cfg) ->
        Parent = self(),
        %% Spawn a slow caller that keeps the gen_statem busy.
        spawn(fun() ->
            Parent ! {first, erllama_model:complete(<<"test_model">>, long_prompt())}
        end),
        timer:sleep(0),
        %% Try a second complete; the gen_statem is in prefilling/
        %% generating and rejects.
        Result = erllama_model:complete(<<"test_model">>, short_prompt()),
        receive
            {first, _} -> ok
        after 5000 -> erlang:error(first_caller_timeout)
        end,
        case Result of
            %% raced and got there first/after
            {ok, _} -> ok;
            %% busy as expected
            {error, busy} -> ok
        end
    end).

%% =============================================================================
%% PR1: completion_result map shape
%% =============================================================================

complete_returns_finish_key_matching_full_tokens_test() ->
    with_model(#{}, fun(Cfg) ->
        {ok, #{
            generated := Generated,
            context_tokens := ContextTokens,
            finish_key := FinishKey,
            cache_hit_kind := HitKind
        }} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        ExpectedTokens = prompt_tokens(long_prompt()) ++ Generated,
        ?assertEqual(ExpectedTokens, ContextTokens),
        ?assertEqual(key_for_tokens(ContextTokens, Cfg), FinishKey),
        ?assertEqual(cold, HitKind)
    end).

complete_finish_key_undefined_when_below_min_tokens_test() ->
    %% min_tokens default 4; the short prompt produces 1 + a small
    %% generation. Force a higher threshold so the finish save is
    %% suppressed and finish_key comes back as undefined.
    with_model(#{min_tokens => 10_000}, fun(_Cfg) ->
        {ok, #{finish_key := FinishKey}} =
            erllama_model:complete(<<"test_model">>, short_prompt(), #{response_tokens => 2}),
        ?assertEqual(undefined, FinishKey)
    end).

complete_committed_tokens_equals_context_tokens_length_test() ->
    with_model(#{}, fun(_Cfg) ->
        {ok, #{context_tokens := Ctx, committed_tokens := N}} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{response_tokens => 4}),
        ?assertEqual(length(Ctx), N)
    end).

prefill_only_returns_finish_key_and_warm_resumes_test() ->
    with_model(#{}, fun(Cfg) ->
        Tokens = prompt_tokens(long_prompt()),
        {ok, #{
            context_tokens := Ctx,
            committed_tokens := N,
            finish_key := FinishKey,
            cache_hit_kind := HitKind
        }} = erllama_model:prefill_only(<<"test_model">>, Tokens),
        ?assertEqual(Tokens, Ctx),
        ?assertEqual(length(Tokens), N),
        ?assertEqual(cold, HitKind),
        ?assertEqual(key_for_tokens(Tokens, Cfg), FinishKey),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000)),
        %% Now resume from FinishKey; the cache should report exact hit.
        {ok, #{cache_hit_kind := exact}} =
            erllama_model:complete(
                <<"test_model">>, long_prompt(), #{
                    parent_key => FinishKey,
                    response_tokens => 2
                }
            )
    end).

%% =============================================================================
%% PR2: per-model observability snapshot (phase / pending_len /
%% last_cache_hit) readable lock-free from outside the gen_statem
%% =============================================================================

phase_starts_idle_and_reflects_state_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(idle, erllama:phase(<<"test_model">>)),
        {ok, _} = erllama_model:complete(<<"test_model">>, short_prompt()),
        %% Back to idle after the synchronous complete returns.
        ?assertEqual(idle, erllama:phase(<<"test_model">>))
    end).

phase_unknown_model_returns_idle_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(idle, erllama:phase(<<"never_loaded">>))
    end).

pending_len_zero_when_idle_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(0, erllama:pending_len(<<"test_model">>))
    end).

pending_len_increments_when_queued_test() ->
    %% Streaming infer with a huge response_tokens keeps the
    %% gen_statem in `generating` long enough to admit a queued
    %% second request. The pending_len read must come back
    %% instantly without serialising behind the in-flight decode —
    %% that's the whole point of the obs ETS table.
    with_model(#{}, fun(_Cfg) ->
        {ok, PromptTokens} = erllama_model:tokenize(<<"test_model">>, <<"hi">>),
        {ok, Ref1} = erllama_model:infer(
            <<"test_model">>, PromptTokens, #{response_tokens => 10000}, self()
        ),
        %% Wait for the first token so we know the model is past
        %% prefill and actively decoding.
        receive
            {erllama_token, Ref1, _} -> ok;
            {erllama_token_id, Ref1, _} -> ok
        after 2000 -> erlang:error(no_first_token)
        end,
        %% Issue a second infer from a separate process — the
        %% gen_statem:call for a queued request blocks until that
        %% request is dispatched, which only happens after Ref1
        %% finishes. From a worker we can fire the call and read
        %% pending_len on the test process while the worker waits.
        Parent = self(),
        spawn_link(fun() ->
            R2 =
                erllama_model:infer(
                    <<"test_model">>, PromptTokens, #{response_tokens => 1}, Parent
                ),
            Parent ! {worker_returned, R2}
        end),
        %% Give the second call a beat to enter the gen_statem
        %% mailbox and land in `handle_common -> enqueue`.
        ok = wait_for_pending_len(<<"test_model">>, 1, 2000),
        %% Drain: cancel Ref1, then expect Ref2 to dispatch.
        ok = erllama_model:cancel(Ref1),
        receive
            {erllama_done, Ref1, _} -> ok
        after 5000 -> erlang:error(timeout_first)
        end,
        Ref2 =
            receive
                {worker_returned, {ok, R}} -> R
            after 5000 -> erlang:error(worker_timeout)
            end,
        receive
            {erllama_done, Ref2, _} -> ok
        after 5000 -> erlang:error(timeout_second)
        end,
        ?assertEqual(0, erllama:pending_len(<<"test_model">>))
    end).

%% Poll pending_len until it reaches Expected or Timeout expires.
%% Returns ok on success, raises on timeout. Used by the queue
%% observability test where the second infer call blocks until the
%% first finishes, so we cannot assert pending_len synchronously
%% after a `{ok, Ref}` return.
wait_for_pending_len(ModelId, Expected, TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_for_pending_len_loop(ModelId, Expected, Deadline).

wait_for_pending_len_loop(ModelId, Expected, Deadline) ->
    case erllama:pending_len(ModelId) of
        Expected ->
            ok;
        _ ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    erlang:error({pending_len_timeout, ModelId, Expected});
                false ->
                    timer:sleep(10),
                    wait_for_pending_len_loop(ModelId, Expected, Deadline)
            end
    end.

last_cache_hit_undefined_before_any_request_test() ->
    with_model(#{}, fun(_Cfg) ->
        ?assertEqual(undefined, erllama:last_cache_hit(<<"test_model">>))
    end).

last_cache_hit_after_cold_admission_reports_cold_test() ->
    %% Cold admission populates the obs row with kind=cold and
    %% prefix_len=0. Distinct from `undefined` (model never
    %% admitted anything) so external routers can tell them apart.
    with_model(#{}, fun(_Cfg) ->
        {ok, _} = erllama_model:complete(<<"test_model">>, long_prompt(), #{
            response_tokens => 2
        }),
        ?assertEqual(
            #{kind => cold, prefix_len => 0},
            erllama:last_cache_hit(<<"test_model">>)
        )
    end).

last_cache_hit_after_warm_resume_test() ->
    with_model(#{}, fun(_Cfg) ->
        Tokens = prompt_tokens(long_prompt()),
        {ok, #{finish_key := FinishKey}} =
            erllama_model:prefill_only(<<"test_model">>, Tokens),
        ?assertMatch({ok, _Row}, wait_for_key(FinishKey, 1000)),
        %% Resume from FinishKey: the exact key for `Tokens` is now
        %% in the cache, so lookup_or_resume hits the exact path
        %% before consulting parent_key.
        {ok, _} = erllama_model:complete(
            <<"test_model">>, long_prompt(), #{
                parent_key => FinishKey,
                response_tokens => 2
            }
        ),
        Hit = erllama:last_cache_hit(<<"test_model">>),
        ?assertMatch(#{kind := exact, prefix_len := _}, Hit),
        #{prefix_len := PrefixLen} = Hit,
        ?assertEqual(length(Tokens), PrefixLen)
    end).

model_info_carries_phase_and_pending_len_test() ->
    with_model(#{}, fun(_Cfg) ->
        Info = erllama_model:model_info(<<"test_model">>),
        ?assertEqual(idle, maps:get(phase, Info)),
        ?assertEqual(0, maps:get(pending_len, Info)),
        ?assertEqual(undefined, maps:get(last_cache_hit, Info)),
        %% Existing keys preserved.
        ?assertEqual(idle, maps:get(status, Info))
    end).

%% =============================================================================
%% Helpers
%% =============================================================================

%% =============================================================================
%% Multi-sequence scheduler (n_seq_max > 1)
%% =============================================================================

%% Two concurrent complete/3 calls with n_seq_max=2 should both
%% return their own results. The gen_statem dispatches them to
%% seq_ids 0 and 1, co-batches their decode through one step/2
%% call per tick, and replies to each caller at its own finish.
two_concurrent_completes_each_return_own_result_test_() ->
    {timeout, 10, fun two_concurrent_completes_each_return_own_result_/0}.

two_concurrent_completes_each_return_own_result_() ->
    ConfigOverrides = #{
        context_opts => #{n_seq_max => 2, n_batch => 64}
    },
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        Parent = self(),
        Pid1 = spawn_link(fun() ->
            Parent !
                {one,
                    erllama_model:complete(
                        <<"test_model">>, <<"hi">>, #{response_tokens => 2}
                    )}
        end),
        Pid2 = spawn_link(fun() ->
            Parent !
                {two,
                    erllama_model:complete(
                        <<"test_model">>, <<"yo">>, #{response_tokens => 2}
                    )}
        end),
        Reply1 =
            receive
                {one, R} -> R
            after 5000 -> erlang:error(timeout_one)
            end,
        Reply2 =
            receive
                {two, R2} -> R2
            after 5000 -> erlang:error(timeout_two)
            end,
        ?assertMatch({ok, #{reply := _}}, Reply1),
        ?assertMatch({ok, #{reply := _}}, Reply2),
        %% Drain link signals.
        _ = Pid1,
        _ = Pid2,
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

%% Once a request finishes its seq_id must return to the idle pool
%% so a subsequent admission can reuse it.
seq_id_freed_on_finish_test_() ->
    {timeout, 10, fun seq_id_freed_on_finish_/0}.

seq_id_freed_on_finish_() ->
    ConfigOverrides = #{
        context_opts => #{n_seq_max => 2, n_batch => 64}
    },
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        %% Run three sequential completes against an n_seq_max=2
        %% model. If finish doesn't recycle seq_ids, the third call
        %% would block forever.
        lists:foreach(
            fun(_) ->
                {ok, _} = erllama_model:complete(
                    <<"test_model">>, <<"hi">>, #{response_tokens => 2}
                )
            end,
            lists:seq(1, 3)
        ),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

%% Once n_seq_max admits are in flight, the next admit queues in
%% pending. Fire 3 concurrent infers against an n_seq_max=2 model;
%% the third must queue and only dispatch after one of the first
%% two finishes.
pending_fifo_fills_when_seq_ids_exhausted_test_() ->
    {timeout, 10, fun pending_fifo_fills_when_seq_ids_exhausted_/0}.

pending_fifo_fills_when_seq_ids_exhausted_() ->
    ConfigOverrides = #{
        context_opts => #{n_seq_max => 2, n_batch => 64}
    },
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        {ok, Tokens} = erllama_model:tokenize(<<"test_model">>, <<"hi">>),
        %% Three concurrent admits. With n_seq_max=2 the third is
        %% queued. All three should ultimately return their refs
        %% and emit a done message.
        Parent = self(),
        Spawn = fun(N) ->
            spawn_link(fun() ->
                R = erllama_model:infer(
                    <<"test_model">>, Tokens, #{response_tokens => 2}, Parent
                ),
                Parent ! {ref, N, R}
            end)
        end,
        _ = [Spawn(N) || N <- [a, b, c]],
        Refs = lists:map(
            fun(_N) ->
                receive
                    {ref, _, {ok, Ref}} -> Ref
                after 5000 -> erlang:error(no_ref)
                end
            end,
            [a, b, c]
        ),
        %% Each Ref should emit at least one erllama_done.
        lists:foreach(
            fun(Ref) ->
                receive
                    {erllama_done, Ref, _} -> ok
                after 5000 -> erlang:error({no_done, Ref})
                end
            end,
            Refs
        ),
        %% Drain any stray token messages.
        drain_messages()
    end).

%% =============================================================================
%% Chunked prefill (PR5)
%% =============================================================================

%% A small prefill_chunk_size forces the slicer to split a 12-token
%% prompt across multiple ticks. The cursor advance must track the
%% actual slice length sent each tick: if it over- or under-counts,
%% the final context_tokens diverges from prompt ++ generated, the
%% finish key changes, and this test fails.
prefill_cursor_advances_in_chunks_test() ->
    with_model(#{prefill_chunk_size => 2}, fun(Cfg) ->
        {ok, #{generated := Gen, finish_key := FinishKey}} =
            erllama_model:complete(<<"test_model">>, long_prompt(), #{
                response_tokens => 3
            }),
        FullTokens = prompt_tokens(long_prompt()) ++ Gen,
        ExpectedKey = key_for_tokens(FullTokens, Cfg),
        ?assertEqual(ExpectedKey, FinishKey),
        ?assertEqual(3, length(Gen)),
        ?assertEqual(idle, erllama_model:status(<<"test_model">>))
    end).

%% Cold save must fire only when the trimmed prefix is fully
%% prefilled, not after each chunk. With prefill_chunk_size=2 the
%% trim is split across several ticks; the cold save fires once at
%% the end, capturing exactly the trim tokens.
prefill_chunks_cold_save_at_trim_boundary_test() ->
    with_model(#{prefill_chunk_size => 2}, fun(Cfg) ->
        {ok, _} = erllama_model:complete(<<"test_model">>, long_prompt(), #{
            response_tokens => 2
        }),
        Tokens = prompt_tokens(long_prompt()),
        ColdKey = key_for_tokens(Tokens, Cfg),
        ?assertMatch({ok, _Row}, wait_for_key(ColdKey, 1000))
    end).

%% Default prefill_chunk_size is max(64, n_batch div 4).
prefill_chunk_size_default_test() ->
    ConfigOverrides = #{context_opts => #{n_batch => 1024}},
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        Policy = erllama_model:get_policy(<<"test_model">>),
        ?assertEqual(256, maps:get(prefill_chunk_size, Policy))
    end).

prefill_chunk_size_default_floor_test() ->
    ConfigOverrides = #{context_opts => #{n_batch => 64}},
    with_model(#{}, ConfigOverrides, fun(_Cfg) ->
        Policy = erllama_model:get_policy(<<"test_model">>),
        ?assertEqual(64, maps:get(prefill_chunk_size, Policy))
    end).

drain_messages() ->
    receive
        _ -> drain_messages()
    after 0 -> ok
    end.

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
