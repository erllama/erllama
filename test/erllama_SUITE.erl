%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% End-to-end Common Test suite for the cache subsystem.
%%
%% Boots the full erllama application, runs realistic scenarios, and
%% verifies counters, supervision behaviour, and concurrent load.
%% Intended as the highest-level guard rail before a release;
%% individual modules are still covered by eunit and PropEr.
%% @end
-module(erllama_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").
-include("erllama_cache.hrl").

-export([
    all/0,
    init_per_suite/1,
    end_per_suite/1,
    init_per_testcase/2,
    end_per_testcase/2
]).

-export([
    cold_then_warm_counters/1,
    multi_turn_session_resume/1,
    longest_prefix_resume_without_parent_key/1,
    warm_read_pins_via_checkout/1,
    eviction_drops_files_and_meta/1,
    concurrent_complete_under_writer_cap/1,
    counters_visible_via_facade/1
]).

%% =============================================================================
%% Common Test boilerplate
%% =============================================================================

all() ->
    [
        cold_then_warm_counters,
        multi_turn_session_resume,
        longest_prefix_resume_without_parent_key,
        warm_read_pins_via_checkout,
        eviction_drops_files_and_meta,
        concurrent_complete_under_writer_cap,
        counters_visible_via_facade
    ].

init_per_suite(Config) ->
    {ok, _Apps} = application:ensure_all_started(erllama),
    Config.

end_per_suite(_Config) ->
    application:stop(erllama),
    ok.

init_per_testcase(TC, Config) ->
    PrivDir = ?config(priv_dir, Config),
    Dir = filename:join(PrivDir, atom_to_list(TC) ++ "_dir"),
    ok = filelib:ensure_path(Dir),
    %% Clear any rows from previous testcases. Cache keys are stable
    %% across testcases (same fingerprint + ctx + tokens), so a stale
    %% row would shadow a fresh save and the writer would short-circuit
    %% on `already_present`.
    {evicted, _} = erllama_cache_meta_srv:gc(),
    DiskSrv = list_to_atom("ct_disk_" ++ atom_to_list(TC)),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    Model = iolist_to_binary(["ct_model_", atom_to_binary(TC)]),
    {ok, _} = erllama_model:start_link(Model, model_config(DiskSrv)),
    erllama_cache_counters:reset(),
    [{disk_srv, DiskSrv}, {model, Model}, {dir, Dir} | Config].

end_per_testcase(_TC, Config) ->
    catch erllama_model:stop(?config(model, Config)),
    catch gen_server:stop(?config(disk_srv, Config)),
    ok.

model_config(DiskSrv) ->
    #{
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        context_size => 4096,
        policy => #{
            min_tokens => 4,
            cold_min_tokens => 4,
            cold_max_tokens => 1000,
            continued_interval => 4,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 200
        }
    }.

%% =============================================================================
%% Test cases
%% =============================================================================

cold_then_warm_counters(Config) ->
    Model = ?config(model, Config),
    Prompt = long_prompt(),
    Before = erllama_cache:get_counters(),
    {ok, _} = erllama_model:complete(Model, Prompt, #{response_tokens => 4}),
    %% Allow async saves to publish.
    timer:sleep(100),
    Mid = erllama_cache:get_counters(),
    ?assert(maps:get(misses, Mid) - maps:get(misses, Before) >= 1),
    ?assert(maps:get(saves_cold, Mid) - maps:get(saves_cold, Before) >= 1),
    %% Repeat: same prompt should hit the cold/finish row, not miss.
    {ok, _} = erllama_model:complete(Model, Prompt, #{response_tokens => 4}),
    timer:sleep(50),
    After = erllama_cache:get_counters(),
    ?assert(maps:get(hits_exact, After) - maps:get(hits_exact, Mid) >= 1),
    ok.

multi_turn_session_resume(Config) ->
    %% The phash2-based stub tokenizer is one-way, so we can't
    %% construct a turn-2 prompt whose tokens are turn-1-full-live ++
    %% extension via real text. Instead: pre-publish a parent row
    %% whose stored tokens match a known prefix of the turn-2 prompt.
    Model = ?config(model, Config),
    DiskSrv = ?config(disk_srv, Config),
    %% Pick three words; their phash2 token IDs become our known prefix.
    Words = ["alpha", "beta", "gamma"],
    PrefixTokens = [
        erlang:phash2(list_to_binary(W)) rem (1 bsl 32)
     || W <- Words
    ],
    ParentKey = key_for_tokens(PrefixTokens),
    %% Manually publish the parent slab via the writer.
    BuildMeta = #{
        save_reason => cold,
        quant_bits => 16,
        fingerprint => binary:copy(<<16#AA>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => PrefixTokens,
        context_size => 4096,
        prompt_text => <<>>,
        hostname => <<"ct">>,
        erllama_version => <<"0.1.0">>
    },
    {ok, _Pub} = erllama_cache_writer:save(
        DiskSrv,
        disk,
        BuildMeta,
        erllama_cache_key:encode_tokens(PrefixTokens)
    ),
    %% Wait until visible.
    wait_until(
        fun() -> erllama_cache_meta_srv:lookup_exact(ParentKey) =/= miss end,
        1000
    ),
    Before = erllama_cache:get_counters(),
    Prompt2 = <<"alpha beta gamma extra">>,
    {ok, _} = erllama_model:complete(Model, Prompt2, #{
        parent_key => ParentKey,
        response_tokens => 2
    }),
    After = erllama_cache:get_counters(),
    ?assertEqual(
        1,
        maps:get(hits_resume, After) - maps:get(hits_resume, Before)
    ),
    ok.

%% Stateless callers (HTTP front-end, agent loops) re-send the full
%% conversation each turn. Without parent_key, the longest-prefix
%% scan should still find the previous turn's saved row by walking
%% backward through stride-aligned prefixes.
longest_prefix_resume_without_parent_key(Config) ->
    Model = ?config(model, Config),
    Prompt1 = long_prompt(),
    %% Stub tokenizer splits on spaces, so appending a new word
    %% extends the token list cleanly without retokenization noise.
    Prompt2 = <<Prompt1/binary, " extra-suffix-token">>,
    {ok, _} = erllama_model:complete(Model, Prompt1, #{response_tokens => 2}),
    timer:sleep(150),
    Before = erllama_cache:get_counters(),
    {ok, _} = erllama_model:complete(Model, Prompt2, #{response_tokens => 2}),
    After = erllama_cache:get_counters(),
    Resumed = maps:get(hits_longest_prefix, After) - maps:get(hits_longest_prefix, Before),
    ?assert(Resumed >= 1),
    ok.

%% Warm reads must go through checkout/checkin, not raw lookup +
%% load, so eviction can't delete the file mid-load. Asserts that a
%% completion run after a warm cache hit bumps both ?POS_HITS (only
%% bumped inside do_checkout) and ?C_HITS_EXACT.
warm_read_pins_via_checkout(Config) ->
    Model = ?config(model, Config),
    Prompt = long_prompt(),
    {ok, _} = erllama_model:complete(Model, Prompt, #{response_tokens => 2}),
    timer:sleep(100),
    Before = erllama_cache:get_counters(),
    {ok, _} = erllama_model:complete(Model, Prompt, #{response_tokens => 2}),
    After = erllama_cache:get_counters(),
    ?assertEqual(
        1,
        maps:get(hits_exact, After) - maps:get(hits_exact, Before)
    ),
    %% Find the cold-save row in the meta and confirm hits >= 1.
    [Row | _] = erllama_cache_meta_srv:dump(),
    ?assert(element(?POS_HITS, Row) >= 1),
    ok.

eviction_drops_files_and_meta(Config) ->
    Model = ?config(model, Config),
    Dir = ?config(dir, Config),
    {ok, _} =
        erllama_model:complete(Model, long_prompt(), #{response_tokens => 4}),
    %% Wait for the async save to publish before checking the
    %% directory; counters lag stub completes by a writer-pool hop.
    wait_until(
        fun() ->
            case file:list_dir(Dir) of
                {ok, L} -> length(L) >= 1;
                _ -> false
            end
        end,
        2000
    ),
    {ok, EntriesBefore} = file:list_dir(Dir),
    {evicted, N} = erllama_cache_meta_srv:gc(),
    ?assert(N >= 1),
    {ok, EntriesAfter} = file:list_dir(Dir),
    ?assert(length(EntriesAfter) =< length(EntriesBefore)),
    ok.

concurrent_complete_under_writer_cap(Config) ->
    %% Spawn N parallel models with distinct fingerprints so each
    %% has its own keyspace, then issue a complete to each. The
    %% writer's semaphore is the only shared concurrency control.
    %% Asserts no crashes and every save lands.
    BaseModel = ?config(model, Config),
    DiskSrv = ?config(disk_srv, Config),
    Parent = self(),
    N = 4,
    Pids = [
        spawn(fun() ->
            ModelN = iolist_to_binary([BaseModel, "_", integer_to_binary(I)]),
            Cfg0 = model_config(DiskSrv),
            FP = crypto:hash(sha256, integer_to_binary(I)),
            Cfg = Cfg0#{fingerprint => FP},
            {ok, _} = erllama_model:start_link(ModelN, Cfg),
            try
                {ok, _} = erllama_model:complete(
                    ModelN, long_prompt(), #{response_tokens => 4}
                ),
                Parent ! {done, I}
            after
                catch erllama_model:stop(ModelN)
            end
        end)
     || I <- lists:seq(1, N)
    ],
    %% Drain
    [
        receive
            {done, _} -> ok
        after 5000 -> ct:fail({timeout, P})
        end
     || P <- Pids
    ],
    timer:sleep(200),
    Snap = erllama_cache:get_counters(),
    ?assert(maps:get(saves_cold, Snap) >= N),
    ok.

counters_visible_via_facade(Config) ->
    Model = ?config(model, Config),
    {ok, _} = erllama_model:complete(Model, long_prompt()),
    timer:sleep(100),
    Snap = erllama_cache:get_counters(),
    ?assert(is_map(Snap)),
    ?assert(map_size(Snap) >= 16),
    %% Reset clears.
    ok = erllama_cache:reset_counters(),
    Empty = erllama_cache:get_counters(),
    ?assertEqual(0, maps:get(misses, Empty)),
    ?assertEqual(0, maps:get(saves_cold, Empty)),
    ok.

%% =============================================================================
%% Helpers
%% =============================================================================

long_prompt() ->
    list_to_binary(
        string:join([integer_to_list(N) || N <- lists:seq(1, 12)], " ")
    ).

key_for_tokens(Tokens) ->
    erllama_cache_key:make(#{
        fingerprint => binary:copy(<<16#AA>>, 32),
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => Tokens
    }).

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
                    timer:sleep(20),
                    wait_until_loop(Pred, Deadline)
            end
    end.
