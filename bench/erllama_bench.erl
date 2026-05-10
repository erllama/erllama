%% @doc
%% End-to-end bench harness. Two scenarios:
%%
%%   cold_vs_warm/2 — single agent, varied prompt lengths. Measures
%%       time-to-first-token + total time for the cold path then the
%%       warm path. Reports cold/warm/speedup.
%%
%%   multi_agent/3  — N agents sharing a system-prompt prefix. One
%%       pre-warm completion populates the cache, then N agents run
%%       concurrently with distinct per-agent task suffixes. Measures
%%       p50/p99 first-token latency across the N agents.
%%
%% Usage from `bench/run.sh`. Each scenario takes a model path and
%% returns a markdown-formatted result block printed to stdout.
%% @end
-module(erllama_bench).

-export([
    cold_vs_warm/2,
    multi_agent/3,
    main/1
]).

-define(SEED_SENTENCE,
    "The quick brown fox jumps over the lazy dog while the curious cat watches "
    "from the windowsill, contemplating the geometry of the afternoon light. "
).

%% =============================================================================
%% Public scenarios
%% =============================================================================

%% Cold-vs-warm across several prompt lengths.
%% PromptTokenTargets :: [pos_integer()] — approximate token counts.
-spec cold_vs_warm(file:name(), [pos_integer()]) -> ok.
cold_vs_warm(ModelPath, PromptTokenTargets) ->
    {ok, _} = ensure_app_started(),
    Tag = "bench_cv_" ++ integer_to_list(erlang:unique_integer([positive])),
    Dir = make_tmp_dir(Tag),
    DiskSrv = list_to_atom("disk_" ++ Tag),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    Model = list_to_atom("model_" ++ Tag),
    {ok, _} = erllama_model:start_link(Model, model_config(ModelPath, DiskSrv)),
    try
        io:format("~n## cold_vs_warm — ~ts~n~n", [filename:basename(ModelPath)]),
        io:format("| prompt tokens | cold (ms) | warm (ms) | speedup | warm hit |~n"),
        io:format("|---:|---:|---:|---:|:--|~n"),
        lists:foreach(
            fun(N) -> run_cold_vs_warm_at(Model, N) end,
            PromptTokenTargets
        ),
        io:format("~n_(times include all of: tokenize → cache lookup → "
                  "prefill or kv_unpack → 1 generated token. Warm includes "
                  "the seq_rm + 1-token re-prefill primer.)_~n")
    after
        catch erllama_model:stop(Model),
        catch gen_server:stop(DiskSrv),
        rm_rf(Dir)
    end,
    ok.

run_cold_vs_warm_at(Model, TokenTarget) ->
    Prompt = generate_prompt(TokenTarget),
    erllama_cache:reset_counters(),
    {ColdUs, _} = timer:tc(
        erllama_model, complete, [Model, Prompt, #{response_tokens => 1}]
    ),
    wait_for_save(2000),
    erllama_cache:reset_counters(),
    {WarmUs, _} = timer:tc(
        erllama_model, complete, [Model, Prompt, #{response_tokens => 1}]
    ),
    Snap = erllama_cache:get_counters(),
    Hit =
        case
            {maps:get(hits_exact, Snap), maps:get(hits_resume, Snap),
                maps:get(hits_longest_prefix, Snap)}
        of
            {1, _, _} -> "exact";
            {_, 1, _} -> "resume";
            {_, _, 1} -> "lp";
            _ -> "miss"
        end,
    Speedup = ColdUs / max(WarmUs, 1),
    io:format(
        "| ~p | ~.1f | ~.1f | ~.2fx | ~s |~n",
        [TokenTarget, ColdUs / 1000.0, WarmUs / 1000.0, Speedup, Hit]
    ),
    ok.

wait_for_save(TimeoutMs) ->
    Deadline = erlang:monotonic_time(millisecond) + TimeoutMs,
    wait_for_save_loop(Deadline).

wait_for_save_loop(Deadline) ->
    Snap = erllama_cache:get_counters(),
    Saves =
        maps:get(saves_cold, Snap) + maps:get(saves_continued, Snap) +
            maps:get(saves_finish, Snap),
    case Saves >= 1 of
        true ->
            ok;
        false ->
            case erlang:monotonic_time(millisecond) > Deadline of
                true ->
                    ok;
                false ->
                    timer:sleep(20),
                    wait_for_save_loop(Deadline)
            end
    end.

%% Many-agent concurrent warm-path bench.
%% After a single pre-warm completion populates the cache, N agents
%% run concurrently with the same shared system prompt + a distinct
%% per-agent task suffix.
-spec multi_agent(file:name(), pos_integer(), pos_integer()) -> ok.
multi_agent(ModelPath, NAgents, SharedTokens) ->
    {ok, _} = ensure_app_started(),
    Tag = "bench_ma_" ++ integer_to_list(erlang:unique_integer([positive])),
    Dir = make_tmp_dir(Tag),
    DiskSrv = list_to_atom("disk_" ++ Tag),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    SharedPrompt = generate_prompt(SharedTokens),
    PrewarmModel = list_to_atom("prewarm_" ++ Tag),
    {ok, _} = erllama_model:start_link(PrewarmModel, model_config(ModelPath, DiskSrv)),
    try
        %% Pre-warm: cold completion that publishes the cache row.
        {_, _, _} = call_complete(PrewarmModel, SharedPrompt, 1),
        catch erllama_model:stop(PrewarmModel),
        timer:sleep(300),
        erllama_cache:reset_counters(),
        Tasks = [
            <<SharedPrompt/binary, " Agent ", (integer_to_binary(I))/binary,
                ": continue with one short observation.">>
         || I <- lists:seq(1, NAgents)
        ],
        Latencies = run_agents_parallel(ModelPath, DiskSrv, Tasks, Tag),
        Sorted = lists:sort(Latencies),
        N = length(Sorted),
        P50 = lists:nth(max(1, N div 2), Sorted),
        P99 = lists:nth(max(1, (N * 99) div 100), Sorted),
        Min = hd(Sorted),
        Max = lists:last(Sorted),
        Mean = lists:sum(Sorted) div max(N, 1),
        Snap = erllama_cache:get_counters(),
        io:format("~n## multi_agent — ~ts (~p agents, ~p shared tokens)~n~n",
                  [filename:basename(ModelPath), NAgents, SharedTokens]),
        io:format("| metric | value |~n|---|---:|~n"),
        io:format("| min latency (ms) | ~.1f |~n", [Min / 1000.0]),
        io:format("| p50 latency (ms) | ~.1f |~n", [P50 / 1000.0]),
        io:format("| p99 latency (ms) | ~.1f |~n", [P99 / 1000.0]),
        io:format("| max latency (ms) | ~.1f |~n", [Max / 1000.0]),
        io:format("| mean latency (ms) | ~.1f |~n", [Mean / 1000.0]),
        io:format("| hits_exact | ~p |~n", [maps:get(hits_exact, Snap)]),
        io:format("| hits_resume | ~p |~n", [maps:get(hits_resume, Snap)]),
        io:format("| hits_longest_prefix | ~p |~n", [maps:get(hits_longest_prefix, Snap)]),
        io:format("| misses | ~p |~n", [maps:get(misses, Snap)]),
        io:format("| longest_prefix_probes | ~p |~n",
                  [maps:get(longest_prefix_probes, Snap)]),
        io:format("| longest_prefix_ns | ~p |~n", [maps:get(longest_prefix_ns, Snap)]),
        io:format("| load_total_ns | ~p |~n", [maps:get(load_total_ns, Snap)])
    after
        catch erllama_model:stop(PrewarmModel),
        catch gen_server:stop(DiskSrv),
        rm_rf(Dir)
    end,
    ok.

run_agents_parallel(ModelPath, DiskSrv, Tasks, Tag) ->
    Parent = self(),
    Pids = [
        spawn(fun() -> run_one_agent(ModelPath, DiskSrv, I, Task, Parent, Tag) end)
     || {I, Task} <- enumerate(Tasks)
    ],
    [
        receive
            {agent_done, _Pid, Lat} -> Lat
        after 120000 -> ct:fail({timeout, Pid})
        end
     || Pid <- Pids
    ].

run_one_agent(ModelPath, DiskSrv, I, Task, Parent, Tag) ->
    ModelName = list_to_atom("agent_" ++ Tag ++ "_" ++ integer_to_list(I)),
    {ok, _} = erllama_model:start_link(ModelName, model_config(ModelPath, DiskSrv)),
    try
        {Lat, _} = timer:tc(erllama_model, complete, [ModelName, Task, #{response_tokens => 1}]),
        Parent ! {agent_done, self(), Lat}
    after
        catch erllama_model:stop(ModelName)
    end.

%% =============================================================================
%% Driver
%% =============================================================================

%% main([Mode]) — invoked from bench/run.sh. Mode is "tiny", "large",
%% or "all". Models are taken from LLAMA_BENCH_TINY / LLAMA_BENCH_LARGE
%% env vars; missing models are skipped.
main([Arg]) ->
    %% `erl -s Mod Func Args` passes Args as a list of atoms.
    Mode =
        case Arg of
            tiny -> [tiny];
            large -> [large];
            all -> [tiny, large];
            _ -> [tiny, large]
        end,
    Targets = [
        {tiny, os:getenv("LLAMA_BENCH_TINY", default_tiny())},
        {large, os:getenv("LLAMA_BENCH_LARGE", "")}
    ],
    lists:foreach(
        fun({Tag, Path}) ->
            case lists:member(Tag, Mode) andalso Path =/= "" andalso filelib:is_regular(Path) of
                true -> run_for(Path);
                false ->
                    case lists:member(Tag, Mode) of
                        true -> io:format(standard_error, "skip ~p (no model at ~ts)~n", [Tag, Path]);
                        false -> ok
                    end
            end
        end,
        Targets
    ),
    halt(0);
main([]) ->
    main(["all"]).

run_for(Path) ->
    cold_vs_warm(Path, [512, 1024, 2048]),
    multi_agent(Path, 4, 1024).

default_tiny() ->
    case os:getenv("HOME") of
        false -> "";
        Home -> filename:join([Home, "Models", "tinyllama-1.1b-chat.gguf"])
    end.

%% =============================================================================
%% Helpers
%% =============================================================================

ensure_app_started() ->
    application:set_env(erllama, scheduler, #{enabled => false}),
    application:ensure_all_started(erllama).

call_complete(Model, Prompt, NTokens) ->
    erllama_model:complete(Model, Prompt, #{response_tokens => NTokens}).

generate_prompt(TargetTokens) ->
    %% TinyLlama vocab averages ~3-4 chars/token. Generate enough to
    %% comfortably exceed the target; the model layer truncates or
    %% rejects nothing — we just want a stable approximate length.
    SeedLen = string:length(?SEED_SENTENCE),
    %% ~3.5 chars/token on average for English with the SentencePiece
    %% tokenizer; pad with a 50% margin so we land near the target.
    Reps = max(1, (TargetTokens * 5) div SeedLen + 1),
    list_to_binary(lists:duplicate(Reps, ?SEED_SENTENCE)).

model_config(Path, DiskSrv) ->
    {ok, Bin} = file:read_file(Path),
    Fp = crypto:hash(sha256, Bin),
    #{
        backend => erllama_model_llama,
        model_path => Path,
        model_opts => #{n_gpu_layers => 0},
        %% n_batch sized to fit our largest bench prompt in a single
        %% llama_decode batch. llama.cpp asserts (and aborts the BEAM)
        %% if n_tokens > n_batch; chunked prefill is a separate
        %% follow-up. For TinyLlama with 2048-token prompts, 4096 is
        %% safely above ceiling.
        context_opts => #{n_ctx => 4096, n_batch => 4096},
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => Fp,
        fingerprint_mode => safe,
        quant_type => q4_k_m,
        quant_bits => 4,
        ctx_params_hash => crypto:hash(sha256, term_to_binary({4096, 4096})),
        context_size => 4096,
        policy => #{
            min_tokens => 32,
            cold_min_tokens => 32,
            cold_max_tokens => 8192,
            continued_interval => 256,
            boundary_trim_tokens => 16,
            boundary_align_tokens => 32,
            session_resume_wait_ms => 500
        }
    }.

make_tmp_dir(Tag) ->
    Base = os:getenv("TMPDIR", "/tmp"),
    Dir = filename:join(Base, "erllama_" ++ Tag),
    ok = filelib:ensure_path(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Entries} -> [file:delete(filename:join(Dir, E)) || E <- Entries];
        _ -> ok
    end,
    file:del_dir(Dir).

enumerate(L) ->
    lists:zip(lists:seq(1, length(L)), L).
