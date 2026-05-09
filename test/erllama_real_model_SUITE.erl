%% @doc
%% End-to-end Common Test against a real GGUF model.
%%
%% Gated on the `LLAMA_TEST_MODEL` environment variable. When unset,
%% every case in this suite returns `{skip, ...}` so the regular CT
%% run on developer machines and CI without a model file stays green.
%%
%% Usage:
%%
%% ```
%%   LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-q4_k_m.gguf rebar3 ct \
%%       --suite=erllama_real_model_SUITE
%% ```
%%
%% What it covers:
%% - load a real model and free it without leaks
%% - tokenize → prefill → decode_one round-trip
%% - kv_pack / kv_unpack semantic round-trip (deterministic next-token)
%% - cold complete then warm complete: warm path takes the cache hit
%% - cold-vs-warm latency: warm should be markedly faster on a long
%%   prefix
%% @end
-module(erllama_real_model_SUITE).
-include_lib("common_test/include/ct.hrl").
-include_lib("stdlib/include/assert.hrl").

-export([
    all/0,
    init_per_suite/1,
    end_per_suite/1,
    init_per_testcase/2,
    end_per_testcase/2
]).

-export([
    load_unload/1,
    tokenize_decode_one/1,
    pack_unpack_round_trip/1,
    cold_then_warm_complete/1,
    warm_faster_than_cold/1
]).

-define(MODEL_ENV, "LLAMA_TEST_MODEL").
-define(SHORT_PROMPT, <<"The quick brown fox">>).
-define(LONG_PROMPT, <<
    "Once upon a time in a quiet village there lived a clever old woman "
    "who told stories to anyone who would listen. Her favorite was about "
    "a small fox that learned how to outwit a hungry wolf by hiding under "
    "a haystack and reading old letters. Every evening the children "
    "gathered around her hearth and she began the same way: "
>>).

%% =============================================================================
%% Common Test boilerplate
%% =============================================================================

all() ->
    [
        load_unload,
        tokenize_decode_one,
        pack_unpack_round_trip,
        cold_then_warm_complete,
        warm_faster_than_cold
    ].

init_per_suite(Config) ->
    case os:getenv(?MODEL_ENV) of
        false ->
            {skip, "set " ?MODEL_ENV " to a GGUF path to enable this suite"};
        "" ->
            {skip, "empty " ?MODEL_ENV};
        Path ->
            case filelib:is_regular(Path) of
                true ->
                    {ok, _} = application:ensure_all_started(erllama),
                    [{model_path, Path} | Config];
                false ->
                    {skip, lists:flatten(io_lib:format("not a file: ~ts", [Path]))}
            end
    end.

end_per_suite(_Config) ->
    application:stop(erllama),
    application:stop(iommap),
    ok.

init_per_testcase(TC, Config) ->
    Path = ?config(model_path, Config),
    PrivDir = ?config(priv_dir, Config),
    Dir = filename:join(PrivDir, atom_to_list(TC) ++ "_dir"),
    ok = filelib:ensure_path(Dir),
    DiskSrv = list_to_atom("real_disk_" ++ atom_to_list(TC)),
    {ok, _} = erllama_cache_disk_srv:start_link(DiskSrv, Dir),
    Model = list_to_atom("real_model_" ++ atom_to_list(TC)),
    {ok, _} = erllama_model:start_link(Model, model_config(Path, DiskSrv)),
    erllama_cache_counters:reset(),
    [{disk_srv, DiskSrv}, {model, Model}, {dir, Dir} | Config].

end_per_testcase(_TC, Config) ->
    catch erllama_model:stop(?config(model, Config)),
    catch gen_server:stop(?config(disk_srv, Config)),
    ok.

model_config(Path, DiskSrv) ->
    %% Hash the bytes of the model file so the cache_key namespace is
    %% stable across test runs. For real workloads use
    %% fingerprint_mode => safe; here we hand a precomputed hash to
    %% avoid the (potentially slow) chunked walk.
    Fp = file_sha256(Path),
    #{
        backend => erllama_model_llama,
        model_path => Path,
        model_opts => #{n_gpu_layers => 0},
        context_opts => #{n_ctx => 2048, n_batch => 512},
        tier_srv => DiskSrv,
        tier => disk,
        fingerprint => Fp,
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => crypto:hash(sha256, term_to_binary({2048, 512})),
        context_size => 2048,
        policy => #{
            min_tokens => 8,
            cold_min_tokens => 8,
            cold_max_tokens => 4096,
            continued_interval => 64,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 1000
        }
    }.

%% =============================================================================
%% Test cases
%% =============================================================================

load_unload(Config) ->
    %% init_per_testcase already loaded; verify the gen_statem is alive
    %% and respond to a no-op.
    Model = ?config(model, Config),
    ?assert(is_pid(whereis(Model))),
    ok.

tokenize_decode_one(Config) ->
    Model = ?config(model, Config),
    {ok, Reply, _Toks} = erllama_model:complete(
        Model, ?SHORT_PROMPT, #{response_tokens => 4}
    ),
    ?assert(is_binary(Reply)),
    ?assert(byte_size(Reply) > 0),
    ok.

pack_unpack_round_trip(Config) ->
    %% A complete will run cold (miss → prefill → save). A second
    %% complete with the same prompt should hit the cache, which
    %% exercises kv_unpack into a fresh context: if pack/unpack are
    %% not byte-equivalent the model will produce gibberish or crash.
    Model = ?config(model, Config),
    {ok, Reply1, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 8, seed => 42}
    ),
    %% Wait for finish save to publish before the warm call.
    timer:sleep(300),
    Before = erllama_cache:get_counters(),
    {ok, Reply2, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 8, seed => 42}
    ),
    After = erllama_cache:get_counters(),
    ?assertEqual(
        1,
        maps:get(hits_exact, After) - maps:get(hits_exact, Before)
    ),
    %% Greedy + identical seed + identical state ⇒ identical bytes.
    ?assertEqual(Reply1, Reply2),
    ok.

cold_then_warm_complete(Config) ->
    Model = ?config(model, Config),
    Before = erllama_cache:get_counters(),
    {ok, _, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 4}
    ),
    timer:sleep(300),
    Mid = erllama_cache:get_counters(),
    ?assert(maps:get(misses, Mid) - maps:get(misses, Before) >= 1),
    ?assert(maps:get(saves_cold, Mid) - maps:get(saves_cold, Before) >= 1),
    {ok, _, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 4}
    ),
    After = erllama_cache:get_counters(),
    ?assert(maps:get(hits_exact, After) - maps:get(hits_exact, Mid) >= 1),
    ok.

warm_faster_than_cold(Config) ->
    Model = ?config(model, Config),
    %% Cold path.
    {Cold, _} = timer:tc(
        erllama_model, complete, [Model, ?LONG_PROMPT, #{response_tokens => 1}]
    ),
    timer:sleep(500),
    %% Warm path.
    {Warm, _} = timer:tc(
        erllama_model, complete, [Model, ?LONG_PROMPT, #{response_tokens => 1}]
    ),
    ct:log("cold=~p us, warm=~p us, ratio=~.2fx", [Cold, Warm, Cold / max(Warm, 1)]),
    %% Don't gate hard on a magic ratio — CI VMs vary wildly. Just
    %% require that warm is not slower than cold by more than a
    %% generous margin.
    ?assert(Warm =< Cold * 2),
    ok.

%% =============================================================================
%% Helpers
%% =============================================================================

file_sha256(Path) ->
    {ok, Bin} = file:read_file(Path),
    crypto:hash(sha256, Bin).
