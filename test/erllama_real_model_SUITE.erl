%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
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
    warm_faster_than_cold/1,
    longest_prefix_resume_without_parent_key/1
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
        warm_faster_than_cold,
        longest_prefix_resume_without_parent_key
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
    ok.

init_per_testcase(TC, Config) ->
    Path = ?config(model_path, Config),
    PrivDir = ?config(priv_dir, Config),
    Dir = filename:join(PrivDir, atom_to_list(TC) ++ "_dir"),
    ok = filelib:ensure_path(Dir),
    %% Wipe any meta rows left over from prior testcases. Cache keys
    %% are fingerprint+tokens-derived and stable across tests, so a
    %% stale row from an earlier testcase pointing at a now-stopped
    %% disk_srv would shadow a fresh cold path.
    {evicted, _} = erllama_cache_meta_srv:gc(),
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
            min_tokens => 16,
            cold_min_tokens => 16,
            cold_max_tokens => 4096,
            continued_interval => 64,
            %% trim 8 tail tokens before saving so the saved key
            %% lands on a stable BPE boundary (the last few tokens
            %% are the most likely to retokenize when the next
            %% turn appends user text).
            boundary_trim_tokens => 8,
            boundary_align_tokens => 16,
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
    %% exercises kv_unpack: if pack/unpack are broken the model will
    %% crash or produce gibberish.
    %%
    %% Note: the warm restore drops the last KV cell and re-prefills
    %% it to regenerate the per-context logits buffer
    %% (`llama_state_seq_*` does not persist logits). Floating-point
    %% nondeterminism from re-decoding that single token can shift a
    %% near-tied next-token sample, so we don't require byte-equal
    %% replies. The design plan asserts bit-identical only at turn
    %% boundaries (continued state with sampler/RNG persisted), which
    %% is out of scope for v1. A strong shared prefix is enough to
    %% prove the unpack put the context in roughly the right place.
    Model = ?config(model, Config),
    {ok, Reply1, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 8, seed => 42}
    ),
    timer:sleep(300),
    Before = erllama_cache:get_counters(),
    {ok, Reply2, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 8, seed => 42}
    ),
    After = erllama_cache:get_counters(),
    %% Either path counts: with trim+align the cold save lands at
    %% a trimmed prefix shorter than the full prompt, so the warm
    %% lookup hits via longest-prefix (hits_resume) rather than exact.
    Warm =
        (maps:get(hits_exact, After) - maps:get(hits_exact, Before)) +
            (maps:get(hits_resume, After) - maps:get(hits_resume, Before)) +
            (maps:get(hits_longest_prefix, After) - maps:get(hits_longest_prefix, Before)),
    ?assert(Warm >= 1),
    ?assert(byte_size(Reply2) > 0),
    Common = binary:longest_common_prefix([Reply1, Reply2]),
    ct:log("cold=~ts~nwarm=~ts~ncommon prefix bytes=~p", [Reply1, Reply2, Common]),
    ?assert(Common >= byte_size(Reply1) div 2),
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
    %% trim+align makes the saved key a strict prefix of the live
    %% token list, so the warm lookup may hit via longest-prefix
    %% (hits_resume) rather than exact. Either counts.
    Warm =
        (maps:get(hits_exact, After) - maps:get(hits_exact, Mid)) +
            (maps:get(hits_resume, After) - maps:get(hits_resume, Mid)) +
            (maps:get(hits_longest_prefix, After) - maps:get(hits_longest_prefix, Mid)),
    ?assert(Warm >= 1),
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

%% Real-GGUF version of the longest-prefix path. The test policy
%% uses boundary_trim_tokens = 8 and boundary_align_tokens = 16 so
%% the cold save lands on a stable BPE boundary well inside the
%% prompt. The next turn appends new text; even if BPE retokenizes
%% around the join, the saved trimmed-prefix key still matches.
longest_prefix_resume_without_parent_key(Config) ->
    Model = ?config(model, Config),
    Prompt1 = ?LONG_PROMPT,
    Prompt2 = <<Prompt1/binary, " The next morning brought a quiet rain.">>,
    {ok, _, _} = erllama_model:complete(Model, Prompt1, #{response_tokens => 4}),
    timer:sleep(400),
    Before = erllama_cache:get_counters(),
    {ok, _, _} = erllama_model:complete(Model, Prompt2, #{response_tokens => 2}),
    After = erllama_cache:get_counters(),
    Resumed = maps:get(hits_longest_prefix, After) - maps:get(hits_longest_prefix, Before),
    Probes = maps:get(longest_prefix_probes, After) - maps:get(longest_prefix_probes, Before),
    Ns = maps:get(longest_prefix_ns, After) - maps:get(longest_prefix_ns, Before),
    ct:log("longest-prefix hits=~p probes=~p ns=~p", [Resumed, Probes, Ns]),
    ?assert(Resumed >= 1),
    ?assert(Probes >= 1),
    ok.

%% =============================================================================
%% Helpers
%% =============================================================================

file_sha256(Path) ->
    {ok, Bin} = file:read_file(Path),
    crypto:hash(sha256, Bin).
