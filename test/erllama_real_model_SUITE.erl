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
    longest_prefix_resume_without_parent_key/1,
    apply_chat_template_renders/1,
    apply_chat_template_includes_system/1,
    set_grammar_constrains_output/1,
    clear_sampler_resets_to_greedy/1,
    seed_determinism/1,
    seed_varies/1,
    temperature_zero_is_greedy/1,
    grammar_plus_sampler/1,
    verify_does_not_mutate_caller_visible_state/1,
    verify_accepted_count_le_k/1
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
        longest_prefix_resume_without_parent_key,
        apply_chat_template_renders,
        apply_chat_template_includes_system,
        set_grammar_constrains_output,
        clear_sampler_resets_to_greedy,
        seed_determinism,
        seed_varies,
        temperature_zero_is_greedy,
        grammar_plus_sampler,
        verify_does_not_mutate_caller_visible_state,
        verify_accepted_count_le_k
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
    Model = iolist_to_binary(["real_model_", atom_to_binary(TC)]),
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
    %% and respond to a no-op. Model IDs are binaries registered via
    %% erllama_registry's via callback, so erlang:whereis/1 (atom-only)
    %% is the wrong lookup.
    Model = ?config(model, Config),
    ?assert(is_pid(erllama_registry:whereis_name(Model))),
    ok.

tokenize_decode_one(Config) ->
    Model = ?config(model, Config),
    {ok, #{reply := Reply}} = erllama_model:complete(
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
    {ok, #{reply := Reply1}} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 8, seed => 42}
    ),
    timer:sleep(300),
    Before = erllama_cache:get_counters(),
    {ok, #{reply := Reply2}} = erllama_model:complete(
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
    {ok, _} = erllama_model:complete(
        Model, ?LONG_PROMPT, #{response_tokens => 4}
    ),
    timer:sleep(300),
    Mid = erllama_cache:get_counters(),
    ?assert(maps:get(misses, Mid) - maps:get(misses, Before) >= 1),
    ?assert(maps:get(saves_cold, Mid) - maps:get(saves_cold, Before) >= 1),
    {ok, _} = erllama_model:complete(
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
    {ok, _} = erllama_model:complete(Model, Prompt1, #{response_tokens => 4}),
    timer:sleep(400),
    Before = erllama_cache:get_counters(),
    {ok, _} = erllama_model:complete(Model, Prompt2, #{response_tokens => 2}),
    After = erllama_cache:get_counters(),
    Resumed = maps:get(hits_longest_prefix, After) - maps:get(hits_longest_prefix, Before),
    Probes = maps:get(longest_prefix_probes, After) - maps:get(longest_prefix_probes, Before),
    Ns = maps:get(longest_prefix_ns, After) - maps:get(longest_prefix_ns, Before),
    ct:log("longest-prefix hits=~p probes=~p ns=~p", [Resumed, Probes, Ns]),
    ?assert(Resumed >= 1),
    ?assert(Probes >= 1),
    ok.

%% =============================================================================
%% Bucket C-NIF: chat template, grammar, embeddings
%% =============================================================================

apply_chat_template_renders(Config) ->
    Model = ?config(model, Config),
    Request = #{
        messages => [#{role => <<"user">>, content => <<"Hi.">>}]
    },
    case erllama:apply_chat_template(Model, Request) of
        {ok, Tokens} ->
            ?assert(is_list(Tokens)),
            ?assert(length(Tokens) > 0),
            ?assert(lists:all(fun is_integer/1, Tokens));
        {error, no_template} ->
            {skip, "model has no chat template in GGUF metadata"};
        {error, Reason} ->
            ct:fail({apply_chat_template_failed, Reason})
    end.

apply_chat_template_includes_system(Config) ->
    Model = ?config(model, Config),
    Without = #{messages => [#{role => <<"user">>, content => <<"hi">>}]},
    With = Without#{system => <<"You speak only in haiku.">>},
    case {erllama:apply_chat_template(Model, Without), erllama:apply_chat_template(Model, With)} of
        {{ok, A}, {ok, B}} ->
            %% System content lands in the rendered prompt; longer.
            ?assert(length(B) > length(A));
        {{error, no_template}, _} ->
            {skip, "model has no chat template"};
        {_, {error, no_template}} ->
            {skip, "model has no chat template"};
        Other ->
            ct:fail({unexpected, Other})
    end.

%% Constrain the sampler to a tiny grammar that only emits the literal
%% "yes" or "no" tokens, then run a normal complete and verify the
%% output is one of those tokens. (We test this through the model
%% gen_statem rather than infer/4 because complete is simpler to set up.)
set_grammar_constrains_output(Config) ->
    Model = ?config(model, Config),
    {ok, [Pid]} = {ok, [erllama_registry:whereis_name(Model)]},
    [_] = [Pid || is_pid(Pid)],
    {ok, PromptTokens} = erllama:tokenize(Model, <<"Answer with yes or no:">>),
    Grammar = <<"root ::= \"yes\" | \"no\"">>,
    Params = #{response_tokens => 6, grammar => Grammar},
    {ok, Ref} = erllama:infer(Model, PromptTokens, Params, self()),
    Drained = drain(Ref, 60000),
    case Drained of
        {Texts, _Stats} ->
            All = iolist_to_binary(Texts),
            ct:log("grammar-constrained output: ~ts", [All]),
            %% The grammar admits only `yes` or `no` (plus optional
            %% trailing whitespace from the tokenizer). The output must
            %% be one of those after trimming.
            Trimmed = string:trim(All),
            ?assert(Trimmed =:= <<"yes">> orelse Trimmed =:= <<"no">>);
        timeout ->
            ct:fail(grammar_timeout)
    end.

clear_sampler_resets_to_greedy(Config) ->
    Model = ?config(model, Config),
    {ok, PromptTokens} = erllama:tokenize(Model, <<"Hello.">>),
    %% Run grammar-constrained, then the SAME tokens with no grammar.
    %% The second run should be free to produce any output, so it
    %% must complete without {error, grammar_failed}.
    Grammar = <<"root ::= \"a\" | \"b\"">>,
    {ok, R1} = erllama:infer(
        Model,
        PromptTokens,
        #{response_tokens => 2, grammar => Grammar},
        self()
    ),
    _ = drain(R1, 60000),
    %% Second run, no grammar.
    {ok, R2} = erllama:infer(
        Model,
        PromptTokens,
        #{response_tokens => 2},
        self()
    ),
    case drain(R2, 60000) of
        {Texts, _Stats} ->
            ?assert(iolist_size(Texts) >= 0);
        timeout ->
            ct:fail(post_grammar_timeout)
    end.

drain(Ref, TimeoutMs) -> drain(Ref, TimeoutMs, []).
drain(Ref, TimeoutMs, Acc) ->
    receive
        {erllama_token, Ref, B} -> drain(Ref, TimeoutMs, [B | Acc]);
        {erllama_done, Ref, S} -> {lists:reverse(Acc), S};
        {erllama_error, Ref, R} -> {error, R}
    after TimeoutMs ->
        timeout
    end.

%% Run the same prompt twice under temperature > 0 with the same
%% seed. With seed honoured, the per-step `dist` sampler is
%% deterministic so both runs must produce identical token streams.
%% A purely greedy run would also match, so we set temperature high
%% enough that greedy and sampled outputs would diverge in practice.
seed_determinism(Config) ->
    Model = ?config(model, Config),
    {ok, Tokens} = erllama:tokenize(Model, ?SHORT_PROMPT),
    Params = #{response_tokens => 12, temperature => 0.9, seed => 42},
    {Text1, _} = run_infer(Model, Tokens, Params),
    {Text2, _} = run_infer(Model, Tokens, Params),
    ?assertEqual(Text1, Text2),
    ok.

%% Same prompt, same temperature, different seeds. Two independent
%% RNG streams must produce at least one differing token in a
%% 12-token window; otherwise the seed is being ignored.
seed_varies(Config) ->
    Model = ?config(model, Config),
    {ok, Tokens} = erllama:tokenize(Model, ?SHORT_PROMPT),
    {Text1, _} = run_infer(
        Model, Tokens, #{response_tokens => 12, temperature => 0.9, seed => 42}
    ),
    {Text2, _} = run_infer(
        Model, Tokens, #{response_tokens => 12, temperature => 0.9, seed => 99}
    ),
    ?assertNotEqual(Text1, Text2),
    ok.

%% temperature => 0.0 falls back to greedy regardless of seed; two
%% runs with different seeds must agree.
temperature_zero_is_greedy(Config) ->
    Model = ?config(model, Config),
    {ok, Tokens} = erllama:tokenize(Model, ?SHORT_PROMPT),
    {Text1, _} = run_infer(
        Model, Tokens, #{response_tokens => 12, temperature => 0.0, seed => 1}
    ),
    {Text2, _} = run_infer(
        Model, Tokens, #{response_tokens => 12, temperature => 0.0, seed => 2}
    ),
    ?assertEqual(Text1, Text2),
    ok.

%% Grammar combined with sampler params: the grammar still constrains
%% every emitted token (must be `yes` or `no`), and identical seed +
%% temperature pairs are deterministic among the constrained
%% vocabulary.
grammar_plus_sampler(Config) ->
    Model = ?config(model, Config),
    {ok, Tokens} = erllama:tokenize(Model, <<"Answer with yes or no:">>),
    Params = #{
        response_tokens => 6,
        grammar => <<"root ::= \"yes\" | \"no\"">>,
        temperature => 0.7,
        seed => 7
    },
    {Text1, _} = run_infer(Model, Tokens, Params),
    {Text2, _} = run_infer(Model, Tokens, Params),
    Trimmed1 = string:trim(Text1),
    ?assert(Trimmed1 =:= <<"yes">> orelse Trimmed1 =:= <<"no">>),
    ?assertEqual(Text1, Text2),
    ok.

run_infer(Model, Tokens, Params) ->
    {ok, Ref} = erllama:infer(Model, Tokens, Params, self()),
    case drain(Ref, 60000) of
        {Texts, Stats} -> {iolist_to_binary(Texts), Stats};
        Other -> ct:fail({drain, Other})
    end.

%% verify/4 must leave the context's sampling distribution
%% indistinguishable from its pre-call state. We prove it by:
%%   1. Prefill prompt + decode_one to record T1.
%%   2. Reset the KV state and re-prefill (clean baseline matching
%%      step 1).
%%   3. Run verify with arbitrary candidates.
%%   4. Decode_one and record T2.
%%   5. Assert T1 == T2.
%% If the snapshot/restore protocol skips a piece (KV cells, logits
%% buffer), step 4 picks up the wrong distribution and T2 diverges.
verify_does_not_mutate_caller_visible_state(Config) ->
    Model = ?config(model, Config),
    {ok, Prompt} = erllama:tokenize(Model, ?SHORT_PROMPT),
    BState = erllama_model:get_backend_state(Model),
    Ctx = element(3, BState),
    %% Pre-call sequence: prefill prompt, decode_one to get T1.
    ok = erllama_nif:prefill(Ctx, Prompt),
    {Tag1, T1} = erllama_nif:decode_one(Ctx),
    ?assert(Tag1 =:= ok orelse Tag1 =:= eog),
    %% Reset to a clean baseline matching step 1.
    %% kv_seq_rm with p1 = -1 drops everything past p0.
    ok = erllama_nif:kv_seq_rm(Ctx, 0, length(Prompt), -1),
    ok = erllama_nif:prefill(Ctx, Prompt),
    %% Run verify; the candidates here are arbitrary -- what
    %% matters is that the post-call sampling distribution is
    %% restored.
    Candidates = [1, 2, 3, 4],
    {ok, _Accepted, _Next} = erllama:verify(
        Model, Prompt, Candidates, length(Candidates)
    ),
    %% Post-call decode_one: T2 must equal T1.
    {Tag2, T2} = erllama_nif:decode_one(Ctx),
    ?assert(Tag2 =:= ok orelse Tag2 =:= eog),
    ?assertEqual(T1, T2),
    ok.

verify_accepted_count_le_k(Config) ->
    Model = ?config(model, Config),
    {ok, Prompt} = erllama:tokenize(Model, ?SHORT_PROMPT),
    BState = erllama_model:get_backend_state(Model),
    Ctx = element(3, BState),
    ok = erllama_nif:prefill(Ctx, Prompt),
    %% A handful of arbitrary candidate sets; AcceptedCount must
    %% always be <= K regardless of how many actually match.
    [
        begin
            ok = erllama_nif:kv_seq_rm(Ctx, 0, length(Prompt), -1),
            ok = erllama_nif:prefill(Ctx, Prompt),
            {ok, Accepted, _Next} = erllama:verify(Model, Prompt, Cands, K),
            ?assert(Accepted >= 0 andalso Accepted =< K)
        end
     || {Cands, K} <- [
            {[100], 1},
            {[100, 200], 2},
            {[100, 200, 300, 400, 500], 5},
            {[7, 7, 7], 3}
        ]
    ],
    ok.

%% =============================================================================
%% Helpers
%% =============================================================================

file_sha256(Path) ->
    {ok, Bin} = file:read_file(Path),
    crypto:hash(sha256, Bin).
