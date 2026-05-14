%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Tests for the multi-sequence backend callbacks on the stub
%% backend. These exercise the behaviour surface directly (no
%% gen_statem in the loop) so the round-trips are observable
%% deterministically — the stub never has to allocate a real
%% llama_context.
-module(erllama_model_stub_tests).
-include_lib("eunit/include/eunit.hrl").

stub_step_prefill_returns_prefilled_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, [{0, prefilled}, {1, prefilled}]} =
        erllama_model_stub:step(S, [
            {0, {prefill, [1, 2, 3]}},
            {1, {prefill, [4, 5, 6]}}
        ]).

stub_step_decode_is_deterministic_per_seq_and_sampler_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler0} = erllama_model_stub:sampler_new(S, #{}),
    {ok, [{0, {token, T1, 0}}]} =
        erllama_model_stub:step(S, [{0, {decode, Sampler0}}]),
    %% Same seq, same sampler -> same token. The stub is deterministic
    %% by design so cache-integration tests can pin outputs.
    {ok, [{0, {token, T2, 0}}]} =
        erllama_model_stub:step(S, [{0, {decode, Sampler0}}]),
    ?assertEqual(T1, T2).

stub_step_decode_differs_across_seqs_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler} = erllama_model_stub:sampler_new(S, #{}),
    %% A scheduler bug that pairs the wrong sampler with the wrong
    %% seq would break this — different seqs must produce different
    %% tokens for the same sampler.
    {ok, Results} =
        erllama_model_stub:step(S, [
            {0, {decode, Sampler}},
            {1, {decode, Sampler}}
        ]),
    [{0, {token, T0, _}}, {1, {token, T1, _}}] = Results,
    ?assertNotEqual(T0, T1).

stub_step_co_batches_prefill_and_decode_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, Sampler} = erllama_model_stub:sampler_new(S, #{}),
    {ok, [{0, {token, T, 0}}, {1, prefilled}]} =
        erllama_model_stub:step(S, [
            {0, {decode, Sampler}},
            {1, {prefill, [10, 20]}}
        ]),
    ?assert(is_integer(T)).

stub_sampler_new_returns_unique_refs_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    {ok, A} = erllama_model_stub:sampler_new(S, #{}),
    {ok, B} = erllama_model_stub:sampler_new(S, #{}),
    ?assertNotEqual(A, B),
    ok = erllama_model_stub:sampler_free(A),
    ok = erllama_model_stub:sampler_free(B).

stub_kv_pack_seq_id_matches_legacy_test() ->
    %% The stub doesn't track per-seq state separately — the binary
    %% comes from encoding the token list — so the seq-aware and
    %% legacy arities should produce identical bytes for the same
    %% token list. Guards against an accidental divergence in the
    %% stub that could mask scheduler bugs.
    {ok, S} = erllama_model_stub:init(#{}),
    Tokens = [11, 22, 33, 44],
    Legacy = erllama_model_stub:kv_pack(S, Tokens),
    SeqAware = erllama_model_stub:kv_pack(S, Tokens, 1),
    ?assertEqual(Legacy, SeqAware).

stub_kv_unpack_seq_id_is_noop_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    ?assertEqual(ok, erllama_model_stub:kv_unpack(S, <<>>, 0)),
    ?assertEqual(ok, erllama_model_stub:kv_unpack(S, <<>>, 1)).

stub_seq_rm_is_noop_test() ->
    {ok, S} = erllama_model_stub:init(#{}),
    ?assertEqual(ok, erllama_model_stub:seq_rm(S, 0)),
    ?assertEqual(ok, erllama_model_stub:seq_rm(S, 7)).
