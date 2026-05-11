%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_nif_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% crc32c/1 — RFC 3720 / Castagnoli reference vectors
%% =============================================================================

%% RFC 3720 Appendix B.4 specifies the expected digest for the
%% 9-byte ASCII string "123456789".
crc32c_check_string_test() ->
    ?assertEqual(16#E3069283, erllama_nif:crc32c(<<"123456789">>)).

%% Empty input: by definition the digest of zero bytes is 0.
crc32c_empty_test() ->
    ?assertEqual(0, erllama_nif:crc32c(<<>>)).

%% Single zero byte. Reference value computed offline against the
%% same Castagnoli polynomial used elsewhere (e.g. the Linux kernel's
%% crc32c implementation): crc32c(<<0>>) = 0x527D5351.
crc32c_zero_byte_test() ->
    ?assertEqual(16#527D5351, erllama_nif:crc32c(<<0>>)).

%% 32 zero bytes: reference 0x8A9136AA.
crc32c_zeros_32_test() ->
    Bytes = binary:copy(<<0>>, 32),
    ?assertEqual(16#8A9136AA, erllama_nif:crc32c(Bytes)).

%% 32 0xFF bytes: reference 0x62A8AB43.
crc32c_ones_32_test() ->
    Bytes = binary:copy(<<16#FF>>, 32),
    ?assertEqual(16#62A8AB43, erllama_nif:crc32c(Bytes)).

%% iolist input: must produce the same digest as the flat binary.
crc32c_accepts_iolist_test() ->
    Flat = <<"123456789">>,
    IoList = [<<"12">>, "345", [<<"6">>, [<<"78">>, [<<"9">>]]]],
    ?assertEqual(
        erllama_nif:crc32c(Flat),
        erllama_nif:crc32c(IoList)
    ).

%% Incremental decomposition: CRC32C is a Rabin-style polynomial; the
%% public API runs a single shot, but we sanity-check that splitting
%% the input across multiple calls (handled internally by iolist
%% inspection) produces the same digest.
crc32c_split_input_test() ->
    Whole = binary:copy(<<"abcdefghij">>, 100),
    Split = [binary:part(Whole, 0, 500), binary:part(Whole, 500, 500)],
    ?assertEqual(
        erllama_nif:crc32c(Whole),
        erllama_nif:crc32c(Split)
    ).

crc32c_badarg_atom_test() ->
    ?assertError(badarg, erllama_nif:crc32c(not_iodata)).

crc32c_badarg_integer_test() ->
    ?assertError(badarg, erllama_nif:crc32c(42)).

%% =============================================================================
%% Resource-arg validation
%% =============================================================================

kv_pack_rejects_non_resource_test() ->
    ?assertError(badarg, erllama_nif:kv_pack(make_ref(), [1, 2, 3], 3)).

kv_unpack_rejects_non_resource_test() ->
    ?assertError(badarg, erllama_nif:kv_unpack(make_ref(), <<>>, 0)).

%% First call to nif_load_model triggers the lazy
%% llama_backend_init (pthread_once), which on macOS includes Metal
%% device discovery and can take several seconds. eunit's default
%% 5s case timeout is too tight for this even though the actual
%% load_from_file failure path is fast.
load_model_rejects_non_existent_path_test_() ->
    {timeout, 60, fun() ->
        Result = erllama_nif:load_model(<<"/no/such/file.gguf">>, #{}),
        ?assertMatch({error, _}, Result)
    end}.

%% =============================================================================
%% End-to-end smoke (gated by LLAMA_TEST_MODEL env var)
%% =============================================================================

llama_load_and_tokenize_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping live smoke test", []};
        Path ->
            {timeout, 60, fun() -> live_smoke(list_to_binary(Path)) end}
    end.

live_smoke(Path) ->
    {ok, Model} = erllama_nif:load_model(Path, #{n_gpu_layers => 0}),
    Tokens = erllama_nif:tokenize(Model, <<"Hello world">>, #{
        add_special => true, parse_special => false
    }),
    ?assert(is_list(Tokens)),
    ?assert(length(Tokens) > 0),
    {ok, Ctx} = erllama_nif:new_context(Model, #{n_ctx => 256}),
    Bin = erllama_nif:kv_pack(Ctx, Tokens, length(Tokens)),
    ?assert(is_binary(Bin)),
    ok = erllama_nif:free_context(Ctx),
    ok = erllama_nif:free_model(Model).

%% =============================================================================
%% Hardening guards (gated by LLAMA_TEST_MODEL)
%% =============================================================================

prefill_context_overflow_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                prefill_overflow(
                    list_to_binary(Path),
                    #{n_ctx => 64, n_batch => 64},
                    100,
                    context_overflow
                )
            end}
    end.

prefill_batch_overflow_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                prefill_overflow(
                    list_to_binary(Path),
                    #{n_ctx => 4096, n_batch => 32},
                    64,
                    batch_overflow
                )
            end}
    end.

embed_context_overflow_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                embed_overflow(
                    list_to_binary(Path),
                    #{n_ctx => 64, n_batch => 64, embeddings => true},
                    100,
                    context_overflow
                )
            end}
    end.

apply_chat_template_invalid_content_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() -> bad_content(list_to_binary(Path)) end}
    end.

prefill_fuzz_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        _Path ->
            {timeout, 300, fun() ->
                true = proper:quickcheck(
                    prop_erllama_nif:prop_prefill_never_crashes(),
                    [{numtests, 100}, {to_file, user}]
                )
            end}
    end.

%% Helpers --------------------------------------------------------------------

prefill_overflow(Path, CtxOpts, NTokens, ExpectedAtom) ->
    {ok, Model} = erllama_nif:load_model(Path, #{n_gpu_layers => 0}),
    {ok, Ctx} = erllama_nif:new_context(Model, CtxOpts),
    try
        ?assertEqual(
            {error, ExpectedAtom},
            erllama_nif:prefill(Ctx, lists:seq(1, NTokens))
        )
    after
        ok = erllama_nif:free_context(Ctx),
        ok = erllama_nif:free_model(Model)
    end.

embed_overflow(Path, CtxOpts, NTokens, ExpectedAtom) ->
    {ok, Model} = erllama_nif:load_model(Path, #{n_gpu_layers => 0}),
    {ok, Ctx} = erllama_nif:new_context(Model, CtxOpts),
    try
        ?assertEqual(
            {error, ExpectedAtom},
            erllama_nif:embed(Ctx, lists:seq(1, NTokens))
        )
    after
        ok = erllama_nif:free_context(Ctx),
        ok = erllama_nif:free_model(Model)
    end.

bad_content(Path) ->
    {ok, Model} = erllama_nif:load_model(Path, #{n_gpu_layers => 0}),
    Req = #{
        messages => [
            #{
                role => <<"user">>,
                content => [
                    #{
                        <<"type">> => <<"text">>,
                        <<"text">> => <<"hi">>
                    }
                ]
            }
        ]
    },
    try
        ?assertEqual(
            {error, invalid_content},
            erllama_nif:apply_chat_template(Model, Req)
        )
    after
        ok = erllama_nif:free_model(Model)
    end.
