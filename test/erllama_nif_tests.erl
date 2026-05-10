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

load_model_rejects_non_existent_path_test() ->
    Result = erllama_nif:load_model(<<"/no/such/file.gguf">>, #{}),
    ?assertMatch({error, _}, Result).

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
