-module(erllama_kv_nif_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% crc32c/1 — RFC 3720 / Castagnoli reference vectors
%% =============================================================================

%% RFC 3720 Appendix B.4 specifies the expected digest for the
%% 9-byte ASCII string "123456789".
crc32c_check_string_test() ->
    ?assertEqual(16#E3069283, erllama_kv_nif:crc32c(<<"123456789">>)).

%% Empty input: by definition the digest of zero bytes is 0.
crc32c_empty_test() ->
    ?assertEqual(0, erllama_kv_nif:crc32c(<<>>)).

%% Single zero byte. Reference value computed offline against the
%% same Castagnoli polynomial used elsewhere (e.g. the Linux kernel's
%% crc32c implementation): crc32c(<<0>>) = 0x527D5351.
crc32c_zero_byte_test() ->
    ?assertEqual(16#527D5351, erllama_kv_nif:crc32c(<<0>>)).

%% 32 zero bytes: reference 0x8A9136AA.
crc32c_zeros_32_test() ->
    Bytes = binary:copy(<<0>>, 32),
    ?assertEqual(16#8A9136AA, erllama_kv_nif:crc32c(Bytes)).

%% 32 0xFF bytes: reference 0x62A8AB43.
crc32c_ones_32_test() ->
    Bytes = binary:copy(<<16#FF>>, 32),
    ?assertEqual(16#62A8AB43, erllama_kv_nif:crc32c(Bytes)).

%% iolist input: must produce the same digest as the flat binary.
crc32c_accepts_iolist_test() ->
    Flat = <<"123456789">>,
    IoList = [<<"12">>, "345", [<<"6">>, [<<"78">>, [<<"9">>]]]],
    ?assertEqual(
        erllama_kv_nif:crc32c(Flat),
        erllama_kv_nif:crc32c(IoList)
    ).

%% Incremental decomposition: CRC32C is a Rabin-style polynomial; the
%% public API runs a single shot, but we sanity-check that splitting
%% the input across multiple calls (handled internally by iolist
%% inspection) produces the same digest.
crc32c_split_input_test() ->
    Whole = binary:copy(<<"abcdefghij">>, 100),
    Split = [binary:part(Whole, 0, 500), binary:part(Whole, 500, 500)],
    ?assertEqual(
        erllama_kv_nif:crc32c(Whole),
        erllama_kv_nif:crc32c(Split)
    ).

crc32c_badarg_atom_test() ->
    ?assertError(badarg, erllama_kv_nif:crc32c(not_iodata)).

crc32c_badarg_integer_test() ->
    ?assertError(badarg, erllama_kv_nif:crc32c(42)).

%% =============================================================================
%% kv_pack / kv_unpack stubs (step 2a)
%% =============================================================================

kv_pack_returns_not_implemented_test() ->
    ?assertEqual(
        {error, not_implemented},
        erllama_kv_nif:kv_pack(make_ref(), [1, 2, 3], 3)
    ).

kv_unpack_returns_not_implemented_test() ->
    ?assertEqual(
        {error, not_implemented},
        erllama_kv_nif:kv_unpack(make_ref(), <<>>, 0)
    ).
