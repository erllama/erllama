%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_cache_key_tests).
-include_lib("eunit/include/eunit.hrl").

fp() -> binary:copy(<<16#AA>>, 32).
ctx() -> binary:copy(<<16#BB>>, 32).

make_returns_32_bytes_test() ->
    Key = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        tokens => [1, 2, 3]
    }),
    ?assertEqual(32, byte_size(Key)).

make_is_deterministic_test() ->
    Components = #{
        fingerprint => fp(),
        quant_type => q4_k_m,
        ctx_params_hash => ctx(),
        tokens => [1, 2, 3, 4]
    },
    ?assertEqual(
        erllama_cache_key:make(Components),
        erllama_cache_key:make(Components)
    ).

make_distinct_for_token_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        tokens => [1, 2, 3]
    }),
    B = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        tokens => [1, 2, 4]
    }),
    ?assertNotEqual(A, B).

make_distinct_for_quant_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => ctx(),
        tokens => [1]
    }),
    B = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => q4_k_m,
        ctx_params_hash => ctx(),
        tokens => [1]
    }),
    ?assertNotEqual(A, B).

make_distinct_for_fingerprint_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => binary:copy(<<16#AA>>, 32),
        quant_type => f16,
        ctx_params_hash => ctx(),
        tokens => [1]
    }),
    B = erllama_cache_key:make(#{
        fingerprint => binary:copy(<<16#AB>>, 32),
        quant_type => f16,
        ctx_params_hash => ctx(),
        tokens => [1]
    }),
    ?assertNotEqual(A, B).

make_distinct_for_ctx_change_test() ->
    A = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BB>>, 32),
        tokens => [1]
    }),
    B = erllama_cache_key:make(#{
        fingerprint => fp(),
        quant_type => f16,
        ctx_params_hash => binary:copy(<<16#BC>>, 32),
        tokens => [1]
    }),
    ?assertNotEqual(A, B).

make_badarg_short_fingerprint_test() ->
    ?assertError(
        function_clause,
        erllama_cache_key:make(#{
            fingerprint => <<1, 2, 3>>,
            quant_type => f16,
            ctx_params_hash => ctx(),
            tokens => []
        })
    ).

quant_byte_round_trip_test() ->
    Quants = [
        f32,
        f16,
        q4_0,
        q4_1,
        q5_0,
        q5_1,
        q8_0,
        q4_k_m,
        q4_k_s,
        q5_k_m,
        q5_k_s,
        q6_k,
        q8_k
    ],
    [
        ?assertEqual({ok, Q}, erllama_cache_key:quant_atom(erllama_cache_key:quant_byte(Q)))
     || Q <- Quants
    ].

quant_atom_unknown_test() ->
    ?assertEqual({error, unknown_quant}, erllama_cache_key:quant_atom(255)).

encode_decode_tokens_round_trip_test() ->
    Tokens = [0, 1, 16#FFFFFFFF, 12345, 67890],
    Bin = erllama_cache_key:encode_tokens(Tokens),
    ?assertEqual(20, byte_size(Bin)),
    ?assertEqual(Tokens, erllama_cache_key:decode_tokens(Bin)).

encode_empty_tokens_test() ->
    ?assertEqual(<<>>, erllama_cache_key:encode_tokens([])),
    ?assertEqual([], erllama_cache_key:decode_tokens(<<>>)).
