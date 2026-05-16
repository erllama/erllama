%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Pure-Erlang unit tests for erllama_model_llama. The NIF-backed
%% paths (init/1, step/2, etc.) require a real GGUF and live in
%% erllama_real_model_SUITE; this module covers the small helpers
%% that have no native dependency.
-module(erllama_model_llama_tests).
-include_lib("eunit/include/eunit.hrl").

%% thinking_signature/3 reads the node-wide signing key from the
%% application environment and HMAC-SHA256s the supplied bytes.
%% Without a key, the function falls back to <<>> -- the documented
%% "no signature available" path.

thinking_signature_returns_empty_without_key_test() ->
    application:unset_env(erllama, thinking_signing_key),
    ?assertEqual(
        <<>>,
        erllama_model_llama:thinking_signature(undefined, 0, <<"some thinking text">>)
    ).

thinking_signature_returns_hmac_with_key_test() ->
    Key = <<"unit-test-key">>,
    Bytes = <<"some thinking text">>,
    application:set_env(erllama, thinking_signing_key, Key),
    try
        Sig = erllama_model_llama:thinking_signature(undefined, 0, Bytes),
        ?assert(is_binary(Sig)),
        ?assertEqual(32, byte_size(Sig)),
        ?assertEqual(crypto:mac(hmac, sha256, Key, Bytes), Sig)
    after
        application:unset_env(erllama, thinking_signing_key)
    end.

thinking_signature_is_deterministic_test() ->
    Key = <<"unit-test-key">>,
    application:set_env(erllama, thinking_signing_key, Key),
    try
        Bytes = <<"identical input">>,
        S1 = erllama_model_llama:thinking_signature(undefined, 0, Bytes),
        S2 = erllama_model_llama:thinking_signature(undefined, 0, Bytes),
        ?assertEqual(S1, S2)
    after
        application:unset_env(erllama, thinking_signing_key)
    end.

thinking_signature_empty_key_treated_as_unset_test() ->
    application:set_env(erllama, thinking_signing_key, <<>>),
    try
        ?assertEqual(
            <<>>,
            erllama_model_llama:thinking_signature(undefined, 0, <<"bytes">>)
        )
    after
        application:unset_env(erllama, thinking_signing_key)
    end.
