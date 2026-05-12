%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Unit tests for top-level erllama:* helpers that do not need a
%% loaded model. Functions that exercise the full inference path
%% live in erllama_streaming_tests, erllama_lora_tests, etc.
-module(erllama_tests).
-include_lib("eunit/include/eunit.hrl").

%% =============================================================================
%% list_cached_prefixes/2
%% =============================================================================

list_cached_prefixes_empty_prompt_test() ->
    %% Short-circuits before touching the registry, so no app start
    %% required.
    ?assertEqual({ok, 0}, erllama:list_cached_prefixes(<<"any">>, [])).

list_cached_prefixes_unloaded_model_test() ->
    %% Same: the registry lookup is the only side-effect, and it
    %% returns undefined for a model that was never loaded.
    {ok, _} = application:ensure_all_started(erllama),
    try
        ?assertEqual(
            {error, model_not_loaded},
            erllama:list_cached_prefixes(<<"never-loaded-model">>, [1, 2, 3])
        )
    after
        ok = application:stop(erllama)
    end.

%% =============================================================================
%% draft_tokens/3
%% =============================================================================

draft_tokens_empty_prefix_test() ->
    %% Short-circuits before infer/4, so no app required.
    ?assertEqual(
        {error, empty_prefix},
        erllama:draft_tokens(<<"any">>, [], #{max => 4})
    ).

%% =============================================================================
%% verify/4 unit tests (the snapshot/restore proof lives in the
%% erllama_real_model_SUITE since it needs a loaded GGUF + the
%% existing test infrastructure for cache + fingerprints).
%% =============================================================================

verify_rejects_non_binary_model_id_test() ->
    ?assertError(function_clause, erllama:verify(not_a_binary, [1], [2], 1)).
