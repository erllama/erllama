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
%% PR4a: nif_step (multi-sequence batched decode)
%% =============================================================================

%% Decoding a freshly-built seq without a prefill behind it must
%% return {error, no_logits} cleanly — the sentinel
%% per_seq[seq_id].last_logits_idx == -1 catches this before reaching
%% llama_sampler_sample, which would GGML_ASSERT and abort the BEAM.
nif_step_decode_without_prefill_rejects_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                {ok, Ctx} =
                    erllama_nif:new_context(Model, #{n_ctx => 256, n_seq_max => 2}),
                {ok, Sampler} = erllama_nif:sampler_new(Ctx, #{}),
                try
                    ?assertEqual(
                        {error, no_logits},
                        erllama_nif:step(Ctx, [{0, {decode, Sampler}}])
                    )
                after
                    ok = erllama_nif:sampler_free(Sampler),
                    ok = erllama_nif:free_context(Ctx),
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% Prefill one seq then decode one token from it. The minimum
%% viable use of nif_step — single-seq, single decode tick.
nif_step_single_seq_prefill_then_decode_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                {ok, Ctx} =
                    erllama_nif:new_context(Model, #{n_ctx => 256, n_seq_max => 2}),
                {ok, Sampler} = erllama_nif:sampler_new(Ctx, #{}),
                try
                    Tokens = erllama_nif:tokenize(
                        Model,
                        <<"Hello world">>,
                        #{add_special => true, parse_special => false}
                    ),
                    {ok, [{0, prefilled}]} =
                        erllama_nif:step(Ctx, [{0, {prefill, Tokens}}]),
                    {ok, [{0, {token, T, _Eog}}]} =
                        erllama_nif:step(Ctx, [{0, {decode, Sampler}}]),
                    ?assert(is_integer(T)),
                    ?assert(T >= 0)
                after
                    ok = erllama_nif:sampler_free(Sampler),
                    ok = erllama_nif:free_context(Ctx),
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% Co-batch: prefill seq 0 in tick 1, then issue decode-on-0 plus
%% prefill-on-1 in tick 2. Result must carry both rows.
nif_step_co_batched_prefill_with_decode_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                {ok, Ctx} =
                    erllama_nif:new_context(Model, #{n_ctx => 256, n_seq_max => 2}),
                {ok, Sampler0} = erllama_nif:sampler_new(Ctx, #{}),
                try
                    Tokens = erllama_nif:tokenize(
                        Model,
                        <<"Hello world">>,
                        #{add_special => true, parse_special => false}
                    ),
                    {ok, [{0, prefilled}]} =
                        erllama_nif:step(Ctx, [{0, {prefill, Tokens}}]),
                    {ok, Results} = erllama_nif:step(Ctx, [
                        {0, {decode, Sampler0}},
                        {1, {prefill, Tokens}}
                    ]),
                    ?assertEqual(2, length(Results)),
                    %% Order: results follow input op order.
                    [{0, {token, T0, _}}, {1, prefilled}] = Results,
                    ?assert(is_integer(T0))
                after
                    ok = erllama_nif:sampler_free(Sampler0),
                    ok = erllama_nif:free_context(Ctx),
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% A bogus seq_id (out of n_seq_max cap) raises badarg.
nif_step_rejects_bad_seq_id_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                {ok, Ctx} =
                    erllama_nif:new_context(Model, #{n_ctx => 64, n_seq_max => 2}),
                try
                    ?assertError(
                        badarg,
                        erllama_nif:step(Ctx, [{99999, {prefill, [1, 2, 3]}}])
                    )
                after
                    ok = erllama_nif:free_context(Ctx),
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% n_seq_max past the compile-time cap is rejected at new_context.
new_context_rejects_n_seq_max_over_cap_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                try
                    ?assertError(
                        badarg,
                        erllama_nif:new_context(Model, #{n_seq_max => 1000000})
                    )
                after
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% =============================================================================
%% PR3: llama.cpp option passthrough (atom enums + tensor_split)
%% =============================================================================

%% A bad split_mode atom must raise `badarg` before the load even
%% runs — caught entirely in the option parser. Uses a fake path so
%% the test does not need LLAMA_TEST_MODEL; the 60 s timeout covers
%% the first-call Metal-backend lazy init on macOS.
load_model_rejects_bad_split_mode_test_() ->
    {timeout, 60, fun() ->
        ?assertError(
            badarg,
            erllama_nif:load_model(
                <<"/no/such/file.gguf">>, #{split_mode => bogus}
            )
        )
    end}.

%% Same for flash_attn / type_k / type_v on new_context — but those
%% need a loaded model to exercise. Gate on LLAMA_TEST_MODEL.
new_context_rejects_bad_flash_attn_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                try
                    ?assertError(
                        badarg,
                        erllama_nif:new_context(Model, #{flash_attn => bogus})
                    )
                after
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

new_context_rejects_bad_type_k_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                try
                    ?assertError(
                        badarg,
                        erllama_nif:new_context(Model, #{type_k => zz})
                    )
                after
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% Bad tensor_split (non-numeric entry) is rejected before the load.
load_model_rejects_bad_tensor_split_test_() ->
    {timeout, 60, fun() ->
        ?assertError(
            badarg,
            erllama_nif:load_model(
                <<"/no/such/file.gguf">>, #{tensor_split => [hello, world]}
            )
        )
    end}.

%% Successful passthrough tests need a real GGUF.
load_model_with_split_mode_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(
                        list_to_binary(Path),
                        #{n_gpu_layers => 0, split_mode => layer}
                    ),
                ok = erllama_nif:free_model(Model)
            end}
    end.

load_model_with_main_gpu_and_tensor_split_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                %% On a CPU-only build these knobs are no-ops; load
                %% must still succeed without errors.
                {ok, Model} =
                    erllama_nif:load_model(
                        list_to_binary(Path),
                        #{
                            n_gpu_layers => 0,
                            main_gpu => 0,
                            tensor_split => [1.0]
                        }
                    ),
                ok = erllama_nif:free_model(Model)
            end}
    end.

new_context_with_flash_attn_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                try
                    {ok, Ctx1} =
                        erllama_nif:new_context(Model, #{
                            n_ctx => 256, flash_attn => auto
                        }),
                    ok = erllama_nif:free_context(Ctx1),
                    {ok, Ctx2} =
                        erllama_nif:new_context(Model, #{
                            n_ctx => 256, flash_attn => false
                        }),
                    ok = erllama_nif:free_context(Ctx2)
                after
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

new_context_with_type_kv_test_() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            {"LLAMA_TEST_MODEL unset; skipping", []};
        Path ->
            {timeout, 60, fun() ->
                {ok, Model} =
                    erllama_nif:load_model(list_to_binary(Path), #{n_gpu_layers => 0}),
                try
                    {ok, Ctx} =
                        erllama_nif:new_context(Model, #{
                            n_ctx => 256, type_k => f16, type_v => f16
                        }),
                    ok = erllama_nif:free_context(Ctx)
                after
                    ok = erllama_nif:free_model(Model)
                end
            end}
    end.

%% =============================================================================
%% VRAM probe
%% =============================================================================

%% nif_vram_info triggers the lazy llama_backend_init (pthread_once)
%% on first call, which on macOS includes Metal device discovery and
%% can take several seconds. Same 60 s timeout pattern as the load
%% test above.
vram_info_returns_ok_or_no_gpu_test_() ->
    {timeout, 60, fun() ->
        case erllama_nif:vram_info() of
            {ok, M} ->
                ?assert(is_map(M)),
                ?assert(maps:is_key(total_b, M)),
                ?assert(maps:is_key(free_b, M)),
                ?assert(maps:is_key(used_b, M)),
                ?assertEqual(
                    maps:get(total_b, M),
                    maps:get(free_b, M) + maps:get(used_b, M)
                );
            {error, no_gpu} ->
                %% CPU-only build: documented contract.
                ok
        end
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
                %% llama.cpp clamps n_ctx up to the model's training
                %% context, so a small requested n_ctx is not honored
                %% on tiny test models. Sending 200_000 tokens exceeds
                %% the training ctx of every current model family
                %% (Llama 3.2 caps at 128k) while staying under the
                %% NIF's ERLLAMA_MAX_TOKENS = 1M cap.
                prefill_overflow(
                    list_to_binary(Path),
                    #{n_ctx => 64, n_batch => 64},
                    200_000,
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
                    200_000,
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
        %% Use token id 0 (always within n_vocab) so the helper does
        %% not accidentally hit the invalid_token path on tiny-vocab
        %% test models before the overflow check fires.
        ?assertEqual(
            {error, ExpectedAtom},
            erllama_nif:prefill(Ctx, lists:duplicate(NTokens, 0))
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
            erllama_nif:embed(Ctx, lists:duplicate(NTokens, 0))
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
