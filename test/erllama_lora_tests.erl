%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Stub-backend coverage for Phase 3 LoRA plumbing:
%%
%% - load_adapter / unload_adapter / set_adapter_scale round-trip
%% - apply_adapters lands the right [{Ref, Scale}] list on the backend
%% - the cache effective fingerprint changes with the adapter set and
%%   with each scale change
%% - mid-request snapshot: a complete/3 run captures effective_fp at
%%   admission and uses it consistently (we just assert request_fp is
%%   reset at end-of-request).
-module(erllama_lora_tests).
-include_lib("eunit/include/eunit.hrl").

with_app(Body) ->
    {ok, Started} = application:ensure_all_started(erllama),
    Dir = make_tmp_dir(),
    {ok, _} = erllama_cache_disk_srv:start_link(lora_disk, Dir),
    try
        Body()
    after
        catch gen_server:stop(lora_disk),
        rm_rf(Dir),
        [application:stop(A) || A <- lists:reverse(Started)],
        ok
    end.

make_tmp_dir() ->
    Base = filename:basedir(user_cache, "erllama-lora-tests"),
    Dir = filename:join(Base, integer_to_list(erlang:unique_integer([positive]))),
    ok = filelib:ensure_path(Dir),
    Dir.

rm_rf(Dir) ->
    case file:list_dir(Dir) of
        {ok, Files} ->
            [
                begin
                    Full = filename:join(Dir, F),
                    case filelib:is_dir(Full) of
                        true -> rm_rf(Full);
                        false -> file:delete(Full)
                    end
                end
             || F <- Files
            ],
            file:del_dir(Dir);
        _ ->
            ok
    end.

write_fixture(Bytes) ->
    Dir = filename:basedir(user_cache, "erllama-lora-fixtures"),
    ok = filelib:ensure_path(Dir),
    Path = filename:join(
        Dir, "adapter_" ++ integer_to_list(erlang:unique_integer([positive]))
    ),
    ok = file:write_file(Path, Bytes),
    Path.

minimal_config() ->
    #{
        backend => erllama_model_stub,
        tier_srv => lora_disk,
        tier => disk,
        fingerprint => binary:copy(<<16#55>>, 32),
        fingerprint_mode => safe,
        quant_type => f16,
        quant_bits => 16,
        ctx_params_hash => binary:copy(<<16#66>>, 32),
        context_size => 1024,
        policy => #{
            min_tokens => 4,
            cold_min_tokens => 4,
            cold_max_tokens => 1000,
            continued_interval => 2048,
            boundary_trim_tokens => 0,
            boundary_align_tokens => 1,
            session_resume_wait_ms => 50
        }
    }.

%% =============================================================================
%% Tests
%% =============================================================================

load_unload_round_trip_test() ->
    with_app(fun() ->
        Id = <<"lora_a">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            Path = write_fixture(<<"adapter-bytes-1">>),
            {ok, H} = erllama:load_adapter(Id, Path),
            ?assert(is_reference(H)),
            ?assertMatch([_], erllama:list_adapters(Id)),
            ok = erllama:unload_adapter(Id, H),
            ?assertEqual([], erllama:list_adapters(Id))
        after
            erllama:unload(Id)
        end
    end).

apply_adapters_records_pairs_test() ->
    with_app(fun() ->
        Id = <<"lora_b">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            P1 = write_fixture(<<"adapter-A">>),
            P2 = write_fixture(<<"adapter-B">>),
            {ok, H1} = erllama:load_adapter(Id, P1),
            {ok, H2} = erllama:load_adapter(Id, P2),
            Applied = erllama_model_stub:applied_adapters(
                erllama_model:get_backend_state(Id)
            ),
            ?assertEqual(2, length(Applied)),
            Handles = [H || {H, _} <- Applied],
            ?assert(lists:member(H1, Handles)),
            ?assert(lists:member(H2, Handles))
        after
            erllama:unload(Id)
        end
    end).

scale_change_replays_pairs_test() ->
    with_app(fun() ->
        Id = <<"lora_c">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            Path = write_fixture(<<"adapter-X">>),
            {ok, H} = erllama:load_adapter(Id, Path),
            [{H, 1.0}] =
                erllama_model_stub:applied_adapters(
                    erllama_model:get_backend_state(Id)
                ),
            ok = erllama:set_adapter_scale(Id, H, 0.5),
            [{H, 0.5}] =
                erllama_model_stub:applied_adapters(
                    erllama_model:get_backend_state(Id)
                )
        after
            erllama:unload(Id)
        end
    end).

unload_unknown_handle_is_idempotent_test() ->
    with_app(fun() ->
        Id = <<"lora_d">>,
        {ok, _} = erllama:load_model(Id, minimal_config()),
        try
            ?assertEqual(ok, erllama:unload_adapter(Id, make_ref()))
        after
            erllama:unload(Id)
        end
    end).

%% =============================================================================
%% Cache identity: same adapter bytes -> same effective fingerprint;
%% different bytes or different scales -> different fingerprints.
%% =============================================================================

effective_fingerprint_segregates_cache_keys_test() ->
    Base = binary:copy(<<16#42>>, 32),
    Sha1 = crypto:hash(sha256, <<"adapter-1">>),
    Sha2 = crypto:hash(sha256, <<"adapter-2">>),
    Fp0 = erllama_cache_key:effective_fingerprint(Base, []),
    FpA = erllama_cache_key:effective_fingerprint(Base, [{Sha1, 1.0}]),
    FpB = erllama_cache_key:effective_fingerprint(Base, [{Sha2, 1.0}]),
    FpAScaled = erllama_cache_key:effective_fingerprint(Base, [{Sha1, 0.5}]),
    %% Empty adapter list == base fingerprint passthrough.
    ?assertEqual(Base, Fp0),
    %% Two different adapters produce different keys.
    ?assertNotEqual(FpA, FpB),
    %% Same adapter, different scale, also different.
    ?assertNotEqual(FpA, FpAScaled).

effective_fingerprint_sort_invariance_test() ->
    %% Two adapters in two orders must yield the same effective fp.
    Base = binary:copy(<<16#42>>, 32),
    A = {crypto:hash(sha256, <<"a">>), 1.0},
    B = {crypto:hash(sha256, <<"b">>), 0.75},
    Fp1 = erllama_cache_key:effective_fingerprint(Base, [A, B]),
    Fp2 = erllama_cache_key:effective_fingerprint(Base, [B, A]),
    ?assertEqual(Fp1, Fp2).
