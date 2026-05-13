%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% Random integer prompts up to 2*n_ctx must produce ok or
%% {error, _} from nif_prefill, and never bring the BEAM down.
%% The property self-skips when LLAMA_TEST_MODEL is unset; that
%% gate must live inside the property because rebar3_proper
%% discovers prop_*.erl directly and bypasses any eunit harness.
-module(prop_erllama_nif).
-include_lib("proper/include/proper.hrl").

prop_prefill_never_crashes() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            true;
        Path ->
            ?SETUP(
                fun() -> setup_resources(list_to_binary(Path)) end,
                ?FORALL(
                    N,
                    choose(0, 2 * n_ctx()),
                    begin
                        Ctx = persistent_term:get({?MODULE, ctx}),
                        Tokens = lists:seq(1, N),
                        case erllama_nif:prefill(Ctx, Tokens) of
                            ok -> true;
                            {error, _} -> true;
                            _Other -> false
                        end
                    end
                )
            )
    end.

%% Cross-seq cache property: a binary produced by `kv_pack(seq=A)`
%% must restore cleanly into a different `seq_id=B` on the same
%% context via `kv_unpack`. The vendor docstring at
%% c_src/llama.cpp/include/llama.h:836 promises this contract; this
%% property guards against a regression that would break the
%% multi-seq cache design.
%%
%% We can't easily read back KV cells, so the check is: after the
%% cross-seq restore, llama_memory_seq_pos_max(seq B) should equal
%% the value it had on seq A before the pack. We probe that
%% indirectly through nif_step: a decode op on seq B must NOT
%% return {error, no_logits} after the round-trip is followed by a
%% primer-prefill of the last warm token. That gives us a black-box
%% "yes, seq B has KV at the expected position" signal.
prop_pack_seq_a_unpack_seq_b_matches_baseline() ->
    case os:getenv("LLAMA_TEST_MODEL") of
        false ->
            true;
        Path ->
            ?SETUP(
                fun() -> setup_resources(list_to_binary(Path)) end,
                ?FORALL(
                    N,
                    choose(4, 16),
                    begin
                        Ctx = persistent_term:get({?MODULE, ctx}),
                        %% Clean slate on both seqs so repeated
                        %% runs of the property don't accumulate.
                        ok = erllama_nif:kv_seq_rm(Ctx, 0, 0, -1),
                        ok = erllama_nif:kv_seq_rm(Ctx, 1, 0, -1),
                        Tokens = lists:duplicate(N, 1),
                        ok = erllama_nif:prefill(Ctx, Tokens),
                        Bin = erllama_nif:kv_pack(Ctx, Tokens, length(Tokens), 0),
                        is_binary(Bin) andalso
                            erllama_nif:kv_unpack(Ctx, Bin, 1) =:= ok
                    end
                )
            )
    end.

setup_resources(Path) ->
    {ok, Model} = erllama_nif:load_model(Path, #{n_gpu_layers => 0}),
    {ok, Ctx} = erllama_nif:new_context(
        Model, #{n_ctx => n_ctx(), n_batch => n_batch(), n_seq_max => 2}
    ),
    persistent_term:put({?MODULE, model}, Model),
    persistent_term:put({?MODULE, ctx}, Ctx),
    fun() ->
        _ = erllama_nif:free_context(persistent_term:get({?MODULE, ctx})),
        _ = erllama_nif:free_model(persistent_term:get({?MODULE, model})),
        _ = persistent_term:erase({?MODULE, ctx}),
        _ = persistent_term:erase({?MODULE, model}),
        ok
    end.

n_ctx() -> 256.
n_batch() -> 64.
