%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model_llama).
-moduledoc """
Real-llama.cpp backend for `erllama_model`.

Owns a `model_ref` and a `context_ref` from `erllama_nif`. The
gen_statem hands its decode/kv operations through this module;
this module forwards to the NIF.

Config (passed through `erllama_model:start_link/2`):
  model_path :: file:name() | binary()  (required)
  model_opts :: map()  (forwarded to erllama_nif:load_model/2)
  context_opts :: map()  (forwarded to erllama_nif:new_context/2)
""".
-behaviour(erllama_model_backend).

-export([
    init/1,
    terminate/1,
    tokenize/2,
    detokenize/2,
    prefill/2,
    decode_one/2,
    kv_pack/2,
    kv_unpack/2,
    seq_rm_last/2,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    load_adapter/2,
    unload_adapter/2,
    apply_adapters/2,
    extra_metadata/1,
    verify/4
]).

-record(s, {
    model :: erllama_nif:model_ref(),
    ctx :: erllama_nif:context_ref(),
    %% Captured once at init for vram_estimate_b derivation. The
    %% values are immutable once the model is loaded; the gen_statem
    %% reads them via extra_metadata/1 at its own init time and
    %% caches the derived estimate in #data.
    model_size_bytes = 0 :: non_neg_integer(),
    total_layers = 0 :: non_neg_integer(),
    %% Signed: llama.cpp uses negative (typically -1) to mean
    %% "offload all layers".
    n_gpu_layers = 0 :: integer()
}).

init(Config) ->
    Path = maps:get(model_path, Config),
    MOpts = maps:get(model_opts, Config, #{}),
    COpts = maps:get(context_opts, Config, #{}),
    NGpuLayers = maps:get(n_gpu_layers, MOpts, 0),
    case erllama_nif:load_model(Path, MOpts) of
        {ok, Model} ->
            case erllama_nif:new_context(Model, COpts) of
                {ok, Ctx} ->
                    {ok, #s{
                        model = Model,
                        ctx = Ctx,
                        model_size_bytes = safe_uint(erllama_nif:model_size(Model)),
                        total_layers = safe_uint(erllama_nif:model_n_layer(Model)),
                        n_gpu_layers = NGpuLayers
                    }};
                {error, _} = E ->
                    erllama_nif:free_model(Model),
                    E
            end;
        {error, _} = E ->
            E
    end.

safe_uint(N) when is_integer(N), N >= 0 -> N;
safe_uint(_) -> 0.

terminate(#s{ctx = Ctx, model = Model}) ->
    erllama_nif:free_context(Ctx),
    erllama_nif:free_model(Model),
    ok.

tokenize(#s{model = M}, Text) ->
    erllama_nif:tokenize(M, Text, #{add_special => true, parse_special => false}).

detokenize(#s{model = M}, Tokens) ->
    erllama_nif:detokenize(M, Tokens).

prefill(#s{ctx = C}, Tokens) ->
    erllama_nif:prefill(C, Tokens).

decode_one(#s{ctx = C}, _ContextTokens) ->
    erllama_nif:decode_one(C).

kv_pack(#s{ctx = C}, Tokens) ->
    erllama_nif:kv_pack(C, Tokens, length(Tokens)).

kv_unpack(#s{ctx = C}, Bin) ->
    erllama_nif:kv_unpack(C, Bin, 0).

%% Drop the cell at position N-1 from seq 0 so the model layer can
%% re-prefill the corresponding token and regenerate logits.
%% `llama_state_seq_*` only persists KV cells, never the per-context
%% logits buffer; without this primer the next sample would read stale
%% (or zero) logits.
seq_rm_last(#s{ctx = C}, NTokens) when NTokens > 0 ->
    erllama_nif:kv_seq_rm(C, 0, NTokens - 1, -1).

apply_chat_template(#s{model = M}, Request) ->
    erllama_nif:apply_chat_template(M, Request).

embed(#s{ctx = C}, Tokens) ->
    erllama_nif:embed(C, Tokens).

set_grammar(#s{ctx = C} = S, Grammar) when is_binary(Grammar) ->
    case erllama_nif:set_grammar(C, Grammar) of
        ok -> {ok, S};
        {error, _} = E -> E
    end;
set_grammar(#s{} = S, undefined) ->
    {ok, S}.

configure_sampler(#s{} = S, Cfg) when map_size(Cfg) =:= 0 ->
    %% No sampler params at all - leave the existing chain alone so
    %% the lazy greedy fallback in the NIF kicks in on first decode.
    {ok, S};
configure_sampler(#s{ctx = C} = S, Cfg) when is_map(Cfg) ->
    case erllama_nif:configure_sampler(C, Cfg) of
        ok -> {ok, S};
        {error, _} = E -> E
    end.

clear_sampler(#s{ctx = C} = S) ->
    ok = erllama_nif:clear_sampler(C),
    {ok, S}.

load_adapter(#s{model = M} = S, Path) ->
    case erllama_nif:adapter_load(M, Path) of
        {ok, AdapterRef} -> {ok, AdapterRef, S};
        {error, _} = E -> E
    end.

unload_adapter(#s{} = S, AdapterRef) ->
    case erllama_nif:adapter_free(AdapterRef) of
        ok -> {ok, S};
        {error, released} -> {ok, S};
        {error, _} = E -> E
    end.

apply_adapters(#s{ctx = C} = S, Adapters) ->
    case erllama_nif:set_adapters(C, Adapters) of
        ok -> {ok, S};
        {error, _} = E -> E
    end.

extra_metadata(#s{
    model_size_bytes = SB, total_layers = TL, n_gpu_layers = NL
}) ->
    #{model_size_bytes => SB, total_layers => TL, n_gpu_layers => NL}.

%% Speculative-decoding verifier. Snapshot+restore protocol so
%% the caller's KV view is unchanged after the call. Empty prefix
%% is rejected: the acceptance / NextToken indexing both require
%% at least one prefix token.
verify(_S, [], _Candidates, _K) ->
    {error, empty_prefix};
verify(#s{ctx = Ctx} = S, PrefixTokens, Candidates, K) when
    is_list(PrefixTokens), is_list(Candidates), is_integer(K), K > 0
->
    KCap = min(K, length(Candidates)),
    Truncated = lists:sublist(Candidates, KCap),
    Input = PrefixTokens ++ Truncated,
    case erllama_nif:forward_with_argmax(Ctx, Input) of
        {ok, Argmax} ->
            P = length(PrefixTokens),
            Accepted = count_accepted(0, P, Truncated, Argmax),
            %% NextToken comes from logits at 0-indexed position
            %% P - 1 + Accepted (the verifier's prediction given
            %% the prefix and the accepted prefix of candidates).
            %% lists:nth/2 is 1-indexed: P + Accepted.
            NextToken = lists:nth(P + Accepted, Argmax),
            ok = restore_after_verify(Ctx, P, lists:last(PrefixTokens)),
            {ok, Accepted, NextToken, S};
        {error, _} = E ->
            E
    end.

count_accepted(Acc, _P, [], _Argmax) ->
    Acc;
count_accepted(Acc, P, [C | Rest], Argmax) ->
    Predicted = lists:nth(P + Acc, Argmax),
    case Predicted of
        C when is_integer(C) -> count_accepted(Acc + 1, P, Rest, Argmax);
        _ -> Acc
    end.

%% Bring the context's KV cells back to the caller's pre-call
%% length P, then re-prefill the last prefix token so the logits
%% buffer is in a sampleable state for any follow-up decode_one.
%% The caller's pre-call decode_ready flag is not preserved: after
%% verify the context is always ready to sample at the prefix end.
%% Documented in the public erllama:verify/4 doc.
restore_after_verify(Ctx, P, LastPrefixToken) ->
    %% kv_seq_rm with p1 = -1 means "to infinity"; combined with
    %% p0 = P, that drops every cell from the candidate batch.
    _ = erllama_nif:kv_seq_rm(Ctx, 0, P, -1),
    _ = erllama_nif:prefill(Ctx, [LastPrefixToken]),
    ok.
