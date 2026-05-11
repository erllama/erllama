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
    apply_adapters/2
]).

-record(s, {
    model :: erllama_nif:model_ref(),
    ctx :: erllama_nif:context_ref()
}).

init(Config) ->
    Path = maps:get(model_path, Config),
    MOpts = maps:get(model_opts, Config, #{}),
    COpts = maps:get(context_opts, Config, #{}),
    case erllama_nif:load_model(Path, MOpts) of
        {ok, Model} ->
            case erllama_nif:new_context(Model, COpts) of
                {ok, Ctx} ->
                    {ok, #s{model = Model, ctx = Ctx}};
                {error, _} = E ->
                    erllama_nif:free_model(Model),
                    E
            end;
        {error, _} = E ->
            E
    end.

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
