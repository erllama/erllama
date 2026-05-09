%% @doc
%% Real-llama.cpp backend for `erllama_model`.
%%
%% Owns a `model_ref` and a `context_ref` from `erllama_nif`. The
%% gen_statem hands its decode/kv operations through this module;
%% this module forwards to the NIF.
%%
%% Config (passed through `erllama_model:start_link/2`):
%%   model_path :: file:name() | binary()  (required)
%%   model_opts :: map()  (forwarded to erllama_nif:load_model/2)
%%   context_opts :: map()  (forwarded to erllama_nif:new_context/2)
%% @end
-module(erllama_model_llama).
-behaviour(erllama_model_backend).

-export([
    init/1,
    terminate/1,
    tokenize/2,
    detokenize/2,
    prefill/2,
    decode_one/2,
    kv_pack/2,
    kv_unpack/2
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
