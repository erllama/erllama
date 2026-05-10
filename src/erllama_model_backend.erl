%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Behaviour describing the operations the `erllama_model` gen_statem
%% needs from a backing inference engine.
%%
%% Two backends ship in v0.2:
%%
%%   `erllama_model_stub` — deterministic phash2-based stubs; used
%%       by tests that don't have a GGUF on disk.
%%   `erllama_model_llama` — real llama.cpp via the NIF.
%%
%% Future backends (mock for fault injection, remote for distributed
%% inference, etc.) can plug in via this same surface.
%% @end
-module(erllama_model_backend).

-type state() :: term().

-callback init(Config :: map()) -> {ok, state()} | {error, term()}.

-callback terminate(state()) -> ok.

-callback tokenize(state(), Text :: binary()) ->
    [erllama_nif:token_id()] | {error, term()}.

-callback detokenize(state(), [erllama_nif:token_id()]) ->
    binary() | {error, term()}.

-callback prefill(state(), [erllama_nif:token_id()]) -> ok | {error, term()}.

-callback decode_one(state(), ContextTokens :: [erllama_nif:token_id()]) ->
    {ok, erllama_nif:token_id()}
    | {eog, erllama_nif:token_id()}
    | {error, term()}.

-callback kv_pack(state(), Tokens :: [erllama_nif:token_id()]) ->
    binary() | {error, term()}.

-callback kv_unpack(state(), Bin :: binary()) -> ok | {error, term()}.

%% Optional. Drop the last KV cell of the active sequence so the
%% caller can re-prefill the corresponding token to regenerate logits
%% after a kv_unpack. Backends that don't carry a real KV cache (the
%% stub) can omit this — `erllama_model` checks `is_exported/3` and
%% skips the primer when absent.
-callback seq_rm_last(state(), NTokens :: pos_integer()) ->
    ok | {error, term()}.

-optional_callbacks([seq_rm_last/2]).

-export_type([state/0]).
