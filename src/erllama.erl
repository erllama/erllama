%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
%% @doc
%% Public façade for the erllama application.
%%
%% The cache subsystem (`erllama_cache`) is independent. This module
%% is the user-facing surface for loading and running models.
%%
%% Typical usage:
%%
%% ```
%%   ok = application:ensure_all_started(erllama).
%%   {ok, Model} = erllama:load_model(#{
%%       backend => erllama_model_llama,
%%       model_path => "/srv/models/tinyllama-1.1b-q4_k_m.gguf",
%%       %% cache configuration:
%%       tier_srv => default_disk,
%%       tier => disk,
%%       fingerprint => Fp,
%%       fingerprint_mode => safe,
%%       quant_type => q4_k_m,
%%       quant_bits => 4,
%%       ctx_params_hash => CtxHash,
%%       context_size => 4096,
%%       policy => Policy
%%   }).
%%   {ok, Reply, _Tokens} = erllama:complete(Model, <<"hello">>).
%%   ok = erllama:unload(Model).
%% '''
%%
%% Models are dynamic children of `erllama_model_sup` (simple_one_for_one).
%% A registered name is auto-generated when the caller does not provide
%% an explicit `model_id` in the config map.
%% @end
-module(erllama).

-export([
    load_model/1,
    load_model/2,
    unload/1,
    complete/2,
    complete/3,
    models/0,
    counters/0
]).

-export_type([model/0]).

-type model() :: atom() | pid().

%% =============================================================================
%% Public API
%% =============================================================================

%% @doc Load a model with an auto-generated id.
-spec load_model(map()) -> {ok, atom()} | {error, term()}.
load_model(Config) when is_map(Config) ->
    load_model(default_id(), Config).

%% @doc Load a model with an explicit id.
-spec load_model(atom(), map()) -> {ok, atom()} | {error, term()}.
load_model(ModelId, Config) when is_atom(ModelId), is_map(Config) ->
    case erllama_model_sup:start_model(ModelId, Config) of
        {ok, _Pid} -> {ok, ModelId};
        {error, {already_started, _}} -> {error, already_loaded};
        {error, _} = E -> E
    end.

%% @doc Unload a model. Terminates the gen_statem cleanly.
-spec unload(model()) -> ok | {error, term()}.
unload(Model) ->
    erllama_model_sup:stop_model(Model).

%% @doc Run a completion against a loaded model.
-spec complete(model(), binary()) ->
    {ok, binary(), [erllama_nif:token_id()]} | {error, term()}.
complete(Model, Prompt) ->
    erllama_model:complete(Model, Prompt).

-spec complete(model(), binary(), map()) ->
    {ok, binary(), [erllama_nif:token_id()]} | {error, term()}.
complete(Model, Prompt, Opts) ->
    erllama_model:complete(Model, Prompt, Opts).

%% @doc List currently-loaded model pids.
-spec models() -> [pid()].
models() ->
    [Pid || {_, Pid, _, _} <- erllama_model_sup:models(), is_pid(Pid)].

%% @doc Snapshot of the cache subsystem operational counters.
-spec counters() -> #{atom() => non_neg_integer()}.
counters() ->
    erllama_cache:get_counters().

%% =============================================================================
%% Internal
%% =============================================================================

default_id() ->
    list_to_atom("erllama_model_" ++ integer_to_list(erlang:unique_integer([positive]))).
