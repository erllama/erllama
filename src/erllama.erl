%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama).
-moduledoc """
Public façade for the erllama application.

The cache subsystem (`erllama_cache`) is independent. This module
is the user-facing surface for loading and running models.

Typical usage:

```
  ok = application:ensure_all_started(erllama).
  {ok, Bin} = file:read_file("/srv/models/tinyllama-1.1b-q4_k_m.gguf").
  {ok, Model} = erllama:load_model(#{
      backend => erllama_model_llama,
      model_path => "/srv/models/tinyllama-1.1b-q4_k_m.gguf",
      fingerprint => crypto:hash(sha256, Bin)
  }).
  {ok, Reply, _Tokens} = erllama:complete(Model, <<"hello">>).
  ok = erllama:unload(Model).
```

Extra cache parameters (`tier`, `tier_srv`, `quant_type`,
`ctx_params_hash`, `policy`, ...) are optional; the defaults route
saves to the RAM tier (`erllama_cache_ram`). See the loading guide
for the full option map and instructions to wire up
`ram_file` / `disk` tier servers.

Models are dynamic children of `erllama_model_sup` (simple_one_for_one).
A registered name is auto-generated when the caller does not provide
an explicit `model_id` in the config map.
""".

-export([
    load_model/1,
    load_model/2,
    unload/1,
    unload_model/1,
    complete/2,
    complete/3,
    infer/4,
    cancel/1,
    models/0,
    list_models/0,
    model_info/1,
    tokenize/2,
    detokenize/2,
    apply_chat_template/2,
    embed/2,
    counters/0
]).

-export_type([model/0, model_id/0, model_info/0]).

-type model_id() :: erllama_registry:model_id().
-type model() :: erllama_model:model().
-type model_info() :: erllama_model:model_info().

%% =============================================================================
%% Public API
%% =============================================================================

-doc "Load a model with an auto-generated id.".
-spec load_model(map()) -> {ok, model_id()} | {error, term()}.
load_model(Config) when is_map(Config) ->
    load_model(default_id(), Config).

-doc "Load a model with an explicit id.".
-spec load_model(model_id(), map()) -> {ok, model_id()} | {error, term()}.
load_model(ModelId, Config) when is_binary(ModelId), is_map(Config) ->
    case erllama_model_sup:start_model(ModelId, Config) of
        {ok, _Pid} -> {ok, ModelId};
        {error, {already_started, _}} -> {error, already_loaded};
        {error, _} = E -> E
    end.

-doc "Unload a model. Terminates the gen_statem cleanly.".
-spec unload(model()) -> ok | {error, term()}.
unload(Model) ->
    erllama_model_sup:stop_model(Model).

-doc """
Alias for `unload/1`. Provided for API symmetry with `load_model/1,2`
and the OpenAI/Ollama-style naming used by downstream HTTP servers.
""".
-spec unload_model(model()) -> ok | {error, term()}.
unload_model(Model) ->
    unload(Model).

-doc "Run a completion against a loaded model.".
-spec complete(model(), binary()) ->
    {ok, binary(), [erllama_nif:token_id()]} | {error, term()}.
complete(Model, Prompt) ->
    erllama_model:complete(Model, Prompt).

-spec complete(model(), binary(), map()) ->
    {ok, binary(), [erllama_nif:token_id()]} | {error, term()}.
complete(Model, Prompt, Opts) ->
    erllama_model:complete(Model, Prompt, Opts).

-doc """
Streaming inference. Returns immediately with a `reference()` that
identifies this request; tokens are delivered to `CallerPid` via
async messages:

- `{erllama_token, Ref, Bin :: binary()}` — text fragment
- `{erllama_done, Ref, Stats}` — normal completion
- `{erllama_error, Ref, Reason}` — failure

`Tokens` is the prompt as a list of token ids; tokenisation is the
caller's responsibility (use `tokenize/2` or apply a chat template
first).
""".
-spec infer(
    model(),
    [erllama_nif:token_id()],
    erllama_model:infer_params(),
    pid()
) ->
    {ok, reference()} | {error, term()}.
infer(Model, Tokens, Params, CallerPid) ->
    erllama_model:infer(Model, Tokens, Params, CallerPid).

-doc """
Cancel an in-flight streaming inference. Idempotent and
fire-and-forget; cancellation is observed at the next inter-token
boundary. The caller still receives a final `{erllama_done, Ref,
Stats}` with `cancelled => true`.
""".
-spec cancel(reference()) -> ok.
cancel(Ref) ->
    erllama_model:cancel(Ref).

-doc """
List currently-loaded model pids (low-level supervisor view). Most
callers want `list_models/0`, which returns metadata maps.
""".
-spec models() -> [pid()].
models() ->
    [Pid || {_, Pid, _, _} <- erllama_model_sup:models(), is_pid(Pid)].

-doc """
List currently-loaded models as `model_info()` maps. Each entry
includes the model id, status, backend, context size, and
quantisation.
""".
-spec list_models() -> [model_info()].
list_models() ->
    lists:filtermap(
        fun({_ModelId, Pid}) ->
            try
                {true, erllama_model:model_info(Pid)}
            catch
                _:_ -> false
            end
        end,
        erllama_registry:all()
    ).

-doc """
Inspect a single loaded model. Returns the same map shape
`list_models/0` produces. Crashes with `noproc` if the model is not
loaded.
""".
-spec model_info(model()) -> model_info().
model_info(Model) ->
    erllama_model:model_info(Model).

-doc """
Tokenise text against a loaded model's tokenizer. Safe to call
concurrently with `complete/2,3`.
""".
-spec tokenize(model(), binary()) ->
    {ok, [erllama_nif:token_id()]} | {error, term()}.
tokenize(Model, Text) ->
    erllama_model:tokenize(Model, Text).

-doc "Detokenise a list of token ids back to text.".
-spec detokenize(model(), [erllama_nif:token_id()]) ->
    {ok, binary()} | {error, term()}.
detokenize(Model, Tokens) ->
    erllama_model:detokenize(Model, Tokens).

-doc """
Render a chat request through the model's chat template and
tokenise. The Request map carries `messages`, `system`, and `tools`.
""".
-spec apply_chat_template(model(), erllama_model_backend:chat_request()) ->
    {ok, [erllama_nif:token_id()]} | {error, term()}.
apply_chat_template(Model, Request) ->
    erllama_model:apply_chat_template(Model, Request).

-doc "Compute an embedding vector for the given prompt tokens.".
-spec embed(model(), [erllama_nif:token_id()]) ->
    {ok, [float()]} | {error, term()}.
embed(Model, Tokens) ->
    erllama_model:embed(Model, Tokens).

-doc "Snapshot of the cache subsystem operational counters.".
-spec counters() -> #{atom() => non_neg_integer()}.
counters() ->
    erllama_cache:get_counters().

%% =============================================================================
%% Internal
%% =============================================================================

default_id() ->
    Int = erlang:unique_integer([positive]),
    iolist_to_binary(["erllama_model_", integer_to_binary(Int)]).
