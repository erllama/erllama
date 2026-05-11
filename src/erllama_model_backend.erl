%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model_backend).
-moduledoc """
Behaviour describing the operations the `erllama_model` gen_statem
needs from a backing inference engine.

Two backends ship in v0.2:

  `erllama_model_stub` — deterministic phash2-based stubs; used
      by tests that don't have a GGUF on disk.
  `erllama_model_llama` — real llama.cpp via the NIF.

Future backends (mock for fault injection, remote for distributed
inference, etc.) can plug in via this same surface.
""".

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
%% stub) can omit this; `erllama_model` checks `is_exported/3` and
%% skips the primer when absent.
-callback seq_rm_last(state(), NTokens :: pos_integer()) ->
    ok | {error, term()}.

%% Optional. Render a normalised chat request through the model's
%% chat template and tokenise in one step. The Request map carries
%% `messages`, `system`, and `tools` (any may be undefined or
%% missing). Backends that don't support chat templating can omit
%% this; callers will get `{error, not_supported}` from the public
%% API.
-callback apply_chat_template(state(), Request :: chat_request()) ->
    {ok, [erllama_nif:token_id()]} | {error, term()}.

%% Optional. Compute an embedding vector for the given prompt tokens.
%% Backends that don't support embeddings can omit this.
-callback embed(state(), [erllama_nif:token_id()]) ->
    {ok, [float()]} | {error, term()}.

%% Optional. Configure the per-request sampler with a GBNF grammar.
%% Equivalent to `configure_sampler(state(), #{grammar => Grammar})`.
%% Kept for backwards compatibility; new code should call
%% `configure_sampler/2`.
-callback set_grammar(state(), Grammar :: binary() | undefined) ->
    {ok, state()} | {error, term()}.

%% Optional. Configure the per-request sampler from a config map.
%% Called by `erllama_model` immediately before the first decode_one
%% of an inference; the chain is reset to greedy on
%% `clear_sampler/1`. Backends that ignore sampling can omit it.
%%
%% Recognised keys (all optional): `grammar`, `repetition_penalty`,
%% `top_k`, `top_p`, `min_p`, `temperature`, `seed`. See
%% `erllama_nif:configure_sampler/2` for the precise semantics.
-callback configure_sampler(state(), sampler_opts()) ->
    {ok, state()} | {error, term()}.

-callback clear_sampler(state()) -> {ok, state()} | {error, term()}.

%% Optional LoRA support. `load_adapter/2` returns an opaque handle
%% identifying the adapter; the handle is passed back to
%% `set_adapter_scale/3` and `unload_adapter/2`. `apply_adapters/2`
%% installs the current attachment set on the underlying context;
%% the model layer calls it whenever the attachment set or any scale
%% changes. The Adapters argument is a list of
%% `{Handle, Scale :: float()}` tuples; an empty list detaches
%% everything. Backends without LoRA support can omit the entire
%% group; the model layer returns `{error, not_supported}` to the
%% public API.
-callback load_adapter(state(), Path :: iodata()) ->
    {ok, term(), state()} | {error, term()}.
-callback unload_adapter(state(), Handle :: term()) ->
    {ok, state()} | {error, term()}.
-callback apply_adapters(state(), [{term(), float()}]) ->
    {ok, state()} | {error, term()}.

-optional_callbacks([
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

-type sampler_opts() :: #{
    grammar => binary(),
    repetition_penalty => float(),
    top_k => non_neg_integer(),
    top_p => float(),
    min_p => float(),
    temperature => float(),
    seed => non_neg_integer()
}.

-export_type([sampler_opts/0]).

-type chat_request() :: #{
    messages := [chat_message()],
    system => binary() | undefined,
    tools => [chat_tool()] | undefined
}.

-type chat_message() :: #{
    role := binary(),
    content := binary() | [map()]
}.

-type chat_tool() :: #{
    name := binary(),
    description => binary(),
    schema => map()
}.

-export_type([chat_request/0, chat_message/0, chat_tool/0]).

-export_type([state/0]).
