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

%% Optional seq-aware variants. Used by the multi-sequence scheduler
%% to thread per-request seq_ids through the same pack/unpack/cell-
%% removal flow the single-seq paths already use. Backends that have
%% not been ported keep the single-seq callbacks above and `seq_id =
%% 0` is implied everywhere.
-callback kv_pack(state(), Tokens :: [erllama_nif:token_id()], seq_id()) ->
    binary() | {error, term()}.

-callback kv_unpack(state(), Bin :: binary(), seq_id()) -> ok | {error, term()}.

%% Optional. Free all KV cells of a given seq_id and drop any
%% per-seq tracking inside the backend. Called by the scheduler on
%% request finish (success, error, or cancel) before the seq_id is
%% returned to the idle pool.
-callback seq_rm(state(), seq_id()) -> ok | {error, term()}.

%% Optional. Drop the last KV cell of the active sequence so the
%% caller can re-prefill the corresponding token to regenerate logits
%% after a kv_unpack. Backends that don't carry a real KV cache (the
%% stub) can omit this; `erllama_model` checks `is_exported/3` and
%% skips the primer when absent.
%% Clear seq 0 of the context KV cache, resetting `n_past` to 0 so the
%% next prefill begins at position 0. Without this, a cold prefill on a
%% context that already served a previous request would auto-position
%% the new tokens after the residual KV cells from the prior call,
%% producing different output for the same prompt + seed.
-callback seq_clear(state()) -> ok | {error, term()}.

-callback seq_rm_last(state(), NTokens :: pos_integer()) ->
    ok | {error, term()}.

%% Optional. Drop the last KV cell of a specific seq_id. Seq-aware
%% counterpart to seq_rm_last/2.
-callback seq_rm_last(state(), seq_id(), NTokens :: pos_integer()) ->
    ok | {error, term()}.

%% Optional. Drive one batched-decode tick across a set of in-flight
%% sequences. Each op is `{prefill, [Token]}` to push the slice into
%% KV (no sampling) or `{decode, sampler_ref()}` to sample one token
%% from the prior tick's logits and decode it. Returns one result per
%% op in input order.
-callback step(state(), [{seq_id(), step_op()}]) ->
    {ok, [{seq_id(), step_result()}]} | {error, term()}.

%% Optional. Build a per-request sampler chain from the same config
%% map `configure_sampler/2` accepts. Returns an opaque handle the
%% scheduler hands to `step/2` as the decode-row sampler. The chain
%% lives until `sampler_free/1` (or the backend's terminate/1).
-callback sampler_new(state(), sampler_opts()) ->
    {ok, sampler_ref()} | {error, term()}.

%% Optional. Release a sampler chain previously built via
%% `sampler_new/2`. Idempotent: a double-free returns `ok` or a
%% backend-defined `{error, released}`.
-callback sampler_free(sampler_ref()) -> ok | {error, term()}.

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

%% Backend-specific metadata used by erllama_model:list_models/0
%% beyond what the gen_statem already tracks. The default backend
%% (erllama_model_llama) returns model byte size, total layer
%% count, and the n_gpu_layers value the user passed at load
%% time. erllama_model uses these to compute `vram_estimate_b`.
-callback extra_metadata(state()) ->
    #{
        model_size_bytes => non_neg_integer(),
        total_layers => non_neg_integer(),
        n_gpu_layers => integer()
    }.

%% Speculative-decoding verifier. Runs PrefixTokens ++ Candidates
%% (truncated to K) through the model with per-position argmax,
%% returns the longest accepted prefix and the model's own next
%% token after it. Mutates and restores the context's KV cells +
%% logits buffer + decode_ready flag so the caller's pre-call view
%% is preserved.
-callback verify(
    state(),
    PrefixTokens :: [erllama_nif:token_id()],
    Candidates :: [erllama_nif:token_id()],
    K :: pos_integer()
) ->
    {ok, AcceptedCount :: non_neg_integer(), NextToken :: erllama_nif:token_id() | eos,
        NewState :: state()}
    | {error, term()}.

-optional_callbacks([
    kv_pack/3,
    kv_unpack/3,
    seq_rm/2,
    seq_clear/1,
    seq_rm_last/2,
    seq_rm_last/3,
    step/2,
    sampler_new/2,
    sampler_free/1,
    apply_chat_template/2,
    embed/2,
    set_grammar/2,
    configure_sampler/2,
    clear_sampler/1,
    load_adapter/2,
    unload_adapter/2,
    apply_adapters/2,
    extra_metadata/1,
    verify/4,
    thinking_signature/2
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

-type seq_id() :: non_neg_integer().
-type sampler_ref() :: term().
-type step_op() ::
    {prefill, [erllama_nif:token_id()]}
    | {decode, sampler_ref()}.
-type step_result() ::
    prefilled
    | {token, erllama_nif:token_id(), 0 | 1}
    %% Thinking-phase token: scheduler detokenises and emits
    %% {erllama_token, Ref, {thinking_delta, Bin}} instead of a plain
    %% text fragment. Backends without extended-thinking support
    %% never emit this variant.
    | {thinking_token, erllama_nif:token_id()}
    %% Marker that the thinking phase has closed for this decode row.
    %% The scheduler resolves a signature via thinking_signature/1
    %% (or `<<>>` when the callback is not exported) and sends
    %% {erllama_thinking_end, Ref, Sig} before any subsequent token.
    | thinking_end.

-export_type([sampler_opts/0, seq_id/0, sampler_ref/0, step_op/0, step_result/0]).

%% Optional. Returns the integrity signature for the most recently
%% closed thinking block on the given sequence. Called by the
%% scheduler once it sees `thinking_end` in a step result, before
%% emitting the corresponding `erllama_thinking_end` message. The
%% binary is opaque to the scheduler and forwarded verbatim to the
%% caller; `<<>>` means "no signature available" and tells the
%% downstream to omit `signature_delta` from its wire output.
-callback thinking_signature(state(), seq_id()) -> binary().

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
