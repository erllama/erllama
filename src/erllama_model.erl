%% Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
%% See the LICENSE file at the project root.
%%
-module(erllama_model).
-moduledoc """
Per-model gen_statem that drives the request flow and wires the
cache subsystem into the model lifecycle.

## State machine

```
  ┌──── admit (idle_seq_ids non-empty)
  │     ┌──── admit (queues in pending when idle_seq_ids empty)
  │     │     ┌──── cast tick (self)
  ▼     ▼     │
 idle ──────▶ running ─────┐
  ▲                        │
  └────── all reqs finished (req_table = #{} AND pending = [])
```

Two states only:

- `idle/3` — `req_table` is empty AND `pending` is empty. Accepts
  admit events (complete, prefill_only, infer) and transitions to
  `running`. Verifies and other read-only ops are allowed.
- `running/3` — one or more `#req{}` are in flight. Accepts further
  admit events (allocate seq_id from `idle_seq_ids` or enqueue in
  `pending`), cancel casts, and the internal `tick` cast. Verify
  is refused with `{error, busy}` because it mutates the context.

## Per-request lifecycle

```
  admit                  step_tick                step_tick               finish_req
  ─────▶ req_table       ──────▶ prefilling       ──────▶ decoding        ────────▶
         (new #req,              (prefill_cursor          (prefill_cursor
          seq_id popped           non-empty,              undefined,
          from                    backend:step             backend:step
          idle_seq_ids,           pushes slice             samples one
          sampler built,          to KV)                   token + decodes)
          warm/cold path
          chosen)
```

Each `#req` records its own `seq_id`, sampler_ref, prompt_tokens,
context_tokens, prefill_cursor, generated, response_target,
cache_hit_kind, and finishing flag. The `req_table` map is keyed
by seq_id.

## step_tick driver

Every tick (one llama_decode call) builds a co-batched op list:

  - For each `#req` with `prefill_cursor =/= undefined`, append
    `{seq_id, {prefill, Slice}}`.
  - For each `#req` with `prefill_cursor =:= undefined` and a
    sampler_ref, append `{seq_id, {decode, sampler_ref}}`.

The op list is bounded by `total_batch_budget` (`context_opts.n_batch`):
decode rows are kept whole, prefill rows are sliced head-first
until the sum fits. The NIF returns `{seq_id, prefilled}` or
`{seq_id, {token, T, EogFlag}}` per row; results land back in the
respective `#req`. Reqs that reach `response_target` tokens, an
eog flag, or a cancel get `finishing = true` and finalise in the
post-step finisher walk (`finish_marked_reqs/2`).

## Per-tick batch budget

`step_tick/1` enforces total tokens ≤ `total_batch_budget` (mirrors
`context_opts.n_batch`, default 512). If `total_batch_budget` is
smaller than the number of in-flight decoders, the gen_statem
crashes deliberately so the supervisor restarts and the operator
fixes `n_batch` / `n_seq_max`. Otherwise prefill rows are sliced
head-first to fit; truncated tails resume next tick.

## Chunked prefill

Each prefill row is additionally capped by the `prefill_chunk_size`
policy knob (default `max(64, n_batch div 4)`, or `infinity` to
disable). The effective slice per prefill row is
`min(length(remaining), prefill_chunk_size, available_budget)`. A
long prompt is therefore sliced across several ticks even when the
batch budget alone would have accommodated it in one, leaving room
for concurrent decoders to make progress between chunks.

## Cache save reasons

- **cold**: fired right after a fresh prompt's prefill completes,
  before any decoding. Saves the trimmed prefix the policy
  produces from `cold_save_split/2`. Per-#req — each new admit
  fires its own cold save at most once.
- **continued**: fired every `continued_interval` tokens during
  decode. Per-#req, gated on the request's `last_save_at`.
- **finish**: fired when the request finishes (success, length
  limit, eog, or cancel). Saves the full context_tokens.
- **evict** / **shutdown**: fired by external triggers and walks
  every in-flight #req, firing one save per non-empty
  context_tokens.

All saves go through `fire_save_if/5` which calls
`backend:kv_pack/3` against the request's seq_id and hands the
binary off to `erllama_cache_writer`.

## Concurrency contract

The gen_statem is the sole writer of the context's KV cells: every
`backend:step/2` and `kv_pack` / `kv_unpack` / `seq_rm` call runs
inside a state callback, so the AGENTS.md paused-context invariant
holds — `kv_pack` only runs between ticks when no `llama_decode`
is in flight. Default `n_seq_max => 1` collapses this to the v0.2
single-tenant flow bit-identically; opting in via
`context_opts.n_seq_max > 1` lets up to N requests run
concurrently through one decode call per tick.

## Backwards compatibility

- Public API (`complete/2,3`, `prefill_only/2`, `infer/4`,
  `cancel/1`, `status/1`, `model_info/1`, `verify/4`, etc.) is
  unchanged.
- Default `n_seq_max => 1` keeps single-tenant behaviour
  bit-identical to v0.2; multi-tenancy is opt-in.
- `phase` on the obs row and in `model_info/1` is still
  `idle | prefilling | generating`, computed from the dominant
  phase across in-flight reqs (`dominant_phase/1`).
""".
-behaviour(gen_statem).

-include("erllama_cache.hrl").

-export([
    start_link/2,
    stop/1,
    complete/2,
    complete/3,
    prefill_only/2,
    infer/4,
    cancel/1,
    status/1,
    evict/1,
    shutdown/1,
    model_info/1,
    tokenize/2,
    detokenize/2,
    apply_chat_template/2,
    embed/2,
    load_adapter/2,
    unload_adapter/2,
    set_adapter_scale/3,
    list_adapters/1,
    get_backend_state/1,
    get_last_sampler_cfg/1,
    get_request_sampler_ref/1,
    get_policy/1,
    cache_key_meta/1,
    verify/4
]).

-export_type([
    model/0,
    model_info/0,
    stats/0,
    completion_result/0,
    prefill_result/0,
    cache_hit_kind/0,
    finish_reason/0,
    infer_params/0
]).

-type model() :: erllama_registry:model_id() | pid().
-type model_info() :: #{
    id := binary(),
    %% Alias for `id`. Added for cluster registry rows so callers
    %% match on a name that does not collide with their own
    %% process-id-typed `id` fields.
    model_id := binary(),
    pid := pid(),
    status := idle | prefilling | generating,
    backend := module(),
    context_size := non_neg_integer(),
    quant_type := atom(),
    quant_bits := non_neg_integer(),
    %% String tag like <<"q4_k_m">> / <<"f16">>. Derived from
    %% quant_type + quant_bits.
    quant_tag := binary(),
    tier := disk | ram_file,
    fingerprint := binary(),
    %% erlang:monotonic_time(nanosecond) at gen_statem init.
    loaded_at_monotonic := integer(),
    %% Best-effort estimate of VRAM footprint for the model when
    %% the gpu offload is configured. 0 if no GPU layers are
    %% offloaded or if the backend cannot report the underlying
    %% sizes (stub backend, etc.).
    vram_estimate_b := non_neg_integer()
}.

-type cache_hit_kind() :: exact | partial | cold.

-type pending_request() ::
    {complete, gen_statem:from(), binary(), map()}
    | {prefill_only, gen_statem:from(), [non_neg_integer()]}
    | {infer, gen_statem:from(), [non_neg_integer()], map(), pid()}.
-type finish_reason() :: stop | length | cancelled.
-type stats() :: #{
    prompt_tokens := non_neg_integer(),
    completion_tokens := non_neg_integer(),
    prefill_ms := non_neg_integer(),
    generation_ms := non_neg_integer(),
    cache_hit_kind := cache_hit_kind(),
    finish_reason := finish_reason(),
    cancelled := boolean(),
    %% Token-exact cache key for the full context (prompt ++ generated).
    %% `undefined` when the finish save was suppressed (e.g. live token
    %% count below `min_tokens`).
    finish_key := binary() | undefined,
    %% Length of `context_tokens` at finish (prompt + generated). Equal
    %% to `prompt_tokens + completion_tokens` unless the cache pruned
    %% the live context (not currently possible).
    committed_tokens := non_neg_integer(),
    %% Anthropic-style per-request cache breakdown. `read` is the
    %% warm prefix length restored from cache at admission;
    %% `created` is the largest contribution this request added to
    %% the cache beyond that prefix (saves below `min_tokens` and
    %% writer back-pressure leave `created` unchanged). Both
    %% default to 0.
    cache_delta := #{
        read := non_neg_integer(),
        created := non_neg_integer()
    },
    %% Only present when a caller-supplied `stop_sequences` entry
    %% fired. The value is the binary of the matched stop string.
    %% Absent on `length`, `cancelled`, or natural end-of-generation
    %% without a stop-string match.
    stop_sequence => binary()
}.

%% Reply shape for `complete/2,3`.
-type completion_result() :: #{
    %% Detokenised reply text. Trimmed at the first occurrence of a
    %% matched `stop_sequences` entry when one fired.
    reply := binary(),
    %% Tokens produced by this request (not including the prompt).
    generated := [non_neg_integer()],
    %% Full context as a token list (prompt ++ generated).
    context_tokens := [non_neg_integer()],
    %% Convenience: length(context_tokens).
    committed_tokens := non_neg_integer(),
    %% Token-exact cache key for the full context. Pass as
    %% `parent_key` on the next request to resume from the warm row.
    %% `undefined` if the finish save was suppressed.
    finish_key := binary() | undefined,
    %% How this request resolved against the cache on admission.
    cache_hit_kind := cache_hit_kind(),
    finish_reason := finish_reason(),
    %% Per-request Anthropic cache breakdown. See `stats()`.
    cache_delta := #{
        read := non_neg_integer(),
        created := non_neg_integer()
    },
    stats := stats(),
    %% Only present when a caller-supplied `stop_sequences` entry
    %% fired. The value is the binary of the matched stop string.
    stop_sequence => binary()
}.

%% Reply shape for `prefill_only/2`.
-type prefill_result() :: #{
    context_tokens := [non_neg_integer()],
    committed_tokens := non_neg_integer(),
    finish_key := binary() | undefined,
    cache_hit_kind := cache_hit_kind(),
    %% Per-request Anthropic cache breakdown. See `stats()`.
    cache_delta := #{
        read := non_neg_integer(),
        created := non_neg_integer()
    }
}.

%% Optional fields the caller may set on `infer/4`. The same fields
%% are honoured by `complete/3` Opts. The sampler chain is rebuilt
%% per-request: grammar -> repetition_penalty -> top_k -> top_p ->
%% min_p -> (temperature > 0 ? temp -> dist(seed) : greedy).
%% `stop_sequences` is a list of binaries; generation halts on the
%% first occurrence (by list order) of any element in the
%% accumulated detokenized output. The match is trimmed from the
%% streamed text and the matched value is reported as
%% `stop_sequence` in the result map / stats.
-type infer_params() :: #{
    response_tokens => pos_integer(),
    parent_key => term(),
    temperature => float(),
    top_p => float(),
    top_k => pos_integer(),
    min_p => float(),
    repetition_penalty => float(),
    seed => non_neg_integer(),
    stop_sequences => [binary()],
    grammar => binary(),
    %% Opt-in extended thinking. When `enabled`, a thinking-capable
    %% backend may emit `{thinking_token, _}` step results that the
    %% scheduler forwards to the streaming caller as
    %% `{erllama_token, Ref, {thinking_delta, Bin}}`. When the
    %% thinking phase closes the scheduler emits a single
    %% `{erllama_thinking_end, Ref, Sig}` before any non-thinking
    %% token. Defaults to `disabled`; backends without thinking
    %% support ignore the flag entirely.
    thinking => enabled | disabled,
    %% Caller-side cap on the thinking phase. Once the scheduler has
    %% forwarded this many `{thinking_delta, _}` payloads for the
    %% request, it synthesises the `erllama_thinking_end` close
    %% early and re-routes any further `{thinking_token, _}` step
    %% results through the normal token pipeline so generation
    %% progresses. Non-positive values and a missing key both mean
    %% "no cap". Only meaningful with `thinking => enabled`.
    thinking_budget_tokens => pos_integer(),
    _ => _
}.

-export([
    init/1,
    callback_mode/0,
    terminate/3
]).

%% State callbacks
-export([idle/3, running/3]).

%% Per-request state. The scheduler holds one #req per in-flight
%% request, indexed by seq_id in #data.req_table. With the default
%% `n_seq_max => 1` exactly one request runs at a time; with
%% n_seq_max > 1 the scheduler co-batches multiple in-flight reqs
%% into one llama_decode tick.
-record(req, {
    seq_id :: non_neg_integer(),
    mode :: standard | streaming | prefill_only,
    caller :: gen_statem:from() | undefined,
    caller_pid :: pid() | undefined,
    request_ref :: reference() | undefined,
    cancel_pending = false :: boolean(),
    prompt_tokens :: [non_neg_integer()],
    %% Tokens already pushed into KV for this request (warm prefix
    %% from cache + every prefill slice + every decoded token).
    context_tokens :: [non_neg_integer()],
    response_target :: non_neg_integer(),
    generated :: [non_neg_integer()],
    last_save_at :: non_neg_integer(),
    %% Tokens still to push through prefill. `undefined` means the
    %% request is decode-ready. A non-empty list means step_tick
    %% will continue prefilling these on the next tick.
    prefill_cursor = undefined :: [non_neg_integer()] | undefined,
    %% effective_fp snapshot captured at admission so a mid-request
    %% adapter mutation cannot shift the cache identity for this
    %% in-flight request.
    request_fp :: <<_:256>> | undefined,
    cache_hit_kind = cold :: cache_hit_kind(),
    cache_hit_prefix_len = 0 :: non_neg_integer(),
    prefill_started_at :: integer() | undefined,
    generation_started_at :: integer() | undefined,
    %% Per-request sampler chain handle. `undefined` for prefill_only.
    sampler_ref :: term() | undefined,
    last_sampler_cfg = undefined :: map() | undefined,
    %% Tokens to prefill AFTER the next cold save fires. Cold-save
    %% policy splits the prompt into a trimmed prefix (which lands
    %% as the cold save) and a remainder. The trimmed prefix goes
    %% into prefill_cursor; this field holds the remainder. After
    %% the trim's prefill tick fires the cold save, this gets
    %% rotated into prefill_cursor on the next tick. `undefined`
    %% when no cold save is pending (no_save policy, prefill_only
    %% mode, or warm-restore path).
    cold_save_remaining = undefined :: undefined | [non_neg_integer()],
    %% Set when this request has emitted its final result. The next
    %% step_tick iteration drops it from req_table; deferring the
    %% removal lets the tick walk results without mid-iteration
    %% mutation of req_table.
    finishing = false :: boolean(),
    %% Caller-supplied stop strings; first-match-wins by list order.
    %% Empty disables the scanner.
    stop_sequences = [] :: [binary()],
    %% binary:compile_pattern/1 result for stop_sequences. `undefined`
    %% when stop_sequences = [].
    stop_pattern = undefined :: binary:cp() | undefined,
    %% Max byte length across stop_sequences. Used to decide how many
    %% trailing bytes of detokenized output to hold back in
    %% pending_text so a stop string spanning a chunk boundary is
    %% still detected.
    stop_max_len = 0 :: non_neg_integer(),
    %% Detokenized bytes not yet flushed to the caller. Suffix of the
    %% generated text that may be the prefix of a future stop match.
    pending_text = <<>> :: binary(),
    %% Matched stop string once a hit fires; `undefined` otherwise.
    %% Drives the finish_reason classifier and the `stop_sequence`
    %% key on the result / stats map.
    matched_stop = undefined :: binary() | undefined,
    %% Per-request thinking opt-in. The scheduler treats any
    %% `{thinking_token, _}` step result on a request with
    %% `thinking = disabled` as a backend bug (stray thinking
    %% output not opted in by the caller).
    thinking = disabled :: enabled | disabled,
    %% Running concat of detokenized thinking-phase bytes. Passed
    %% to the backend's `thinking_signature/3` so it can sign the
    %% observed text directly (e.g. via HMAC). Reset to <<>> on
    %% each thinking_end so a second thinking block on the same
    %% request gets its own signature input.
    thinking_bytes = <<>> :: binary(),
    %% Set when the request was already failed with an
    %% `erllama_error` (e.g. backend contract violation). The
    %% finish path skips the `erllama_done` send in that case so
    %% the caller doesn't see both.
    errored = undefined :: term() | undefined,
    %% Anthropic-style cache delta: tokens this request added to
    %% cache that weren't already there. Tracks the largest save
    %% contribution above the warm prefix seen during the request
    %% lifetime. Combined with `cache_hit_prefix_len` (which is the
    %% read side) it powers the `cache_delta` map surfaced on
    %% `completion_result()`, `stats()`, and `prefill_result()`.
    cache_delta_created = 0 :: non_neg_integer(),
    %% Running concat of detokenised tool-call bytes for the current
    %% span. Forwarded verbatim to the streaming caller as the
    %% `Full` payload of `{erllama_tool_call_end, _, Full}` when the
    %% span closes, then reset for any subsequent span on the same
    %% request.
    tool_call_bytes = <<>> :: binary(),
    %% Caller-side thinking-phase cap (`undefined` means no cap).
    %% Counted in delivered `{thinking_delta, _}` payloads, not raw
    %% thinking_token step results, so an empty detokenisation does
    %% not consume the budget.
    thinking_budget = undefined :: pos_integer() | undefined,
    %% Number of `{thinking_delta, _}` payloads delivered so far for
    %% this request. Compared against `thinking_budget` after each
    %% emit; on overflow the scheduler synthesises the
    %% `erllama_thinking_end` close immediately.
    thinking_count = 0 :: non_neg_integer(),
    %% Set after the budget-triggered `thinking_end`. Any further
    %% `{thinking_token, _}` step result is treated as a normal
    %% `{token, _, 0}` for the rest of the request so generation
    %% progresses without surfacing further `thinking_delta`s.
    thinking_capped = false :: boolean()
}).

-record(data, {
    model_id :: binary(),
    tier_srv :: atom(),
    tier :: disk | ram_file,
    %% Base model fingerprint; constant for the life of the model.
    fingerprint :: <<_:256>>,
    fingerprint_mode :: safe | gguf_chunked | fast_unsafe,
    quant_type :: erllama_cache_key:quant_type(),
    quant_bits :: non_neg_integer(),
    ctx_params_hash :: <<_:256>>,
    context_size :: non_neg_integer(),
    policy :: erllama_cache_policy:config(),
    %% Inference backend: erllama_model_stub | erllama_model_llama.
    backend :: module(),
    backend_state :: term(),
    %% Captured at init for list_models metadata.
    loaded_at_monotonic :: integer(),
    %% Best-effort VRAM footprint of the loaded model, in bytes.
    %% 0 when no GPU layers are offloaded or when the backend does
    %% not report enough metadata to derive it.
    vram_estimate_b = 0 :: non_neg_integer(),
    %% Attached LoRA adapters. Each entry holds the backend's opaque
    %% handle, the file sha256 (for cache-key derivation), and the
    %% current scale. effective_fp = sha256(fingerprint || sorted
    %% pairs of sha+scale). Recomputed on every attachment change.
    adapters = [] :: [#{handle := term(), sha := <<_:256>>, scale := float()}],
    effective_fp :: <<_:256>>,
    %% In-flight requests indexed by seq_id. Empty when the model
    %% is idle.
    req_table = #{} :: #{non_neg_integer() => #req{}},
    %% Free list of seq_ids available for admission. Initialised to
    %% [0, 1, ..., n_seq_max - 1] at init/1; head-pop on admit,
    %% head-push on finish.
    idle_seq_ids = [0] :: [non_neg_integer()],
    %% Total seq_ids available; mirrors `context_opts.n_seq_max`.
    %% Default 1 keeps single-tenant behaviour bit-identical.
    n_seq_max = 1 :: pos_integer(),
    %% Maximum total tokens per step tick. Bounds the co-batched
    %% llama_decode so the NIF never exceeds the context's n_batch.
    %% Mirrors `context_opts.n_batch`; step_tick slices prefill rows
    %% if needed to fit.
    total_batch_budget = 512 :: pos_integer(),
    %% Snapshot of the last cache hit for the obs row. Updated on
    %% admission; survives across requests so external routers can
    %% read it between admissions.
    last_cache_hit_kind = undefined :: cache_hit_kind() | undefined,
    last_cache_hit_prefix_len = 0 :: non_neg_integer(),
    %% Snapshot of the sampler-config map fed to backend:sampler_new
    %% on the most recent admission. Test-visible via
    %% `get_last_sampler_cfg/1`.
    last_sampler_cfg = undefined :: map() | undefined,
    %% Test-visible holdover so erllama_sampler_tests can inspect
    %% the current sampler_ref after a complete returns.
    last_sampler_ref = undefined :: term() | undefined,
    %% FIFO queue of admits that arrived while idle_seq_ids was
    %% empty. Each entry is one of:
    %%   {complete, From, Prompt, Opts}
    %%   {prefill_only, From, PromptTokens}
    %%   {infer, From, Tokens, Params, CallerPid}
    pending = [] :: [pending_request()]
}).

%% =============================================================================
%% Public API
%% =============================================================================

-spec start_link(binary(), map()) -> {ok, pid()} | {error, term()}.
start_link(ModelId, Config) when is_binary(ModelId) ->
    gen_statem:start_link(
        {via, erllama_registry, ModelId}, ?MODULE, [ModelId, Config], []
    ).

-spec stop(model()) -> ok.
stop(Model) ->
    gen_statem:stop(via(Model)).

-spec complete(model(), binary()) ->
    {ok, completion_result()} | {error, term()}.
complete(Model, Prompt) ->
    complete(Model, Prompt, #{}).

-spec complete(model(), binary(), map()) ->
    {ok, completion_result()} | {error, term()}.
complete(Model, Prompt, Opts) ->
    gen_statem:call(via(Model), {complete, Prompt, Opts}, infinity).

-doc """
Decode a prompt into KV state and fire a finish save, without
sampling any output tokens. Returns the `finish_key` so the caller
can hand it as `parent_key` to a subsequent `complete/3` or
`infer/4` for token-exact warm restore.

`PromptTokens` is the prompt as a list of token ids. Tokenisation
is the caller's responsibility (use `tokenize/2` or apply a chat
template first). The cache behaviour mirrors `complete/3`: an exact
or longest-prefix warm restore is taken when available, otherwise
the prompt is prefilled cold.

`finish_key` is `undefined` if the finish save was suppressed
because the token count is below the configured `min_tokens`.
""".
-spec prefill_only(model(), [non_neg_integer()]) ->
    {ok, prefill_result()} | {error, term()}.
prefill_only(Model, PromptTokens) when is_list(PromptTokens) ->
    gen_statem:call(via(Model), {prefill_only, PromptTokens}, infinity).

-doc """
Streaming inference. Admits a request and immediately returns a
unique `reference()`; tokens are delivered to `CallerPid` via
asynchronous messages:

- `{erllama_token, Ref, binary()}` per generated token (text fragment;
  suppressed when the detokenized binary is empty)
- `{erllama_token_id, Ref, integer()}` per generated token (always
  delivered, including for tokens whose text fragment is empty;
  used by speculative-decoding collectors)
- `{erllama_done, Ref, stats()}` on normal completion
- `{erllama_error, Ref, term()}` on failure

`Tokens` is the prompt as a list of token ids - tokenisation is the
caller's responsibility (use `tokenize/2` or apply a chat template
first). `Params` is an `infer_params()` map.

Calls that arrive while a previous request is in flight are queued
FIFO. The reply `{ok, Ref}` is sent as soon as the call is admitted;
streaming events follow once the queue head advances to this
request.
""".
-spec infer(model(), [non_neg_integer()], infer_params(), pid()) ->
    {ok, reference()} | {error, term()}.
infer(Model, Tokens, Params, CallerPid) when
    is_list(Tokens), is_map(Params), is_pid(CallerPid)
->
    gen_statem:call(via(Model), {infer, Tokens, Params, CallerPid}, infinity).

-doc """
Cancel an in-flight streaming inference. Idempotent and fire-and-
forget: returns `ok` even if the ref is unknown (already finished or
never existed). The cancellation is observed at the next
inter-token boundary; the model emits a final `{erllama_done, Ref,
Stats}` with `cancelled => true` after the running decode step
completes.
""".
-spec cancel(reference()) -> ok.
cancel(Ref) when is_reference(Ref) ->
    case erllama_inflight:lookup(Ref) of
        {ok, ModelPid} ->
            gen_statem:cast(ModelPid, {cancel, Ref}),
            ok;
        {error, not_found} ->
            ok
    end.

-spec status(model()) -> idle | prefilling | generating.
status(Model) ->
    gen_statem:call(via(Model), status).

-doc """
Request that the model evict its current state. Fires an `evict`
save synchronously if there is anything in the context. Called by
`erllama_scheduler` (future) when GPU memory pressure requires this
model to release its context handle. No-op when the model is idle
with no live context.
""".
-spec evict(model()) -> ok.
evict(Model) ->
    gen_statem:call(via(Model), evict).

-doc """
Fire a `shutdown` save synchronously and return. Called from the
application's `prep_stop` hook so live state survives a graceful
restart.
""".
-spec shutdown(model()) -> ok.
shutdown(Model) ->
    gen_statem:call(via(Model), shutdown).

-doc """
Snapshot of the model's metadata.

Returns a `model_info()` map with status, context size, quantisation,
backend, fingerprint, and tier. Safe to call from any state - the
gen_statem handles it as a common event without disrupting in-flight
inference.
""".
-spec model_info(model()) -> model_info().
model_info(Model) ->
    gen_statem:call(via(Model), model_info).

-doc """
Tokenise a string using the model's tokenizer. Returns a list of
token IDs. Safe to call concurrently with `complete/2,3`; tokenisation
runs against the model's static vocabulary, not the live KV cache.
""".
-spec tokenize(model(), binary()) ->
    {ok, [non_neg_integer()]} | {error, term()}.
tokenize(Model, Text) when is_binary(Text) ->
    gen_statem:call(via(Model), {tokenize, Text}).

-doc """
Detokenise a list of token IDs back to a string. Safe to call
concurrently with `complete/2,3`.
""".
-spec detokenize(model(), [non_neg_integer()]) ->
    {ok, binary()} | {error, term()}.
detokenize(Model, Tokens) when is_list(Tokens) ->
    gen_statem:call(via(Model), {detokenize, Tokens}).

-doc """
Render a normalised chat request through the model's chat template
and tokenise in one step. The Request map carries `messages`,
`system`, and `tools`; the per-model template decides where each
field lands in the prompt.

Returns `{error, not_supported}` if the backend does not implement
chat templating.
""".
-spec apply_chat_template(model(), erllama_model_backend:chat_request()) ->
    {ok, [non_neg_integer()]} | {error, term()}.
apply_chat_template(Model, Request) when is_map(Request) ->
    gen_statem:call(via(Model), {apply_chat_template, Request}).

-doc """
Compute an embedding vector for the given prompt tokens.
""".
-spec embed(model(), [non_neg_integer()]) ->
    {ok, [float()]} | {error, term()}.
embed(Model, Tokens) when is_list(Tokens) ->
    gen_statem:call(via(Model), {embed, Tokens}).

-doc """
Load a LoRA adapter from a GGUF file and attach it to the model
with scale 1.0. Returns an opaque handle the caller threads into
`unload_adapter/2` and `set_adapter_scale/3`. The adapter's sha256 is
folded into the effective fingerprint so cache rows produced under
this adapter never collide with rows from a different adapter set.
""".
-spec load_adapter(model(), file:filename_all()) ->
    {ok, term()} | {error, term()}.
load_adapter(Model, Path) ->
    gen_statem:call(via(Model), {load_adapter, Path}).

-doc """
Detach + free a previously loaded adapter. Idempotent: a second call
on the same handle returns `ok`.
""".
-spec unload_adapter(model(), term()) -> ok | {error, term()}.
unload_adapter(Model, Handle) ->
    gen_statem:call(via(Model), {unload_adapter, Handle}).

-doc """
Change an attached adapter's scale. Re-applies the full set on the
underlying context.
""".
-spec set_adapter_scale(model(), term(), float()) -> ok | {error, term()}.
set_adapter_scale(Model, Handle, Scale) when is_number(Scale) ->
    gen_statem:call(via(Model), {set_adapter_scale, Handle, float(Scale)}).

-doc """
List currently attached adapters as `[#{handle => H, scale => F}]`.
The handle is the same opaque value `load_adapter/2` returned.
""".
-spec list_adapters(model()) -> [#{handle := term(), scale := float()}].
list_adapters(Model) ->
    gen_statem:call(via(Model), list_adapters).

%% =============================================================================
%% gen_statem callbacks
%% =============================================================================

callback_mode() -> state_functions.

%% Test-only: returns the current backend state. Used by sampler
%% plumbing tests to assert that configure_sampler/2 lands the right
%% map on the stub backend. Not part of the public API; the test
%% suite is the only caller.
-doc false.
get_backend_state(Model) ->
    {_State, Data} = sys:get_state(via(Model)),
    Data#data.backend_state.

%% Test-only: returns the sampler-config map the most recent admit
%% passed into backend:sampler_new/2. `undefined` if no sampler has
%% been built (fresh model, or last request was prefill_only). Used
%% by erllama_sampler_tests to verify the cfg projection without
%% poking at the opaque sampler_ref.
-doc false.
get_last_sampler_cfg(Model) ->
    {_State, Data} = sys:get_state(via(Model)),
    Data#data.last_sampler_cfg.

%% Test-only: returns the current per-request sampler_ref or
%% `undefined`. Used to assert the ref is freed at finish.
-doc false.
get_request_sampler_ref(Model) ->
    {_State, Data} = sys:get_state(via(Model)),
    Data#data.last_sampler_ref.

%% Test-only: returns the resolved policy map. Used to assert
%% defaults (e.g. `prefill_chunk_size`) without poking record
%% layout from the test module.
-doc false.
get_policy(Model) ->
    {_State, Data} = sys:get_state(via(Model)),
    Data#data.policy.

%% Snapshot of the cache key triple a probe needs to hit the
%% cache for this model's current state. Effective fingerprint
%% (with attached LoRA composition) so the lookup matches what
%% runtime requests would hit.
-spec cache_key_meta(model()) ->
    #{fingerprint := binary(), quant_type := atom(), ctx_params_hash := binary()}.
cache_key_meta(Model) ->
    gen_statem:call(via(Model), cache_key_meta).

%% Speculative-decoding verifier. Synchronous; runs the verifier
%% pass against the model's context and returns
%% {ok, AcceptedCount, NextToken}. Only allowed when the model
%% gen_statem is idle: a concurrent in-flight infer would have its
%% context state mutated by the verify pass, so we reject from
%% other states with {error, busy}.
-spec verify(
    model(),
    [erllama_nif:token_id()],
    [erllama_nif:token_id()],
    pos_integer()
) ->
    {ok, non_neg_integer(), erllama_nif:token_id() | eos} | {error, term()}.
verify(Model, PrefixTokens, Candidates, K) ->
    gen_statem:call(via(Model), {verify, PrefixTokens, Candidates, K}).

init([ModelId, Config]) ->
    Backend = maps:get(backend, Config, erllama_model_stub),
    case Backend:init(Config) of
        {ok, BState} ->
            Data = build_init_data(ModelId, Config, Backend, BState),
            ok = obs_install_initial(Data),
            {ok, idle, Data};
        {error, Reason} ->
            {stop, Reason}
    end.

build_init_data(ModelId, Config, Backend, BState) ->
    Fp = maps:get(fingerprint, Config, default_fingerprint()),
    CtxOpts = maps:get(context_opts, Config, #{}),
    NSeqMax = maps:get(n_seq_max, CtxOpts, 1),
    NBatch = maps:get(n_batch, CtxOpts, 512),
    #data{
        model_id = ModelId,
        tier_srv = maps:get(tier_srv, Config, erllama_cache_ram),
        tier = maps:get(tier, Config, ram),
        fingerprint = Fp,
        fingerprint_mode = maps:get(fingerprint_mode, Config, safe),
        quant_type = maps:get(quant_type, Config, f16),
        quant_bits = maps:get(quant_bits, Config, 16),
        ctx_params_hash = maps:get(ctx_params_hash, Config, default_ctx_params_hash()),
        context_size = maps:get(context_size, Config, 4096),
        policy = resolve_policy(Config, NBatch),
        backend = Backend,
        backend_state = BState,
        adapters = [],
        effective_fp = Fp,
        loaded_at_monotonic = erlang:monotonic_time(nanosecond),
        vram_estimate_b = compute_vram_estimate(Backend, BState),
        req_table = #{},
        idle_seq_ids = lists:seq(0, NSeqMax - 1),
        n_seq_max = NSeqMax,
        total_batch_budget = max(1, NBatch)
    }.

%% Best-effort: ask the backend for the byte size, total layer count,
%% and n_gpu_layers it captured at load time. Backends without the
%% optional callback (or that return missing keys) get 0.
compute_vram_estimate(Backend, BState) ->
    case erlang:function_exported(Backend, extra_metadata, 1) of
        false ->
            0;
        true ->
            Meta = Backend:extra_metadata(BState),
            Size = maps:get(model_size_bytes, Meta, 0),
            Total = maps:get(total_layers, Meta, 0),
            NGpu = maps:get(n_gpu_layers, Meta, 0),
            case {Size, Total, NGpu} of
                {0, _, _} -> 0;
                {_, 0, _} -> 0;
                {_, _, NG} when NG =< 0 -> Size;
                {_, T, NG} when NG >= T -> Size;
                {S, T, NG} -> (S * NG) div T
            end
    end.

%% Per-model policy. Caller can override any subset; missing keys
%% fall back to the app env defaults declared in `erllama.app.src`.
%%
%% `prefill_chunk_size` defaults to `max(64, NBatch div 4)` so a
%% long prompt doesn't monopolise the batch and stall concurrent
%% decoders. Pass `infinity` to disable per-row chunking (the per-
%% tick batch budget still applies).
resolve_policy(Config, NBatch) ->
    Defaults = #{
        min_tokens => application:get_env(erllama, min_tokens, 512),
        cold_min_tokens => application:get_env(erllama, cold_min_tokens, 512),
        cold_max_tokens => application:get_env(erllama, cold_max_tokens, 30000),
        continued_interval => application:get_env(erllama, continued_interval, 2048),
        boundary_trim_tokens => application:get_env(erllama, boundary_trim_tokens, 32),
        boundary_align_tokens => application:get_env(erllama, boundary_align_tokens, 2048),
        session_resume_wait_ms => application:get_env(erllama, session_resume_wait_ms, 500),
        prefill_chunk_size => max(64, NBatch div 4)
    },
    maps:merge(Defaults, maps:get(policy, Config, #{})).

terminate(_Reason, _State, #data{model_id = ModelId, backend = B, backend_state = S}) ->
    _ = erllama_inflight:obs_delete(ModelId),
    B:terminate(S),
    ok;
terminate(_Reason, _State, _Data) ->
    ok.

%% Placeholder fingerprint when none supplied.
%%
%% Production code must always pass a real fingerprint via
%% `crypto:hash(sha256, ModelBytes)`. The default exists only so the
%% minimal `load_model/1` example in the docs runs without an
%% operator having to compute a hash first.
%%
%% Sharing the default across two distinct models lets the cache
%% accidentally false-hit between them (same default fp + same
%% tokens + same ctx_params -> same key). Hardly anyone hits this
%% in practice because real prompts differ, but it is unsafe under
%% adversarial inputs.
default_fingerprint() ->
    binary:copy(<<0>>, 32).

%% Same caveat as default_fingerprint/0. Pass a real
%% `crypto:hash(sha256, term_to_binary({Nctx, Nbatch}))` in
%% production.
default_ctx_params_hash() ->
    binary:copy(<<0>>, 32).

%% =============================================================================
%% State: idle
%% =============================================================================

idle({call, From}, {complete, Prompt, Opts}, Data) ->
    admit({complete, From, Prompt, Opts}, Data);
idle({call, From}, {prefill_only, PromptTokens}, Data) ->
    admit({prefill_only, From, PromptTokens}, Data);
idle({call, From}, {infer, Tokens, Params, CallerPid}, Data) ->
    admit({infer, From, Tokens, Params, CallerPid}, Data);
idle({call, From}, status, Data) ->
    {keep_state, Data, [{reply, From, idle}]};
idle({call, From}, {verify, PrefixTokens, Candidates, K}, Data) ->
    Reply = run_verify(PrefixTokens, Candidates, K, Data),
    case Reply of
        {ok, _, _, NewBState} ->
            NewData = Data#data{backend_state = NewBState},
            {keep_state, NewData, [{reply, From, public_verify_reply(Reply)}]};
        {error, _} = E ->
            {keep_state, Data, [{reply, From, E}]}
    end;
idle(EventType, EventContent, Data) ->
    handle_common(idle, EventType, EventContent, Data).

%% =============================================================================
%% State: running (one or more in-flight requests)
%% =============================================================================

running({call, From}, {complete, Prompt, Opts}, Data) ->
    admit({complete, From, Prompt, Opts}, Data);
running({call, From}, {prefill_only, PromptTokens}, Data) ->
    admit({prefill_only, From, PromptTokens}, Data);
running({call, From}, {infer, Tokens, Params, CallerPid}, Data) ->
    admit({infer, From, Tokens, Params, CallerPid}, Data);
running({call, From}, status, Data) ->
    %% Phase reported is the dominant phase across in-flight reqs:
    %% if any seq is still prefilling, report `prefilling`; else
    %% `generating`. Empty req_table only happens between ticks.
    Phase = dominant_phase(Data),
    {keep_state, Data, [{reply, From, Phase}]};
running({call, From}, {verify, _, _, _}, Data) ->
    %% Verify mutates the live context; refuse while any seq is in
    %% flight to keep the snapshot/restore invariant intact.
    {keep_state, Data, [{reply, From, {error, busy}}]};
running(cast, tick, Data) ->
    step_tick(Data);
running(EventType, EventContent, Data) ->
    handle_common(running, EventType, EventContent, Data).

run_verify(PrefixTokens, Candidates, K, Data) ->
    Backend = Data#data.backend,
    case erlang:function_exported(Backend, verify, 4) of
        false ->
            {error, not_supported};
        true ->
            Backend:verify(Data#data.backend_state, PrefixTokens, Candidates, K)
    end.

public_verify_reply({ok, Accepted, NextToken, _NewBState}) ->
    {ok, Accepted, NextToken}.

%% Admit one pending_request(). Pops a seq_id from idle_seq_ids and
%% kicks the request off; if no seq_ids are free, queues in
%% `pending` (the caller's gen_statem:call still blocks since we do
%% not reply until the queued request is eventually started).
admit(Item, Data = #data{idle_seq_ids = []}) ->
    %% No seq_ids available — queue. The caller stays blocked on
    %% gen_statem:call until step_tick frees a slot and the
    %% dispatch path runs this admit. For streaming infer/4 the
    %% caller still doesn't get its Ref until then.
    NewData = enqueue(Item, Data),
    case Data#data.req_table of
        Empty when map_size(Empty) =:= 0 -> {next_state, idle, NewData};
        _ -> {keep_state, NewData}
    end;
admit(Item, Data = #data{idle_seq_ids = [SeqId | Rest]}) ->
    case start_request(Item, SeqId, Data#data{idle_seq_ids = Rest}) of
        {ok, NewData, Actions} ->
            schedule_tick(),
            {next_state, running, NewData, Actions};
        {error, Reason, From} ->
            %% Sampler build or pre-validation failed. Return the
            %% seq_id to the free list and reply to the caller.
            Data1 = Data#data{idle_seq_ids = [SeqId | Rest]},
            {keep_state, Data1, [{reply, From, {error, Reason}}]}
    end.

%% Build a #req for `Item`, run the cache lookup, install in
%% req_table. Returns {ok, NewData, ReplyActions} on success or
%% {error, Reason, From} on a synchronous-rejection path (e.g.
%% sampler_new failed).
start_request({complete, From, Prompt, Opts}, SeqId, Data) ->
    case sampler_for(Opts, Data) of
        {ok, SamplerRef, SamplerCfg, Data0} ->
            PromptTokens = backend_call(Data0, tokenize, [Prompt]),
            {Stops, StopPat, StopMax} = stop_sequences_from(Opts),
            Req = #req{
                seq_id = SeqId,
                mode = standard,
                caller = From,
                prompt_tokens = PromptTokens,
                response_target = maps:get(response_tokens, Opts, 4),
                generated = [],
                last_save_at = 0,
                context_tokens = [],
                request_fp = Data0#data.effective_fp,
                sampler_ref = SamplerRef,
                last_sampler_cfg = SamplerCfg,
                prefill_started_at = erlang:monotonic_time(millisecond),
                stop_sequences = Stops,
                stop_pattern = StopPat,
                stop_max_len = StopMax,
                thinking = thinking_from(Opts),
                thinking_budget = thinking_budget_from(Opts)
            },
            ParentKey = maps:get(parent_key, Opts, undefined),
            Req1 = setup_lookup(Req, ParentKey, Data0),
            Data1 = put_req(Data0, Req1),
            {ok, snapshot_admission(Data1, Req1), []};
        {error, Reason} ->
            {error, Reason, From}
    end;
start_request({prefill_only, From, PromptTokens}, SeqId, Data) ->
    Req = #req{
        seq_id = SeqId,
        mode = prefill_only,
        caller = From,
        prompt_tokens = PromptTokens,
        response_target = 0,
        generated = [],
        last_save_at = 0,
        context_tokens = [],
        request_fp = Data#data.effective_fp,
        sampler_ref = undefined,
        last_sampler_cfg = undefined,
        prefill_started_at = erlang:monotonic_time(millisecond)
    },
    Req1 = setup_lookup(Req, undefined, Data),
    Data1 = put_req(Data, Req1),
    %% sampler_ref reset on the data-level holdover so
    %% get_request_sampler_ref reflects current state.
    Data2 = Data1#data{last_sampler_ref = undefined},
    {ok, snapshot_admission(Data2, Req1), []};
start_request({infer, From, Tokens, Params, CallerPid}, SeqId, Data) ->
    case sampler_for(Params, Data) of
        {ok, SamplerRef, SamplerCfg, Data0} ->
            Ref = make_ref(),
            ok = erllama_inflight:register(Ref, self()),
            {Stops, StopPat, StopMax} = stop_sequences_from(Params),
            Req = #req{
                seq_id = SeqId,
                mode = streaming,
                caller_pid = CallerPid,
                request_ref = Ref,
                prompt_tokens = Tokens,
                response_target = maps:get(response_tokens, Params, 64),
                generated = [],
                last_save_at = 0,
                context_tokens = [],
                request_fp = Data0#data.effective_fp,
                sampler_ref = SamplerRef,
                last_sampler_cfg = SamplerCfg,
                prefill_started_at = erlang:monotonic_time(millisecond),
                stop_sequences = Stops,
                stop_pattern = StopPat,
                stop_max_len = StopMax,
                thinking = thinking_from(Params),
                thinking_budget = thinking_budget_from(Params)
            },
            ParentKey = maps:get(parent_key, Params, undefined),
            Req1 = setup_lookup(Req, ParentKey, Data0),
            Data1 = put_req(Data0, Req1),
            {ok, snapshot_admission(Data1, Req1), [{reply, From, {ok, Ref}}]};
        {error, Reason} ->
            {error, Reason, From}
    end.

%% Run the cache lookup for this request's prompt and set up the
%% #req's `prefill_cursor` and `context_tokens` accordingly. The
%% warm path runs the kv_unpack + seq_rm_last primer flow under
%% the request's seq_id; the cold path leaves the full prompt in
%% the prefill cursor.
setup_lookup(Req, ParentKey, Data) ->
    case lookup_or_resume(Req#req.prompt_tokens, ParentKey, Req, Data) of
        {warm, ContextTokens, RemainingTokens, HitKind} ->
            setup_warm(Req, ContextTokens, RemainingTokens, HitKind, Data);
        cold ->
            setup_cold(Req, Data)
    end.

setup_warm(Req, ContextTokens, RemainingTokens, HitKind, Data) ->
    ok = warm_restore_primer(Req#req.seq_id, ContextTokens, Data),
    N = length(ContextTokens),
    case ContextTokens of
        [] ->
            Req#req{
                prefill_cursor = RemainingTokens,
                context_tokens = [],
                cache_hit_kind = HitKind,
                cache_hit_prefix_len = 0
            };
        _ ->
            Last = lists:last(ContextTokens),
            Kept = lists:sublist(ContextTokens, N - 1),
            Req#req{
                prefill_cursor = [Last | RemainingTokens],
                context_tokens = Kept,
                cache_hit_kind = HitKind,
                cache_hit_prefix_len = N
            }
    end.

%% Reset the seq's KV before a cold prefill so per_seq.next_pos
%% starts at 0, then split the prompt per the cold-save policy.
%% The trim slice goes into prefill_cursor; the remainder is held
%% in cold_save_remaining and rotated in by maybe_fire_cold_save
%% after the trim's prefill tick has fired the save.
setup_cold(Req, Data) ->
    ok = backend_seq_clear(Req#req.seq_id, Data),
    Tokens = Req#req.prompt_tokens,
    case erllama_cache_policy:cold_save_split(Tokens, Data#data.policy) of
        {trim, TrimmedPrefix, RemainingTokens} ->
            Req#req{
                prefill_cursor = TrimmedPrefix,
                cold_save_remaining = RemainingTokens,
                context_tokens = [],
                cache_hit_kind = cold,
                cache_hit_prefix_len = 0
            };
        no_save ->
            Req#req{
                prefill_cursor = Tokens,
                cold_save_remaining = undefined,
                context_tokens = [],
                cache_hit_kind = cold,
                cache_hit_prefix_len = 0
            }
    end.

%% Wipe seq's KV state so prefill starts at position 0. Used on the
%% cold path to defend against leftover cells from a prior
%% admission on this seq_id (single-tenant n_seq_max=1 reuses seq 0
%% across requests).
backend_seq_clear(SeqId, #data{backend = Mod, backend_state = S}) ->
    case erlang:function_exported(Mod, seq_rm, 2) of
        true ->
            _ = Mod:seq_rm(S, SeqId),
            ok;
        false ->
            case erlang:function_exported(Mod, seq_clear, 1) of
                true ->
                    _ = Mod:seq_clear(S),
                    ok;
                false ->
                    ok
            end
    end.

%% Warm-restore primer: load the cached KV into this seq via
%% kv_unpack, drop the last cell so the model layer can re-prefill
%% it to refresh logits. Mirrors the v0.2 prime_logits/3 flow but
%% per-seq.
%%
%% Both seq_rm_last arities take the seq's CURRENT length and
%% remove the cell at position N-1 (via kv_seq_rm(SeqId, N-1, -1)).
%% Passing N here, not 1: N is the cell-count after kv_unpack, and
%% the backend uses N-1 as the start position for the removal.
warm_restore_primer(_SeqId, [], _Data) ->
    ok;
warm_restore_primer(SeqId, ContextTokens, Data) ->
    N = length(ContextTokens),
    Mod = Data#data.backend,
    S = Data#data.backend_state,
    case erlang:function_exported(Mod, seq_rm_last, 3) of
        true ->
            _ = Mod:seq_rm_last(S, SeqId, N),
            ok;
        false ->
            case erlang:function_exported(Mod, seq_rm_last, 2) of
                true when SeqId =:= 0 ->
                    _ = Mod:seq_rm_last(S, N),
                    ok;
                _ ->
                    ok
            end
    end.

%% Stash a snapshot of the cache_hit + sampler info from the most
%% recent admission so the obs row and test accessors keep
%% reporting it after the request finishes.
snapshot_admission(Data, Req) ->
    Data#data{
        last_cache_hit_kind = Req#req.cache_hit_kind,
        last_cache_hit_prefix_len = Req#req.cache_hit_prefix_len,
        last_sampler_cfg = Req#req.last_sampler_cfg,
        last_sampler_ref = Req#req.sampler_ref
    }.

schedule_tick() ->
    gen_statem:cast(self(), tick).

%% Report `prefilling` if any in-flight seq still has a non-empty
%% prefill_cursor; otherwise `generating`. Empty req_table is only
%% transient between ticks.
dominant_phase(#data{req_table = T}) ->
    Reqs = maps:values(T),
    case Reqs of
        [] ->
            generating;
        _ ->
            HasPrefilling = lists:any(
                fun(R) -> R#req.prefill_cursor =/= undefined end,
                Reqs
            ),
            case HasPrefilling of
                true -> prefilling;
                false -> generating
            end
    end.

put_req(Data, Req) ->
    Data#data{req_table = maps:put(Req#req.seq_id, Req, Data#data.req_table)}.

remove_req(Data, SeqId) ->
    Data#data{req_table = maps:remove(SeqId, Data#data.req_table)}.

%% =============================================================================
%% Common event handler
%% =============================================================================

handle_common(_State, cast, {cancel, Ref}, Data) ->
    %% Walk req_table to find the seq carrying this request_ref and
    %% mark it cancel_pending. Stale or unknown refs are silently
    %% ignored, matching the v0.1 behaviour.
    case find_req_by_ref(Data, Ref) of
        {ok, Req} ->
            Req1 = Req#req{cancel_pending = true},
            NewData = put_req(Data, Req1),
            {keep_state, NewData};
        not_found ->
            {keep_state, Data}
    end;
handle_common(_State, {call, From}, evict, Data) ->
    {keep_state, fire_save_for_reason(evict, Data), [{reply, From, ok}]};
handle_common(_State, {call, From}, shutdown, Data) ->
    {keep_state, fire_save_for_reason(shutdown, Data), [{reply, From, ok}]};
handle_common(State, {call, From}, model_info, Data) ->
    reply(From, build_model_info(State, Data), Data);
handle_common(_State, {call, From}, {tokenize, Text}, Data) ->
    reply(From, wrap_ok(backend_call(Data, tokenize, [Text])), Data);
handle_common(_State, {call, From}, {detokenize, Tokens}, Data) ->
    reply(From, wrap_ok(backend_call(Data, detokenize, [Tokens])), Data);
handle_common(_State, {call, From}, {apply_chat_template, Request}, Data) ->
    reply(From, optional_backend_call(Data, apply_chat_template, [Request]), Data);
handle_common(_State, {call, From}, {embed, Tokens}, Data) ->
    reply(From, optional_backend_call(Data, embed, [Tokens]), Data);
handle_common(State, {call, From}, {load_adapter, Path}, Data) ->
    handle_load_adapter(State, From, Path, Data);
handle_common(State, {call, From}, {unload_adapter, Handle}, Data) ->
    handle_unload_adapter(State, From, Handle, Data);
handle_common(State, {call, From}, {set_adapter_scale, Handle, Scale}, Data) ->
    handle_set_adapter_scale(State, From, Handle, Scale, Data);
handle_common(_State, {call, From}, list_adapters, Data) ->
    Listing = [
        #{handle => H, scale => Scale}
     || #{handle := H, scale := Scale} <- Data#data.adapters
    ],
    reply(From, Listing, Data);
handle_common(_State, {call, From}, {verify, _, _, _}, Data) ->
    %% verify mutates context state; reject from any non-idle
    %% state so a concurrent infer's KV view stays consistent.
    {keep_state, Data, [{reply, From, {error, busy}}]};
handle_common(_State, {call, From}, cache_key_meta, Data) ->
    %% Effective fingerprint reflects the model's current LoRA
    %% composition. Using the base #data.fingerprint here would
    %% mis-key cache lookups whenever an adapter is attached.
    Meta = #{
        fingerprint => Data#data.effective_fp,
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash
    },
    reply(From, Meta, Data);
handle_common(_State, _EventType, _EventContent, Data) ->
    {keep_state, Data}.

handle_load_adapter(_State, From, Path, Data) ->
    Loaded =
        case load_adapter_impl(Path, Data) of
            {ok, Handle, Data1} -> {ok, Data1, {ok, Handle}};
            {error, _} = E -> E
        end,
    case Loaded of
        {ok, NewData, Reply} -> {keep_state, NewData, [{reply, From, Reply}]};
        {error, _} = E2 -> {keep_state, Data, [{reply, From, E2}]}
    end.

handle_unload_adapter(_State, From, Handle, Data) ->
    reply_with_data_update(From, Data, unload_adapter_impl(Handle, Data), ok).

handle_set_adapter_scale(_State, From, Handle, Scale, Data) ->
    reply_with_data_update(
        From, Data, set_adapter_scale_impl(Handle, Scale, Data), ok
    ).

%% Helper: convert an `impl` function's `{ok, Data1} | {error, _}`
%% return into a `keep_state` transition that replies to From either
%% with OkReply (on success) or with the error tuple verbatim.
reply_with_data_update(From, _OldData, {ok, Data1}, OkReply) ->
    {keep_state, Data1, [{reply, From, OkReply}]};
reply_with_data_update(From, OldData, {error, _} = E, _OkReply) ->
    {keep_state, OldData, [{reply, From, E}]}.

%% LoRA mutation helpers. Each call recomputes effective_fp from the
%% updated adapter list and reapplies the full set on the backend. We
%% read the adapter file once to derive its sha256 so the cache key
%% can fold it in deterministically.
load_adapter_impl(Path, Data) ->
    Mod = Data#data.backend,
    case erlang:function_exported(Mod, load_adapter, 2) of
        false -> {error, not_supported};
        true -> load_adapter_step1(Path, Data)
    end.

load_adapter_step1(Path, Data) ->
    case adapter_sha256(Path) of
        {ok, Sha} -> load_adapter_step2(Path, Sha, Data);
        {error, _} = E -> E
    end.

load_adapter_step2(Path, Sha, Data) ->
    Mod = Data#data.backend,
    case Mod:load_adapter(Data#data.backend_state, Path) of
        {ok, Handle, S1} ->
            Entry = #{handle => Handle, sha => Sha, scale => 1.0},
            New = Data#data.adapters ++ [Entry],
            apply_and_recompute(
                Data#data{backend_state = S1, adapters = New}, Handle
            );
        {error, _} = E ->
            E
    end.

unload_adapter_impl(Handle, Data) ->
    case find_adapter(Handle, Data#data.adapters) of
        false ->
            %% Idempotent: unknown / already removed.
            {ok, Data};
        {value, _Entry, Rest} ->
            unload_adapter_step1(Handle, Rest, Data)
    end.

unload_adapter_step1(Handle, Rest, Data) ->
    Mod = Data#data.backend,
    case erlang:function_exported(Mod, unload_adapter, 2) of
        false ->
            {error, not_supported};
        true ->
            case Mod:unload_adapter(Data#data.backend_state, Handle) of
                {ok, S1} ->
                    apply_and_recompute(
                        Data#data{backend_state = S1, adapters = Rest}, ok
                    );
                {error, _} = E ->
                    E
            end
    end.

set_adapter_scale_impl(Handle, Scale, Data) ->
    case find_adapter(Handle, Data#data.adapters) of
        false ->
            {error, not_found};
        {value, Entry, Rest} ->
            Updated = Entry#{scale => Scale},
            Adapters1 = Rest ++ [Updated],
            Data1 = Data#data{adapters = Adapters1},
            apply_and_recompute(Data1, ok)
    end.

find_adapter(Handle, List) ->
    case [E || E <- List, maps:get(handle, E) =:= Handle] of
        [] -> false;
        [E | _] -> {value, E, [X || X <- List, maps:get(handle, X) =/= Handle]}
    end.

%% Reapply the adapter set on the backend (if it supports it) and
%% recompute the effective fingerprint. The Result is what the public
%% API hands back to the caller: either an `ok` marker or
%% `{ok, Handle}` for load.
apply_and_recompute(Data, Result) ->
    case apply_current_adapters(Data) of
        {ok, S1} ->
            finalize_recompute(Data#data{backend_state = S1}, Result);
        {error, _} = E ->
            E
    end.

apply_current_adapters(#data{backend = Mod, backend_state = S, adapters = A}) ->
    case erlang:function_exported(Mod, apply_adapters, 2) of
        true ->
            Pairs = [{maps:get(handle, X), maps:get(scale, X)} || X <- A],
            Mod:apply_adapters(S, Pairs);
        false ->
            {ok, S}
    end.

finalize_recompute(Data, Result) ->
    ShaScales = [
        {maps:get(sha, A), maps:get(scale, A)}
     || A <- Data#data.adapters
    ],
    NewFp = erllama_cache_key:effective_fingerprint(
        Data#data.fingerprint, ShaScales
    ),
    Data1 = Data#data{effective_fp = NewFp},
    case Result of
        ok -> {ok, Data1};
        Handle -> {ok, Handle, Data1}
    end.

adapter_sha256(Path) ->
    case file:read_file(Path) of
        {ok, Bin} -> {ok, crypto:hash(sha256, Bin)};
        {error, _} = E -> E
    end.

%% Helper: synchronous reply with no state change.
reply(From, Reply, Data) ->
    {keep_state, Data, [{reply, From, Reply}]}.

%% Wrap a backend tokenize/detokenize raw result in `{ok, _}` so the
%% public API surface stays uniform across backends.
wrap_ok({error, _} = E) -> E;
wrap_ok(Result) -> {ok, Result}.

%% Like backend_call/3, but for callbacks declared optional in the
%% behaviour. If the backend module does not export the function
%% returns `{error, not_supported}` instead of crashing.
optional_backend_call(#data{backend = Mod, backend_state = S}, Fn, Args) ->
    Arity = length(Args) + 1,
    case erlang:function_exported(Mod, Fn, Arity) of
        true -> apply(Mod, Fn, [S | Args]);
        false -> {error, not_supported}
    end.

build_model_info(State, Data) ->
    %% Mirror the obs-row semantics so `model_info/1` and the
    %% lock-free accessors (`erllama:last_cache_hit/1`, etc.) agree
    %% on what counts as "no admission yet". The obs row starts with
    %% `undefined` kind at init and is only mutated by
    %% lookup_or_resume on a real admission.
    LastHit =
        case erllama_inflight:obs_get(Data#data.model_id) of
            {_Id, _Phase, _Pending, undefined, _PrefixLen} -> undefined;
            {_Id, _Phase, _Pending, Kind, PrefixLen} -> #{kind => Kind, prefix_len => PrefixLen};
            undefined -> undefined
        end,
    #{
        id => Data#data.model_id,
        model_id => Data#data.model_id,
        pid => self(),
        status => State,
        %% Alias for `status`; matches the obs table vocabulary used
        %% by `erllama:phase/1` and the cluster router.
        phase => State,
        pending_len => length(Data#data.pending),
        last_cache_hit => LastHit,
        backend => Data#data.backend,
        context_size => Data#data.context_size,
        quant_type => Data#data.quant_type,
        quant_bits => Data#data.quant_bits,
        quant_tag => erllama_cache_key:quant_tag(
            Data#data.quant_type, Data#data.quant_bits
        ),
        tier => Data#data.tier,
        fingerprint => Data#data.fingerprint,
        loaded_at_monotonic => Data#data.loaded_at_monotonic,
        vram_estimate_b => Data#data.vram_estimate_b
    }.

%% =============================================================================
%% Internal: observability snapshot (per-model ETS row)
%% =============================================================================

%% Row shape: {ModelId, Phase, PendingLen, LastCacheHitKind, LastCacheHitPrefixLen}.
%% The initial row carries `undefined` for the last-hit kind so external
%% routers can distinguish "model has never admitted a request" from
%% "model's last admission was cold" — both are valid signals.
obs_install_initial(Data) ->
    Row = {Data#data.model_id, idle, 0, undefined, 0},
    _ = erllama_inflight:obs_put(Data#data.model_id, Row),
    ok.

obs_refresh(Phase, Data) ->
    _ = erllama_inflight:obs_put(Data#data.model_id, obs_row(Phase, Data)),
    ok.

obs_row(Phase, Data) ->
    {
        Data#data.model_id,
        Phase,
        length(Data#data.pending),
        Data#data.last_cache_hit_kind,
        Data#data.last_cache_hit_prefix_len
    }.

%% =============================================================================
%% Internal: step_tick driver
%% =============================================================================
%%
%% Each tick: walk req_table, build a co-batched op list (prefill
%% rows + decode rows), apply the per-tick batch budget, call
%% backend:step/2, apply results back into the req_table, finish
%% any req that hit its terminal condition, then re-arm or return
%% to idle.

step_tick(Data) ->
    ok = obs_refresh(running_phase(Data), Data),
    Data1 = honour_cancellations(Data),
    case build_op_list(Data1) of
        {[], FinishersFirst} ->
            %% Nothing to step (e.g. only finishing-marked reqs).
            %% Drive the finishers and return.
            tick_after_step(Data1, FinishersFirst, []);
        {OpList, FinishersFirst} ->
            case backend_call(Data1, step, [OpList]) of
                {ok, Results} ->
                    Pairs = pair_ops_with_results(OpList, Results),
                    Data2 = apply_step_results(Pairs, Data1),
                    tick_after_step(Data2, FinishersFirst, []);
                {error, _} = Err ->
                    %% A backend error mid-tick poisons every in-flight
                    %% request — we cannot attribute it to a single row.
                    %% Stop with an error to every live caller; the
                    %% supervisor restarts the gen_statem.
                    fail_all_requests(Data1, Err),
                    {stop, {step_failed, Err}, Data1}
            end
    end.

%% Pair each op with its result by matching on seq_id. The backend
%% returns results in op-list order but we tolerate reordering by
%% looking up each op's result via its seq_id; the (seq_id, op)
%% pair is unique within a tick.
pair_ops_with_results(OpList, Results) ->
    ResultBySeq = maps:from_list(Results),
    [{Op, maps:get(SeqId, ResultBySeq)} || {SeqId, _} = Op <- OpList].

%% After a step (or a no-op tick), walk finishing-marked reqs,
%% emit replies/stream-done, free samplers + seqs, dispatch from
%% the pending FIFO if any seq slot freed up, then re-arm or idle.
tick_after_step(Data, PreFinishers, Actions0) ->
    %% Mark requests that hit terminal conditions in this tick.
    Data1 = mark_terminal(Data),
    {Data2, Actions1} = finish_marked_reqs(Data1, Actions0),
    {Data3, Actions2} = dispatch_pending_admits(Data2, Actions1),
    %% Pre-finishers were marked before the step (e.g. cancel hit). They
    %% are already drained inside finish_marked_reqs, so PreFinishers
    %% is just informational.
    _ = PreFinishers,
    case map_size(Data3#data.req_table) of
        0 ->
            ok = obs_refresh(idle, Data3),
            {next_state, idle, Data3, Actions2};
        _ ->
            schedule_tick(),
            {keep_state, Data3, Actions2}
    end.

%% Honour any cancel_pending flags raised by the cancel cast since
%% the previous tick: mark those reqs as finishing so the finisher
%% pass picks them up.
honour_cancellations(Data) ->
    Reqs = maps:values(Data#data.req_table),
    lists:foldl(
        fun
            (#req{cancel_pending = true} = R, Acc) ->
                put_req(Acc, R#req{finishing = true});
            (_R, Acc) ->
                Acc
        end,
        Data,
        Reqs
    ).

%% Build the op list for this tick. Returns {OpList, FinishersFirst}.
%% FinishersFirst is requests that have already hit a terminal
%% condition (response_target met, eog flag, cancel) and don't
%% participate in this tick — they're finalised in the post-step
%% finisher walk.
build_op_list(Data) ->
    Reqs = maps:values(Data#data.req_table),
    Active = [R || R <- Reqs, not R#req.finishing],
    {DecodeOps, PrefillOps} = lists:foldr(
        fun(R, {Decodes, Prefills}) ->
            case R#req.prefill_cursor of
                undefined ->
                    case R#req.sampler_ref of
                        undefined ->
                            %% prefill_only completed prefill — it's a
                            %% finisher this tick.
                            {Decodes, Prefills};
                        Ref ->
                            {[{R#req.seq_id, {decode, Ref}} | Decodes], Prefills}
                    end;
                [] ->
                    {Decodes, Prefills};
                Slice ->
                    {Decodes, [{R#req.seq_id, {prefill, Slice}} | Prefills]}
            end
        end,
        {[], []},
        Active
    ),
    Ops = apply_batch_budget(DecodeOps ++ PrefillOps, Data),
    {Ops, []}.

%% Bound the op list to total_batch_budget. Decode rows (each 1
%% token) are kept whole; prefill rows are sliced head-first.
%% Each prefill row is additionally capped at `prefill_chunk_size`
%% so a long prompt doesn't monopolise the batch.
apply_batch_budget(Ops, Data) ->
    Budget = Data#data.total_batch_budget,
    ChunkSize = maps:get(prefill_chunk_size, Data#data.policy, infinity),
    DecCount = length([{S, O} || {S, {decode, _}} = {S, O} <- Ops, _ <- [1]]),
    case DecCount >= Budget of
        true ->
            %% n_batch is smaller than the number of in-flight
            %% decoders. Operator misconfiguration; surface as an
            %% empty-tick error.
            erlang:error({n_batch_too_small_for_n_seq_max, Budget, DecCount});
        false ->
            slice_prefills(Ops, Budget - DecCount, ChunkSize)
    end.

slice_prefills(Ops, Remaining, ChunkSize) ->
    slice_prefills(Ops, Remaining, ChunkSize, []).

slice_prefills([], _Remaining, _ChunkSize, Acc) ->
    lists:reverse(Acc);
slice_prefills([{SeqId, {decode, R}} | T], Remaining, ChunkSize, Acc) ->
    slice_prefills(T, Remaining, ChunkSize, [{SeqId, {decode, R}} | Acc]);
slice_prefills([{_SeqId, {prefill, _Tokens}} | T], Remaining, ChunkSize, Acc) when
    Remaining =< 0
->
    %% Budget exhausted: drop this prefill from the current tick;
    %% it'll resume next tick. apply_step_results doesn't see this
    %% op so the cursor stays where it was.
    slice_prefills(T, 0, ChunkSize, Acc);
slice_prefills([{SeqId, {prefill, Tokens}} | T], Remaining, ChunkSize, Acc) ->
    Cap = prefill_slice_cap(length(Tokens), Remaining, ChunkSize),
    case Cap >= length(Tokens) of
        true ->
            slice_prefills(
                T, Remaining - length(Tokens), ChunkSize, [{SeqId, {prefill, Tokens}} | Acc]
            );
        false ->
            Slice = lists:sublist(Tokens, Cap),
            slice_prefills(T, Remaining - Cap, ChunkSize, [{SeqId, {prefill, Slice}} | Acc])
    end.

prefill_slice_cap(N, Remaining, infinity) ->
    min(N, Remaining);
prefill_slice_cap(N, Remaining, ChunkSize) when is_integer(ChunkSize), ChunkSize > 0 ->
    min(N, min(Remaining, ChunkSize)).

%% Apply step result list back into the req_table. Each entry is
%% the op the scheduler sent paired with the backend's result for
%% that op. Prefill results advance the cursor by the actual slice
%% length the slicer kept; decode results append the sampled token.
apply_step_results([], Data) ->
    Data;
apply_step_results([{{SeqId, {prefill, Slice}}, prefilled} | T], Data) ->
    Req = maps:get(SeqId, Data#data.req_table),
    SentLen = length(Slice),
    Cursor0 =
        case Req#req.prefill_cursor of
            undefined -> [];
            L -> L
        end,
    Remaining = lists:nthtail(SentLen, Cursor0),
    NewContext = Req#req.context_tokens ++ Slice,
    NewCursor =
        case Remaining of
            [] -> undefined;
            _ -> Remaining
        end,
    Req1 = Req#req{
        prefill_cursor = NewCursor,
        context_tokens = NewContext,
        generation_started_at =
            case Req#req.generation_started_at of
                undefined -> erlang:monotonic_time(millisecond);
                G -> G
            end
    },
    %% Cold save fires only on the transition from cursor non-empty
    %% to cursor empty: at that point the seq's KV holds exactly the
    %% trimmed prefix that cold_save_split selected, and the
    %% remainder rotates back into prefill_cursor.
    Req2 =
        case NewCursor of
            undefined -> maybe_fire_cold_save(Req1#req{last_save_at = 0}, Data);
            _ -> Req1
        end,
    apply_step_results(T, put_req(Data, Req2));
apply_step_results([{{SeqId, {decode, _}}, {token, Tok, EogFlag}} | T], Data) ->
    Req = maps:get(SeqId, Data#data.req_table),
    Req1 = req_append_token(Req, Tok),
    Req2 = req_stream_emit(Req1, Tok, Data),
    %% Mark terminal if eog, response_target reached, or a stop
    %% sequence fired. matched_stop is set inside req_stream_emit
    %% on a hit; check it here so the request is marked finishing
    %% at the same point the existing EOG/length paths are.
    GenLen = length(Req2#req.generated),
    Done =
        EogFlag =:= 1 orelse
            GenLen >= Req2#req.response_target orelse
            Req2#req.matched_stop =/= undefined,
    Req3 =
        case Done of
            true -> Req2#req{finishing = true};
            false -> maybe_fire_continued_for_req(Req2, Data)
        end,
    apply_step_results(T, put_req(Data, Req3));
apply_step_results([{{SeqId, {decode, S}}, {thinking_token, Tok}} | T], Data) ->
    Req = maps:get(SeqId, Data#data.req_table),
    case Req#req.thinking_capped of
        true ->
            %% Caller-side thinking budget was already hit. Route
            %% this token through the normal post-thinking pipeline
            %% so generation progresses; the backend can keep
            %% emitting thinking_token without surprises.
            apply_step_results(
                [{{SeqId, {decode, S}}, {token, Tok, 0}} | T], Data
            );
        false ->
            %% Thinking tokens are NOT appended to generated/context_tokens:
            %% they belong to a separate content track and the cache
            %% is keyed on the post-thinking output. The backend is
            %% responsible for any internal bookkeeping needed to
            %% keep its KV state consistent.
            Req1 = req_thinking_emit(Req, Tok, Data),
            apply_step_results(T, put_req(Data, Req1))
    end;
apply_step_results([{{SeqId, {decode, _}}, thinking_end} | T], Data) ->
    Req = maps:get(SeqId, Data#data.req_table),
    Req1 = req_thinking_end(Req, Data),
    apply_step_results(T, put_req(Data, Req1));
apply_step_results([{{SeqId, {decode, _}}, {tool_call_token, Tok}} | T], Data) ->
    Req = maps:get(SeqId, Data#data.req_table),
    Req1 = req_tool_call_emit(Req, Tok, Data),
    apply_step_results(T, put_req(Data, Req1));
apply_step_results([{{SeqId, {decode, _}}, tool_call_end} | T], Data) ->
    Req = maps:get(SeqId, Data#data.req_table),
    Req1 = req_tool_call_end(Req, Data),
    apply_step_results(T, put_req(Data, Req1)).

%% Append the token to both context_tokens and generated.
req_append_token(Req, Token) ->
    Req#req{
        context_tokens = Req#req.context_tokens ++ [Token],
        generated = Req#req.generated ++ [Token]
    }.

%% Stream a token to the caller (no-op for non-streaming modes
%% unless stop_sequences are active, in which case we still need to
%% accumulate detokenized bytes so an early match can stop the
%% request and trim the eventual standard-mode reply).
req_stream_emit(
    #req{stop_pattern = undefined, mode = streaming, caller_pid = Pid, request_ref = Ref} = Req,
    Token,
    Data
) ->
    case backend_call(Data, detokenize, [[Token]]) of
        Bin when is_binary(Bin), Bin =/= <<>> ->
            Pid ! {erllama_token, Ref, Bin};
        _ ->
            ok
    end,
    Pid ! {erllama_token_id, Ref, Token},
    Req;
req_stream_emit(#req{stop_pattern = undefined} = Req, _Token, _Data) ->
    Req;
req_stream_emit(Req, Token, Data) ->
    Chunk =
        case backend_call(Data, detokenize, [[Token]]) of
            B when is_binary(B) -> B;
            _ -> <<>>
        end,
    Buf = <<(Req#req.pending_text)/binary, Chunk/binary>>,
    case binary:match(Buf, Req#req.stop_pattern) of
        {Pos, Len} ->
            Match = binary:part(Buf, Pos, Len),
            Prefix = binary:part(Buf, 0, Pos),
            maybe_emit_stream(Req, Prefix, Token),
            Req#req{pending_text = <<>>, matched_stop = Match};
        nomatch ->
            case Req#req.mode of
                streaming ->
                    Hold = max(Req#req.stop_max_len - 1, 0),
                    Total = byte_size(Buf),
                    EmitN = max(Total - Hold, 0),
                    Safe = binary:part(Buf, 0, EmitN),
                    Tail = binary:part(Buf, EmitN, Total - EmitN),
                    maybe_emit_stream(Req, Safe, Token),
                    Req#req{pending_text = Tail};
                _ ->
                    Req#req{pending_text = Buf}
            end
    end.

maybe_emit_stream(
    #req{mode = streaming, caller_pid = Pid, request_ref = Ref}, Text, Token
) when
    is_pid(Pid), is_reference(Ref)
->
    case Text of
        <<>> -> ok;
        _ -> Pid ! {erllama_token, Ref, Text}
    end,
    Pid ! {erllama_token_id, Ref, Token},
    ok;
maybe_emit_stream(_Req, _Text, _Token) ->
    ok.

%% Thinking-phase token. Detokenises the new id and forwards it as
%% {erllama_token, Ref, {thinking_delta, Bin}} to streaming callers
%% that opted in via `thinking = enabled`. The bytes are also kept
%% on the #req so the backend's thinking_signature/2 fallback can
%% be derived from the accumulated text. A thinking_token arriving
%% on a non-streaming or thinking-disabled request is treated as a
%% backend bug -- the request is failed via erllama_error.
req_thinking_emit(#req{thinking = disabled} = Req, _Token, _Data) ->
    fail_thinking_disabled(Req),
    Req#req{finishing = true, errored = thinking_not_enabled};
req_thinking_emit(
    #req{mode = streaming, caller_pid = Pid, request_ref = Ref} = Req,
    Token,
    Data
) when
    is_pid(Pid), is_reference(Ref)
->
    Bin =
        case backend_call(Data, detokenize, [[Token]]) of
            B when is_binary(B) -> B;
            _ -> <<>>
        end,
    Req1 = Req#req{
        thinking_bytes = <<(Req#req.thinking_bytes)/binary, Bin/binary>>
    },
    case Bin of
        <<>> ->
            %% Empty detokenisation doesn't consume the budget and
            %% isn't observed by the caller.
            Req1;
        _ ->
            Pid ! {erllama_token, Ref, {thinking_delta, Bin}},
            Req2 = Req1#req{thinking_count = Req1#req.thinking_count + 1},
            maybe_cap_thinking(Req2, Data)
    end;
req_thinking_emit(Req, _Token, _Data) ->
    %% Non-streaming caller with thinking_token in flight: drop the
    %% delta but keep the bytes for the eventual signature so
    %% downstream consumers that just want stats still see a
    %% coherent thinking_end. Mostly a defensive branch.
    Req.

%% Caller-side thinking budget. When the count of delivered
%% `{thinking_delta, _}` payloads reaches the configured budget,
%% synthesise the `erllama_thinking_end` close immediately so the
%% caller sees a coherent thinking block; subsequent
%% `{thinking_token, _}` step results are routed through the
%% normal token pipeline by apply_step_results.
maybe_cap_thinking(#req{thinking_budget = undefined} = Req, _Data) ->
    Req;
maybe_cap_thinking(#req{thinking_count = N, thinking_budget = B} = Req, _Data) when
    N < B
->
    Req;
maybe_cap_thinking(
    #req{caller_pid = Pid, request_ref = Ref} = Req, Data
) when
    is_pid(Pid), is_reference(Ref)
->
    Sig = thinking_signature(Req, Data),
    Pid ! {erllama_thinking_end, Ref, Sig},
    Req#req{thinking_bytes = <<>>, thinking_capped = true};
maybe_cap_thinking(Req, _Data) ->
    Req#req{thinking_bytes = <<>>, thinking_capped = true}.

req_thinking_end(#req{thinking_capped = true} = Req, _Data) ->
    %% The caller-side budget already synthesised the close;
    %% suppress a late `thinking_end` from the backend so the
    %% caller doesn't see two close markers.
    Req;
req_thinking_end(
    #req{mode = streaming, caller_pid = Pid, request_ref = Ref} = Req,
    Data
) when
    is_pid(Pid), is_reference(Ref)
->
    Sig = thinking_signature(Req, Data),
    Pid ! {erllama_thinking_end, Ref, Sig},
    Req#req{thinking_bytes = <<>>};
req_thinking_end(Req, _Data) ->
    Req#req{thinking_bytes = <<>>}.

%% Tool-call token: detokenise and forward as
%% {erllama_token, Ref, {tool_call_delta, Bin}} to streaming
%% callers. Bytes accumulate on the #req so the closing
%% `tool_call_end` can deliver the full concatenation. Non-streaming
%% callers see no message but still accumulate, in case the result
%% map surface ever exposes tool calls.
req_tool_call_emit(
    #req{mode = streaming, caller_pid = Pid, request_ref = Ref} = Req,
    Token,
    Data
) when
    is_pid(Pid), is_reference(Ref)
->
    Bin =
        case backend_call(Data, detokenize, [[Token]]) of
            B when is_binary(B) -> B;
            _ -> <<>>
        end,
    case Bin of
        <<>> -> ok;
        _ -> Pid ! {erllama_token, Ref, {tool_call_delta, Bin}}
    end,
    Req#req{tool_call_bytes = <<(Req#req.tool_call_bytes)/binary, Bin/binary>>};
req_tool_call_emit(Req, Token, Data) ->
    Bin =
        case backend_call(Data, detokenize, [[Token]]) of
            B when is_binary(B) -> B;
            _ -> <<>>
        end,
    Req#req{tool_call_bytes = <<(Req#req.tool_call_bytes)/binary, Bin/binary>>}.

%% Close the current tool-call span. The full concatenated bytes go
%% out in one message so the downstream's exact-replay map can store
%% them without re-buffering chunks. Then reset the buffer for the
%% next span on this request.
req_tool_call_end(
    #req{mode = streaming, caller_pid = Pid, request_ref = Ref} = Req,
    _Data
) when
    is_pid(Pid), is_reference(Ref)
->
    Pid ! {erllama_tool_call_end, Ref, Req#req.tool_call_bytes},
    Req#req{tool_call_bytes = <<>>};
req_tool_call_end(Req, _Data) ->
    Req#req{tool_call_bytes = <<>>}.

%% Resolve the integrity signature for the just-closed thinking
%% block. Calls the backend's optional `thinking_signature/3` with
%% the accumulated thinking bytes; otherwise falls back to `<<>>`
%% so the downstream omits `signature_delta` from its SSE output.
thinking_signature(#req{seq_id = SeqId, thinking_bytes = Bytes}, #data{
    backend = Mod, backend_state = S
}) ->
    case erlang:function_exported(Mod, thinking_signature, 3) of
        true ->
            case Mod:thinking_signature(S, SeqId, Bytes) of
                Bin when is_binary(Bin) -> Bin;
                _ -> <<>>
            end;
        false ->
            <<>>
    end.

%% Surface the contract violation to the caller and shut the
%% request down. A thinking_token arriving on a request that did not
%% set `thinking = enabled` indicates a backend that's ignoring the
%% per-request flag.
fail_thinking_disabled(#req{
    mode = streaming,
    caller_pid = Pid,
    request_ref = Ref
}) when
    is_pid(Pid), is_reference(Ref)
->
    erllama_inflight:unregister(Ref),
    Pid ! {erllama_error, Ref, thinking_not_enabled},
    ok;
fail_thinking_disabled(_Req) ->
    ok.

%% Mark any req whose generated count >= response_target as
%% finishing. Called after apply_step_results so we don't mutate
%% the table mid-iteration.
mark_terminal(Data) ->
    Reqs = maps:values(Data#data.req_table),
    lists:foldl(
        fun(R, Acc) ->
            case R#req.finishing of
                true ->
                    Acc;
                false when
                    R#req.prefill_cursor =:= undefined,
                    R#req.mode =:= prefill_only
                ->
                    %% prefill_only finishes as soon as prefill is done.
                    put_req(Acc, R#req{finishing = true});
                false ->
                    Acc
            end
        end,
        Data,
        Reqs
    ).

%% Cold-save firing between the trim-prefill tick and the remainder
%% tick. At this point the seq's KV holds exactly the trimmed
%% prefix — kv_pack captures that state. The remainder is rotated
%% into prefill_cursor so the next tick continues the prefill. With
%% cold_save_remaining = undefined nothing fires (no_save policy,
%% prefill_only mode, or warm path).
maybe_fire_cold_save(Req = #req{cold_save_remaining = undefined}, _Data) ->
    Req;
maybe_fire_cold_save(
    Req = #req{cold_save_remaining = Remaining, context_tokens = Trimmed},
    Data
) ->
    Req0 = fire_save_for_tokens(cold, Trimmed, Req, Data),
    NextCursor =
        case Remaining of
            [] -> undefined;
            _ -> Remaining
        end,
    Req0#req{
        cold_save_remaining = undefined,
        last_save_at = length(Trimmed),
        prefill_cursor = NextCursor
    }.

%% Continued-save: fire every continued_interval tokens since the
%% last save fired.
maybe_fire_continued_for_req(Req, Data) ->
    LiveCount = length(Req#req.context_tokens),
    Should = erllama_cache_policy:should_continued_save(
        LiveCount, Req#req.last_save_at, Data#data.policy
    ),
    case Should of
        true ->
            Req0 = fire_save_for_tokens(continued, Req#req.context_tokens, Req, Data),
            Req0#req{last_save_at = LiveCount};
        false ->
            Req
    end.

%% Walk all #req with finishing=true; for each, fire its finish
%% save, build the reply / stream-done, free its sampler, free its
%% seq, push the seq_id back, and remove it from req_table.
finish_marked_reqs(Data, Actions) ->
    Reqs = maps:values(Data#data.req_table),
    Finishers = [R || R <- Reqs, R#req.finishing],
    lists:foldl(
        fun(R, {AccData, AccActions}) ->
            FinishReason =
                case R#req.cancel_pending of
                    true ->
                        cancelled;
                    false when R#req.matched_stop =/= undefined ->
                        stop;
                    false ->
                        case length(R#req.generated) >= R#req.response_target of
                            true -> length;
                            false -> stop
                        end
                end,
            finish_req(R, FinishReason, AccData, AccActions)
        end,
        {Data, Actions},
        Finishers
    ).

finish_req(Req, FinishReason, Data, Actions) ->
    FinishKey = finish_key_or_undefined(
        fire_finish_save_for_req(Req#req.context_tokens, Req, Data)
    ),
    Req1 =
        case FinishKey of
            undefined -> Req;
            _ -> bump_cache_delta(Req, Req#req.context_tokens)
        end,
    Stats = build_stats_for_req(
        FinishReason, Req1#req.cancel_pending, FinishKey, Req1
    ),
    Action = finish_action(Req1, FinishReason, FinishKey, Stats, Data),
    %% Release the sampler and free the seq's KV before returning
    %% the seq_id to the pool. The gen_statem holds the context
    %% mutex so no concurrent reader of this seq_id is possible.
    _ = release_sampler(Req1, Data),
    _ = release_seq(Req1#req.seq_id, Data),
    Data1 = remove_req(Data, Req1#req.seq_id),
    Data2 = Data1#data{idle_seq_ids = [Req1#req.seq_id | Data1#data.idle_seq_ids]},
    {Data2, Actions ++ Action}.

finish_action(#req{mode = standard, caller = From} = Req, FinishReason, FinishKey, Stats, Data) ->
    Reply0 = backend_call(Data, detokenize, [Req#req.generated]),
    Reply = trim_at_match(Reply0, Req#req.matched_stop),
    Result0 = #{
        reply => Reply,
        generated => Req#req.generated,
        context_tokens => Req#req.context_tokens,
        committed_tokens => length(Req#req.context_tokens),
        finish_key => FinishKey,
        cache_hit_kind => Req#req.cache_hit_kind,
        finish_reason => FinishReason,
        cache_delta => cache_delta_for(Req),
        stats => Stats
    },
    Result = maybe_add_stop_sequence(Result0, Req#req.matched_stop),
    [{reply, From, {ok, Result}}];
finish_action(#req{mode = streaming, errored = E} = Req, _FinishReason, _FinishKey, Stats, _Data) ->
    case E of
        undefined ->
            flush_pending_text(Req),
            send_done_for_req(Req, Stats);
        _ ->
            %% Error was already surfaced; skip the done message so
            %% the caller doesn't observe both for the same request.
            ok
    end,
    [];
finish_action(
    #req{mode = prefill_only, caller = From} = Req, _FinishReason, FinishKey, _Stats, _Data
) ->
    Result = #{
        context_tokens => Req#req.context_tokens,
        committed_tokens => length(Req#req.context_tokens),
        finish_key => FinishKey,
        cache_hit_kind => Req#req.cache_hit_kind,
        cache_delta => cache_delta_for(Req)
    },
    [{reply, From, {ok, Result}}].

send_done_for_req(#req{request_ref = Ref, caller_pid = Pid}, Stats) when
    is_pid(Pid), is_reference(Ref)
->
    erllama_inflight:unregister(Ref),
    Pid ! {erllama_done, Ref, Stats},
    ok;
send_done_for_req(_Req, _Stats) ->
    ok.

%% Fail every in-flight request with the same error. Used when a
%% backend:step/2 call returns an error that cannot be attributed
%% to a single row (e.g. llama_decode exception).
fail_all_requests(Data, Err) ->
    Reqs = maps:values(Data#data.req_table),
    lists:foreach(
        fun(R) ->
            case R#req.mode of
                streaming ->
                    case {R#req.request_ref, R#req.caller_pid} of
                        {Ref, Pid} when is_pid(Pid), is_reference(Ref) ->
                            erllama_inflight:unregister(Ref),
                            Pid ! {erllama_error, Ref, Err};
                        _ ->
                            ok
                    end;
                _ ->
                    %% sync callers have to receive a reply or they
                    %% deadlock; gen_statem will fail the call on
                    %% stop, surfacing {error, _} to them.
                    ok
            end,
            _ = release_sampler(R, Data),
            _ = release_seq(R#req.seq_id, Data)
        end,
        Reqs
    ),
    ok.

release_sampler(#req{sampler_ref = undefined}, _Data) ->
    ok;
release_sampler(#req{sampler_ref = Ref}, #data{backend = Mod}) ->
    case erlang:function_exported(Mod, sampler_free, 1) of
        true ->
            _ = Mod:sampler_free(Ref),
            ok;
        false ->
            ok
    end.

release_seq(SeqId, #data{backend = Mod, backend_state = S}) ->
    case erlang:function_exported(Mod, seq_rm, 2) of
        true ->
            _ = Mod:seq_rm(S, SeqId),
            ok;
        false ->
            ok
    end.

%% Dispatch one queued admit if a seq_id became free. Repeats while
%% slots remain and pending is non-empty.
dispatch_pending_admits(Data, Actions) ->
    case {Data#data.idle_seq_ids, Data#data.pending} of
        {[], _} ->
            {Data, Actions};
        {_, []} ->
            {Data, Actions};
        {[SeqId | RestIds], [Head | RestPend]} ->
            Data1 = Data#data{
                idle_seq_ids = RestIds,
                pending = RestPend
            },
            case start_request(Head, SeqId, Data1) of
                {ok, Data2, MoreActions} ->
                    dispatch_pending_admits(Data2, Actions ++ MoreActions);
                {error, Reason, From} ->
                    Data2 = Data1#data{idle_seq_ids = [SeqId | RestIds]},
                    dispatch_pending_admits(
                        Data2,
                        Actions ++ [{reply, From, {error, Reason}}]
                    )
            end
    end.

enqueue(Item, Data) ->
    Data#data{pending = Data#data.pending ++ [Item]}.

finish_key_or_undefined({ok, Key}) -> Key;
finish_key_or_undefined(skipped) -> undefined.

%% Phase reported on the obs row while in `running`. Mirrors
%% dominant_phase/1 but renamed to keep the obs callsite explicit.
running_phase(Data) ->
    dominant_phase(Data).

%% Find a request in req_table whose request_ref matches. Used by
%% cancel/2. O(n) in the number of in-flight reqs (typically ≤
%% n_seq_max).
find_req_by_ref(Data, Ref) ->
    Found = lists:filter(
        fun(R) -> R#req.request_ref =:= Ref end,
        maps:values(Data#data.req_table)
    ),
    case Found of
        [R | _] -> {ok, R};
        [] -> not_found
    end.

%% Build the sampler-config subset from request opts. Only the keys
%% the sampler chain cares about; everything else (response_tokens,
%% parent_key) is dropped.
-define(SAMPLER_KEYS, [
    grammar,
    repetition_penalty,
    top_k,
    top_p,
    min_p,
    temperature,
    seed
]).

sampler_cfg_from(Opts) ->
    maps:with(?SAMPLER_KEYS, Opts).

%% Build a per-request sampler chain via the new behaviour callback
%% (`backend:sampler_new/2`). Returns the ref + the cfg that was
%% sent to the backend; the caller stashes both on the #req. Falls
%% back to {ok, undefined, Cfg, Data} if the backend doesn't
%% implement sampler_new (legacy backends; they also won't implement
%% step/2 so the request will fail on first tick — only kept for
%% the error story).
sampler_for(Opts, Data = #data{backend = Mod, backend_state = S}) ->
    Cfg = sampler_cfg_from(Opts),
    case erlang:function_exported(Mod, sampler_new, 2) of
        true ->
            case Mod:sampler_new(S, Cfg) of
                {ok, Ref} -> {ok, Ref, Cfg, Data};
                {error, _} = E -> E
            end;
        false ->
            {ok, undefined, Cfg, Data}
    end.

%% Parse `stop_sequences` from an Opts/Params map. Non-binary or
%% empty-binary entries are dropped silently; an entirely missing or
%% empty list yields {[], undefined, 0} which disables the scanner.
stop_sequences_from(Opts) ->
    Raw = maps:get(stop_sequences, Opts, []),
    Stops = [S || S <- Raw, is_binary(S), S =/= <<>>],
    case Stops of
        [] ->
            {[], undefined, 0};
        _ ->
            Pat = binary:compile_pattern(Stops),
            Max = lists:max([byte_size(S) || S <- Stops]),
            {Stops, Pat, Max}
    end.

%% Caller's thinking opt-in. Anything other than the atom `enabled`
%% (including a missing key, `false`, or the atom `disabled`) keeps
%% the request out of the thinking pipeline.
thinking_from(Opts) ->
    case maps:get(thinking, Opts, disabled) of
        enabled -> enabled;
        _ -> disabled
    end.

%% Caller's thinking-phase budget. Non-positive integers and any
%% non-integer (including a missing key) yield `undefined`, which
%% means "no cap".
thinking_budget_from(Opts) ->
    case maps:get(thinking_budget_tokens, Opts, undefined) of
        N when is_integer(N), N > 0 -> N;
        _ -> undefined
    end.

%% Per-#req stats. Mirrors the v0.2 build_stats/4 but reads from the
%% request record instead of #data.
build_stats_for_req(FinishReason, Cancelled, FinishKey, Req) ->
    Now = erlang:monotonic_time(millisecond),
    PrefillStart = Req#req.prefill_started_at,
    GenStart = Req#req.generation_started_at,
    PrefillMs =
        case {PrefillStart, GenStart} of
            {undefined, _} -> 0;
            {_, undefined} -> max(0, Now - PrefillStart);
            _ -> max(0, GenStart - PrefillStart)
        end,
    GenMs =
        case GenStart of
            undefined -> 0;
            _ -> max(0, Now - GenStart)
        end,
    Stats0 = #{
        prompt_tokens => length(Req#req.prompt_tokens),
        completion_tokens => length(Req#req.generated),
        prefill_ms => PrefillMs,
        generation_ms => GenMs,
        cache_hit_kind => Req#req.cache_hit_kind,
        finish_reason => FinishReason,
        cancelled => Cancelled,
        finish_key => FinishKey,
        committed_tokens => length(Req#req.context_tokens),
        cache_delta => cache_delta_for(Req)
    },
    maybe_add_stop_sequence(Stats0, Req#req.matched_stop).

%% Anthropic-style per-request cache breakdown. `read` counts tokens
%% served from the warm prefix at admission; `created` counts tokens
%% this request added to the cache beyond that prefix (largest save
%% contribution seen across cold / continued / finish / evict /
%% shutdown). Both default to 0 when no relevant cache activity
%% happened.
cache_delta_for(Req) ->
    #{
        read => Req#req.cache_hit_prefix_len,
        created => Req#req.cache_delta_created
    }.

%% Additive: include `stop_sequence` only when a stop string fired.
maybe_add_stop_sequence(Map, undefined) ->
    Map;
maybe_add_stop_sequence(Map, Match) when is_binary(Match) ->
    Map#{stop_sequence => Match}.

%% Standard-mode reply trim: cut the bulk-detokenized reply at the
%% first occurrence of the matched stop string. The bulk detokenize
%% can differ slightly from per-token concat for some tokenizers, so
%% we re-scan rather than reuse pending_text.
trim_at_match(Reply, undefined) ->
    Reply;
trim_at_match(Reply, Match) when is_binary(Reply), is_binary(Match) ->
    case binary:match(Reply, Match) of
        {Pos, _Len} -> binary:part(Reply, 0, Pos);
        nomatch -> Reply
    end.

%% Flush any text held back in pending_text as one final
%% `{erllama_token, _, _}` message. Only relevant for streaming
%% requests with stop_sequences active and no match: the trailing
%% (max_stop_len - 1) bytes were withheld in case they were the
%% prefix of a future match; at the terminal step they are safe to
%% emit. When matched_stop is set, pending_text is already cleared
%% so this is a no-op.
flush_pending_text(#req{
    mode = streaming,
    caller_pid = Pid,
    request_ref = Ref,
    pending_text = Tail
}) when
    is_pid(Pid), is_reference(Ref), is_binary(Tail), Tail =/= <<>>
->
    Pid ! {erllama_token, Ref, Tail},
    ok;
flush_pending_text(_Req) ->
    ok.

%% =============================================================================
%% Internal: cache integration
%% =============================================================================

%% Multi-seq variant: takes the #req that's being admitted so the
%% fingerprint and kv_unpack/seq_rm calls target the correct seq.
lookup_or_resume(PromptTokens, ParentKey, Req, Data) ->
    Key = make_key(PromptTokens, Req, Data),
    case pin_and_load(Key, Req#req.seq_id, Data) of
        {ok, ContextTokens} ->
            erllama_cache_counters:incr(?C_HITS_EXACT),
            {warm, ContextTokens, [], exact};
        miss when ParentKey =/= undefined ->
            try_session_resume(PromptTokens, ParentKey, Req, Data);
        miss ->
            try_longest_prefix(PromptTokens, Req, Data)
    end.

%% Stateless callers (HTTP front-end, agent loops that resend the
%% full conversation each turn) don't have a parent_key to thread.
%% Walk back through the prompt by stride and pick the longest
%% cached prefix; fall through to cold if nothing matches.
try_longest_prefix(PromptTokens, Req, Data) ->
    KeyMeta = #{
        fingerprint => request_fp(Req, Data),
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash
    },
    Stride = maps:get(boundary_align_tokens, Data#data.policy, 2048),
    Min = maps:get(min_tokens, Data#data.policy, 512),
    case erllama_cache_meta_srv:lookup_longest_prefix(KeyMeta, PromptTokens, Stride, Min) of
        {ok, PrefixLen, Row} ->
            resume_at_prefix(
                element(?POS_KEY, Row), PrefixLen, PromptTokens, Req, Data, partial
            );
        miss ->
            erllama_cache_counters:incr(?C_MISSES),
            cold
    end.

%% Pin + load the row, then verify the tokens really are the first
%% PrefixLen of PromptTokens. The key is sha256 of the tokens, so a
%% hit implies equality, but we belt-and-braces it here.
resume_at_prefix(Key, PrefixLen, PromptTokens, Req, Data, HitKind) ->
    case pin_and_load(Key, Req#req.seq_id, Data) of
        {ok, ParentTokens} when length(ParentTokens) =:= PrefixLen ->
            case is_strict_prefix(ParentTokens, PromptTokens) of
                true ->
                    Remaining = lists:nthtail(PrefixLen, PromptTokens),
                    erllama_cache_counters:incr(?C_HITS_LONGEST_PREFIX),
                    {warm, ParentTokens, Remaining, HitKind};
                false ->
                    erllama_cache_counters:incr(?C_MISSES),
                    cold
            end;
        _ ->
            erllama_cache_counters:incr(?C_MISSES),
            cold
    end.

%% Note: the resume hit counter is bumped inside `try_session_resume`
%% only on a verified strict-prefix match.

try_session_resume(PromptTokens, ParentKey, Req, Data) ->
    Wait = maps:get(session_resume_wait_ms, Data#data.policy, 500),
    %% First wait for the row to publish (the previous turn's
    %% finish-save may still be in flight), then pin via checkout.
    case erllama_cache_meta_srv:lookup_exact_or_wait(ParentKey, Wait) of
        {ok, _Row} ->
            case pin_and_load(ParentKey, Req#req.seq_id, Data) of
                {ok, ParentTokens} ->
                    case is_strict_prefix(ParentTokens, PromptTokens) of
                        true ->
                            Remaining = lists:nthtail(length(ParentTokens), PromptTokens),
                            erllama_cache_counters:incr(?C_HITS_RESUME),
                            {warm, ParentTokens, Remaining, partial};
                        false ->
                            erllama_cache_counters:incr(?C_MISSES),
                            cold
                    end;
                miss ->
                    erllama_cache_counters:incr(?C_MISSES),
                    cold
            end;
        miss ->
            erllama_cache_counters:incr(?C_MISSES),
            cold
    end.

%% checkout the row, load + unpack the payload under the pin (into
%% the request's seq_id), then checkin. Returns the row's stored
%% token list on success or `miss` if the row was evicted between
%% the prior lookup and our checkout.
pin_and_load(Key, SeqId, Data) ->
    T0 = erlang:monotonic_time(nanosecond),
    Result =
        case erllama_cache_meta_srv:checkout(Key, self()) of
            {ok, HolderRef, Tier, Loc, _Header, TokensBin} ->
                try
                    Bin = load_payload(Tier, Loc, Key, Data),
                    case Bin of
                        <<>> ->
                            miss;
                        _ ->
                            ok = backend_kv_unpack(Bin, SeqId, Data),
                            Tokens =
                                case TokensBin of
                                    undefined -> [];
                                    _ -> erllama_cache_key:decode_tokens(TokensBin)
                                end,
                            {ok, Tokens}
                    end
                after
                    ok = erllama_cache_meta_srv:checkin(HolderRef)
                end;
            {error, busy} ->
                miss;
            miss ->
                miss
        end,
    Elapsed = erlang:monotonic_time(nanosecond) - T0,
    erllama_cache_counters:add(?C_LOAD_TOTAL_NS, max(Elapsed, 0)),
    Result.

load_payload(ram, _Loc, Key, _Data) ->
    case erllama_cache_ram:load(Key) of
        {ok, B} -> B;
        miss -> <<>>
    end;
load_payload(_Tier, _Loc, Key, Data) ->
    case erllama_cache_disk_srv:load(Data#data.tier_srv, Key) of
        {ok, _Info, Payload} -> Payload;
        _ -> <<>>
    end.

%% Generic per-tokens save. Used for cold and continued saves; finish
%% save goes through fire_finish_save_for_req. Returns the updated
%% Req with cache_delta_created bumped when the save actually fired.
fire_save_for_tokens(Reason, Tokens, Req, Data) ->
    Should =
        case Reason of
            cold ->
                Min = maps:get(min_tokens, Data#data.policy),
                length(Tokens) >= Min;
            _ ->
                true
        end,
    case fire_save_if(Should, Reason, Tokens, Req, Data) of
        fired -> bump_cache_delta(Req, Tokens);
        not_fired -> Req
    end.

%% Returns `{ok, Key}` if a finish save fired. Returns `skipped` if
%% the policy suppressed it (live token count below `min_tokens`).
fire_finish_save_for_req(LiveTokens, Req, Data) ->
    Should = erllama_cache_policy:should_finish_save(
        length(LiveTokens), Data#data.policy
    ),
    case fire_save_if(Should, finish, LiveTokens, Req, Data) of
        fired -> {ok, make_key(LiveTokens, Req, Data)};
        not_fired -> skipped
    end.

fire_save_if(false, _Reason, _Tokens, _Req, _Data) ->
    not_fired;
fire_save_if(true, Reason, Tokens, Req, Data) ->
    BuildMeta = build_meta_for(Reason, Tokens, Req, Data),
    T0 = erlang:monotonic_time(nanosecond),
    Payload = backend_kv_pack(Tokens, Req#req.seq_id, Data),
    Elapsed = erlang:monotonic_time(nanosecond) - T0,
    erllama_cache_counters:add(?C_PACK_TOTAL_NS, max(Elapsed, 0)),
    case
        erllama_cache_writer:save(
            Data#data.tier_srv, Data#data.tier, BuildMeta, Payload, 0
        )
    of
        {ok, _} ->
            fired;
        {error, already_present} ->
            %% A prior save (typically this request's cold save) or
            %% a concurrent writer already published the key. The
            %% data is in cache, so credit this request for it.
            fired;
        {error, _SaveErr} ->
            erllama_cache_counters:incr(?C_SAVES_DROPPED),
            not_fired
    end.

%% Bump the largest save contribution above the warm prefix. Each
%% call site passes the tokens it just persisted; the
%% `cache_delta_created` field tracks the max so a continued save
%% followed by a finish save doesn't double-count.
bump_cache_delta(Req, Tokens) ->
    Contribution = max(length(Tokens) - Req#req.cache_hit_prefix_len, 0),
    Req#req{
        cache_delta_created = max(Req#req.cache_delta_created, Contribution)
    }.

%% Evict and shutdown saves: walk every in-flight req and fire one
%% save per req that has non-empty context_tokens.
fire_save_for_reason(_Reason, Data) when map_size(Data#data.req_table) =:= 0 ->
    Data;
fire_save_for_reason(Reason, Data) ->
    NewTable = maps:map(
        fun(_SeqId, Req) ->
            case Req#req.context_tokens of
                [] ->
                    Req;
                Tokens ->
                    Req0 =
                        case fire_save_if(true, Reason, Tokens, Req, Data) of
                            fired -> bump_cache_delta(Req, Tokens);
                            not_fired -> Req
                        end,
                    Req0#req{last_save_at = length(Tokens)}
            end
        end,
        Data#data.req_table
    ),
    Data#data{req_table = NewTable}.

build_meta_for(SaveReason, Tokens, Req, Data) ->
    #{
        save_reason => SaveReason,
        quant_bits => Data#data.quant_bits,
        fingerprint => request_fp(Req, Data),
        fingerprint_mode => Data#data.fingerprint_mode,
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash,
        tokens => Tokens,
        context_size => Data#data.context_size,
        prompt_text => <<>>
    }.

make_key(Tokens, Req, Data) ->
    erllama_cache_key:make(#{
        fingerprint => request_fp(Req, Data),
        quant_type => Data#data.quant_type,
        ctx_params_hash => Data#data.ctx_params_hash,
        tokens => Tokens
    }).

%% Fingerprint to use for cache identity. Returns the per-request
%% snapshot if captured at admission; otherwise the current
%% effective fingerprint.
request_fp(#req{request_fp = undefined}, #data{effective_fp = FP}) -> FP;
request_fp(#req{request_fp = FP}, _Data) -> FP.

%% Per-seq kv_pack: prefer the seq-aware arity if the backend
%% implements it, otherwise fall back to the 2-arity (which is
%% seq_id=0 implicitly).
backend_kv_pack(Tokens, SeqId, #data{backend = Mod, backend_state = S}) ->
    case erlang:function_exported(Mod, kv_pack, 3) of
        true -> Mod:kv_pack(S, Tokens, SeqId);
        false -> Mod:kv_pack(S, Tokens)
    end.

%% Per-seq kv_unpack: prefer the seq-aware arity.
backend_kv_unpack(Bin, SeqId, #data{backend = Mod, backend_state = S}) ->
    case erlang:function_exported(Mod, kv_unpack, 3) of
        true -> Mod:kv_unpack(S, Bin, SeqId);
        false -> Mod:kv_unpack(S, Bin)
    end.

is_strict_prefix([], _) -> true;
is_strict_prefix([H | T1], [H | T2]) -> is_strict_prefix(T1, T2);
is_strict_prefix(_, _) -> false.

%% =============================================================================
%% Internal: backend dispatch
%% =============================================================================

backend_call(#data{backend = Mod, backend_state = S}, Fn, Args) ->
    apply(Mod, Fn, [S | Args]).

%% Resolve a model() reference to a Pid that gen_statem:call/2,3
%% accepts. Binary IDs go through the registry; pids pass through.
%% Crashes with `{noproc, {erllama_model, not_found, ModelId}}` if
%% the model is not registered, so callers do not have to special-
%% case that path - they get a useful error tag rather than a bare
%% `noproc`.
via(Pid) when is_pid(Pid) ->
    Pid;
via(ModelId) when is_binary(ModelId) ->
    case erllama_registry:whereis_name(ModelId) of
        Pid when is_pid(Pid) -> Pid;
        undefined -> exit({noproc, {?MODULE, not_found, ModelId}})
    end.

%% prime_logits/3 is gone — the warm-restore primer is now part of
%% the per-#req flow. setup_lookup/3 drops the last KV cell via
%% warm_restore_primer/3, and the resulting #req.prefill_cursor
%% carries `[LastWarm | Remaining]` so the next step_tick re-prefills
%% the primer token and any remaining tail in one shot.
