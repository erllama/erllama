# Changelog

All notable changes to erllama are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.5.1] - 2026-05-17

Documentation-only patch on top of 0.5.0.

### Added

- `## Tool-call handling` section in the README describing what
  erllama exposes (per-model `tool_call_markers`, the
  `{tool_call_delta, _}` / `{erllama_tool_call_end, _, Full}`
  streaming wire, and the automatic greedy-on-syntax sampler
  swap) and what it deliberately leaves to the HTTP layer (tool
  id minting, JSON parsing, canonicalisation).
- New `guides/tool-calls.md` companion to the README section,
  linked from the documentation table.
- New `internals/request-lifecycle.md` describing the per-model
  `gen_statem` admission, cache resolution, decode, and save
  pipeline.
- `internals/c-safety-audit.md` added to the HexDocs navigation.

### Changed

- README rewritten for sharper top-of-funnel: tighter "Why" list,
  cleaner Quick taste, Common patterns block replacing the old
  long example.
- Architecture diagram corrected — `erllama_cache_ramfile_srv`
  and `erllama_cache_disk_srv` are operator-started standalone
  servers, not children of `erllama_cache_sup`. Added
  `erllama_registry` and `erllama_inflight` which were missing.
- "Inside a request" lifecycle updated for the multi-seq
  scheduler: two states (idle/running) instead of the v0.1
  three-phase model, with co-batched `nif_step` and inline
  thinking/tool-call marker recognition.

## [0.5.0] - 2026-05-16

Tool-call exact-replay scaffolding for downstream HTTP front ends.
Models loaded with `tool_call_markers` produce structured boundary
messages on the streaming wire so a caller can capture the exact
bytes the model sampled, store them under a tool id, and splice
them back verbatim on later turns to keep the KV-cache prefix
match working. Companion primitives expose explicit suffix replay
and sticky per-session seq_id pinning.

### Added

- `tool_call_markers => #{start, end, payload_start (optional),
  payload_end (optional)}` on `erllama:load_model/2` Config. Each
  binary is tokenised through the model's own vocabulary at load
  time; multi-token markers are supported. Omitting the key keeps
  the backend on the existing path (#39, #42).
- Streaming wire on `infer/4` gains
  `{erllama_token, Ref, {tool_call_delta, Bin}}` per chunk and a
  single `{erllama_tool_call_end, Ref, Full :: binary()}` per
  span, with `Full` carrying every emitted delta concatenated so
  the downstream's exact-replay map stores them verbatim without
  re-buffering (#39).
- `erllama:prefill_only/3` accepting `Opts` with `parent_key`.
  When passed a prior turn's `finish_key`, the call warm-restores
  from that row and prefills only the new suffix before firing the
  finish save — useful for chaining cache-warming calls across
  turns (#40).
- `session_id => term()` on `infer/4` Params and
  `complete/3` / `prefill_only/3` Opts. Pins the underlying seq_id
  to that session across requests so the next turn whose prompt
  continues the stored tokens truncates-and-prefills in place on
  the already-live KV cells (`cache_hit_kind => sticky`).
  Concurrent admits on the same `session_id` return `{error,
  sticky_busy}`. Release with `erllama:end_session/2` (#41).
- Per-request greedy sampler swap on tool-call syntax tokens.
  Models with `tool_call_markers` build a second sampler chain
  (`temperature => 0`) at admission; the scheduler routes syntax
  tokens through it so a tool call is byte-deterministic from a
  fixed prefix. Optional payload markers flip back to the request's
  normal sampler for caller-supplied string contents so they stay
  diverse (#42).

### Changed

- `step_result()` on `erllama_model_backend` gains four variants
  (`{tool_call_token, _}`, `tool_call_end`, `{tool_call_payload_open,
  _}`, `{tool_call_payload_close, _}`). Backends without
  tool-call markers emit none of them.
- `erllama_model_stub` phase machine re-keyed from sampler ref onto
  seq_id so mid-request sampler swaps (the new greedy-on-syntax
  path) don't reset state across ticks. `seq_rm` now cleans the
  per-seq phase entry.

## [0.4.0] - 2026-05-16

Anthropic-Messages compatibility follow-ups: per-request cache
delta accounting, a real thinking sampler in the llama.cpp
backend, and caller-side thinking-budget clipping. All three are
strict additions; existing consumers see no shape change unless
they opt in.

### Added

- `cache_delta => #{read := N, created := N}` on
  `completion_result()`, `stats()`, and `prefill_result()` so the
  downstream Anthropic Messages server can emit accurate
  `cache_creation_input_tokens` / `cache_read_input_tokens`
  values. `read` is the warm prefix length restored at admission;
  `created` is the largest contribution this request added to the
  cache beyond that prefix (#35).
- `thinking_markers => #{start := binary(), end := binary()}` on
  `erllama:load_model/2` Config. The backend tokenises both
  strings through the model's vocabulary at load time and the
  `step/2` wrapper maps any sampled token matching a marker into
  `{thinking_token, _}` or `thinking_end`. Multi-token markers
  (BPE-split `<think>`) are supported. Omitting the key keeps the
  backend on the non-thinking path (#36).
- `thinking_signing_key` application env. When set, the real
  backend's `thinking_signature/3` HMAC-SHA256s the observed
  thinking-phase bytes with this key; unset returns `<<>>` so
  the downstream omits `signature_delta` (#36).
- `thinking_budget_tokens => pos_integer()` on `infer/4` `Params`.
  Caps the number of `{thinking_delta, _}` payloads delivered
  before the scheduler synthesises `{erllama_thinking_end, _, _}`
  and re-routes further model thinking tokens through the normal
  post-thinking pipeline (#37).

### Changed

- `thinking_signature/2` callback on `erllama_model_backend`
  bumped to `/3` (third argument is the accumulated thinking
  bytes). Backwards-compatible: only `erllama_model_stub`
  implemented the optional callback in 0.3 and the stub +
  scheduler + new `erllama_model_llama` move in lockstep (#36).

## [0.3.0] - 2026-05-16

Anthropic-Messages compatibility additions on top of 0.2.0:
caller-supplied stop sequences with trimmed output, an opt-in
extended-thinking message surface with per-block integrity
signatures, and a round of NIF safety hardening on the C/C++ side.

### Added

- `stop_sequences :: [binary()]` on `infer/4` `Params` and
  `complete/3` `Opts`. Generation halts on the first occurrence of
  any element in the accumulated detokenised output. The match is
  trimmed from the streamed `{erllama_token, _, _}` chunks and the
  synchronous `reply`, and the matched binary is reported as
  `stop_sequence` on the result map (`complete/3`) and stats map
  (`infer/4` done message). The key is absent when generation hit
  `length`, was cancelled, or reached EOG without a match. The
  previously reserved `stop` placeholder is renamed to
  `stop_sequences`; it was never wired up so this is not a
  breaking change (#32).
- `thinking => enabled | disabled` on `infer/4` `Params` (default
  `disabled`). When `enabled` against a thinking-capable backend,
  streaming requests receive `{erllama_token, Ref, {thinking_delta,
  Bin}}` fragments and a single `{erllama_thinking_end, Ref, Sig}`
  close marker before any subsequent token. `Sig` is an opaque
  integrity signature the downstream forwards verbatim into the
  Anthropic `signature_delta` SSE event, or `<<>>` when no
  signature is available (#33).
- `erllama_model_backend` gains `{thinking_token, token_id()}` and
  `thinking_end` variants on `step_result()` plus an optional
  `thinking_signature/2` callback. Backends without extended
  thinking emit neither variant and require no changes (#33).

### Changed

- `llama_batch_init`, `llama_batch_free`, and `llama_batch_get_one`
  are now routed through `erllama_safe_batch_*` `noexcept` shims
  so a C++ exception cannot unwind through the C NIF frame (#30).
- Per-thread `thread_local` storage replaces the process-global
  log buffer used by the malformed-GGUF classifier; concurrent
  model loads no longer scramble each other's `GGML_ASSERT` text
  on a NULL return (#31).
- NIF unload no longer calls `llama_backend_free` (avoids a
  `pthread_once` wedge on `.so` reload paths) and clears the
  `llama_log_set` callback so a post-unload log emission cannot
  dispatch into freed memory (#31).

## [0.2.0] - 2026-05-15

Multi-sequence batched scheduling, map-shaped completion results,
chunked prefill, per-model observability, and direct passthrough of
the llama.cpp multi-GPU / flash-attention / KV-quant params.

### Changed (breaking)

- `erllama:complete/2,3` and `erllama_model:complete/2,3` now return
  `{ok, completion_result()}` instead of the legacy
  `{ok, ReplyBinary, GeneratedTokens}` tuple. The map carries:
  - `reply :: binary()`
  - `generated :: [token_id()]`
  - `context_tokens :: [token_id()]` (prompt ++ generated)
  - `committed_tokens :: non_neg_integer()` (`length(context_tokens)`)
  - `finish_key :: cache_key() | undefined` — token-exact key for the
    full context, suitable as `parent_key` on the next turn;
    `undefined` if the finish save was suppressed
  - `cache_hit_kind :: exact | partial | cold`
  - `finish_reason :: stop | length | cancelled`
  - `stats :: stats()`

  Mechanical migration:

  ```erlang
  %% before
  {ok, Reply, _Tokens} = erllama:complete(Model, Prompt).
  %% after
  {ok, #{reply := Reply, finish_key := FK}} =
      erllama:complete(Model, Prompt).
  ```

- Streaming `{erllama_done, Ref, Stats}` (`infer/4`) `Stats` map
  gains two additive keys: `finish_key` and `committed_tokens`. No
  shape break for existing consumers (#21).

### Added

#### Multi-sequence batched scheduler (#24, #25, #26, #27)

- `erllama_nif:step/2` is the new multi-sequence batched decode
  primitive. One `llama_decode` call mixes prefill and decode rows
  freely (SARATHI-style co-batching), bounded by the live context's
  `n_batch`. Returns `{error, batch_overflow}` cleanly so a
  budget-aware scheduler can shrink and retry, and `{error,
  no_logits}` when a decode row has no prefill yet on its seq.
- Per-context `per_seq[]` tracking (`last_logits_idx`, `next_pos`)
  with `ERLLAMA_N_SEQ_MAX_CAP = 256`. `kv_unpack` / `kv_seq_rm`
  refresh the per-seq position so subsequent `step` calls see
  correct state.
- `erllama_model_backend` behaviour gains optional callbacks
  `step/2`, `sampler_new/2`, `sampler_free/1`, `seq_rm/2`,
  `seq_rm_last/3`, plus seq-aware `kv_pack/3` and `kv_unpack/3`.
  All optional; existing backends keep compiling.
- The model gen_statem now runs a multi-tenant scheduler. With
  `context_opts.n_seq_max => 1` (the default), behaviour is
  bit-identical to 0.1: exactly one request runs at a time. Setting
  `n_seq_max > 1` lets up to N requests prefill and decode
  concurrently through one `llama_decode` per tick. State collapses
  from `idle/prefilling/generating` to `idle/running`; admissions
  past the seq-id capacity queue FIFO in `pending`.
- Each in-flight request owns its own sampler chain (built at
  admission via `backend:sampler_new/2`, freed at finish), so
  concurrent requests with different `temperature` / `seed` /
  `grammar` settings never share sampler state.
- Cache save reasons (`cold`, `continued`, `finish`, `evict`,
  `shutdown`) all thread through the request's `seq_id` and remain
  token-exact per-sequence.

#### Chunked prefill (#28)

- `prefill_chunk_size` policy knob caps how many tokens a single
  prefill row contributes to one tick. Default
  `max(64, n_batch div 4)`; pass `infinity` to disable. A long
  prompt is sliced across multiple ticks so it never monopolises
  the batch and concurrent decoders keep making progress between
  chunks. Layered on top of the `n_batch` per-tick budget.

#### `prefill_only/2` (#21)

- `erllama:prefill_only/2` and `erllama_model:prefill_only/2` decode
  a prompt into KV state and fire a finish save without sampling
  any output tokens. Returns a `prefill_result()` map carrying
  `context_tokens`, `committed_tokens`, `finish_key`, and
  `cache_hit_kind`. Useful for priming the cache before a burst of
  short follow-ups, or for holding a warm session across long pauses
  without consuming generation budget.

#### Per-model observability (#22)

- New public ETS table `erllama_model_obs`, owned by
  `erllama_inflight`, written by each model gen_statem on every
  state transition and read lock-free from any process (including
  remote nodes via `erpc`).
- Four new accessors on `erllama`:
  - `phase/1`: `idle | prefilling | generating` for one model id.
  - `pending_len/1`: gen_statem pending FIFO depth (calls queued
    behind whatever is currently running).
  - `last_cache_hit/1`: `#{kind, prefix_len}` of the most recent
    admission, or `undefined` if the model has never admitted.
  - `queue_depth/1`: per-model variant of the existing global
    `queue_depth/0`; counts admitted streaming `infer/4` rows.
- `model_info/1` map gains `phase`, `pending_len`, and
  `last_cache_hit` keys. Additive: existing keys preserved.

#### llama.cpp option passthrough (#23)

- `erllama_nif:load_model/2` now reads three additional keys from
  `model_opts`:
  - `split_mode :: none | layer | row`: multi-GPU split policy.
  - `main_gpu :: non_neg_integer()`: GPU index when `split_mode = none`.
  - `tensor_split :: [float()]`: per-device proportions (up to 16
    entries; shorter lists zero-fill).
- `erllama_nif:new_context/2` reads three more from `context_opts`:
  - `flash_attn :: boolean() | auto`: enable, disable, or defer to
    llama.cpp.
  - `type_k`, `type_v :: f16 | f32 | bf16 | q4_0 | q5_0 | q5_1 | q8_0`:
    KV cache element type for keys and values.
- Bad atoms raise `badarg` before the load runs.

### Fixed

- `warm_restore_primer` passed `1` instead of the current cell count
  to the prefill primer, so warm restores that ran the primer at a
  non-zero offset wrote KV at the wrong position. The primer now
  takes the live cell count from the per-seq tracker.
- The cold prefill path could fire the cold save inside the
  remainder prefill rather than between the trim prefix and the
  remainder, leading to a save row that did not match the
  trim-aligned boundary. The cold save now fires at the
  cursor-emptied transition between the trim and the remainder.

### Internal

- New types exported from `erllama_model`: `completion_result/0`,
  `prefill_result/0`.
- `erllama_model_t` now owns the `tensor_split` buffer; the vendored
  llama.cpp aliases the pointer rather than copying it, so its
  storage must outlive the model.
- `erllama_model_stub` derives per-seq tokens from
  `phash2({decode_step_stub, SeqId, Sampler})` so a scheduler bug
  that swaps samplers between seqs becomes observable in tests.

## [0.1.2] - 2026-05-12

Cluster-routing primitives, speculative-decoding verifier, a
cold-path correctness fix, and C-safety CI tooling. All additions
are backwards-compatible; existing API call sites unchanged.

### Added

#### Cluster routing and load-balancing (#13)

- `erllama:queue_depth/0` returns O(1) inflight count via an
  atomics counter parked in persistent_term, readable cross-node
  via `erpc`. Used by the upcoming `erllama_cluster` load
  balancer (least_loaded, power_of_two strategies).
- `erllama:list_cached_prefixes/2` returns the longest cached
  prefix length of a token list for a given model on this node,
  across all cache tiers. Used by the cluster cache-affinity
  router.
- `erllama_nif:vram_info/0` walks every loaded ggml backend and
  sums free + total memory across non-CPU devices; returns
  `{error, no_gpu}` on a CPU-only build. Used by the cluster
  scheduler for bin-packing model placement.
- `erllama:list_models/0` map gains `model_id`, `quant_tag`,
  `loaded_at_monotonic`, and `vram_estimate_b` keys. Existing
  keys (`id`, `pid`, `status`, etc.) are unchanged.

#### Speculative decoding (#13)

- `erllama:draft_tokens/3` synchronously generates up to `Max`
  next-token ids for a prefix. Times out at 30 s with a clean
  cancel + drain so the caller's mailbox stays clean. Empty
  prefix is rejected as `{error, empty_prefix}`.
- `erllama:verify/4` runs `PrefixTokens ++ Candidates` through
  the model in one forward pass and returns the longest accepted
  prefix length plus the verifier's own next token. Acceptance
  walks `Argmax[P + i - 1] == c_i` and stops at the first
  mismatch. End-of-generation tokens map to the atom `eos`.
  Snapshot + restore protocol leaves the caller's pre-call
  context view unchanged. Allowed only from the model
  gen_statem's idle state; non-idle callers receive
  `{error, busy}`.
- Token-id streaming: `erllama_model:stream_emit` now also sends
  `{erllama_token_id, Ref, Id}` on every produced token, in
  addition to the existing `{erllama_token, Ref, Bin}`.
  Empty-text tokens (special tokens, BPE merges with no visible
  bytes) still produce an id message. Existing consumers ignore
  the new tag.

#### Backend behaviour

- Optional callbacks `extra_metadata/1` (vram-related model
  metadata) and `verify/4` (speculative verifier) on
  `erllama_model_backend`. Backends that omit either get a
  graceful `{error, not_supported}` fallback (#13).
- Optional callback `seq_clear/1` on `erllama_model_backend`.
  Llama backend implements it as `llama_state_seq_rm(0, 0, -1)`.
  Called by the model layer at the top of `enter_prefilling`;
  see the Fixed section below (#16).

#### NIFs (#13)

- `nif_model_size/1`, `nif_model_n_layer/1`,
  `nif_forward_with_argmax/2`, `nif_vram_info/0`. Previously
  unreachable from Erlang.

### Fixed

- Cold-path prefill KV-state leak: `erllama_model:enter_prefilling`
  did not reset the llama_context's KV cache before the new
  prefill. `llama_batch_get_one` auto-positions at `n_past`, so a
  second cold request on the same model wrote its prompt KV at
  `[previous_n_past..]` instead of `[0..]`, producing different
  output for the same prompt + seed across calls. The new
  `seq_clear/1` callback wipes seq 0 before the cold prefill.
  Warm restores via `kv_unpack` were already correct (#16).

### Changed

- Vendored `c_src/llama.cpp/` bumped from `b9093` to `b9119`
  (16 files, mostly Metal/CUDA tweaks plus a new
  `ggml-cuda/allreduce` kernel pair). Public `llama.h` API used
  by the NIF is unchanged (#15).
- `erllama_inflight:register/2` and `unregister/1` switched to
  `ets:insert_new` / `ets:take` so the new atomics counter sees
  only true admissions and true removals; double-register or
  double-unregister become observable no-ops (#13).

### Internal

- New CI jobs: `sanitizers` (ASan+UBSan against
  `tinyllamas/stories260K.gguf` under `LD_PRELOAD`'d libasan),
  `clang-tidy` (NIF sources only), and `scan-build`
  (Clang Static Analyzer with `--status-bugs`) (#14).
- New `c_src/CMakeLists.txt` options:
  `ENABLE_ASAN`, `ENABLE_TSAN`, `ENABLE_UBSAN`, `ENABLE_CLANG_TIDY`,
  scoped to the `erllama_nif` target; default OFF.
- `.clang-tidy` config at repo root. Vendored llama.cpp is not
  linted.

### ROADMAP

- Pipeline parallelism deferred; blocked on upstream llama.cpp
  adding layer-range execution to `llama_decode`. The cluster
  degrades gracefully via `function_exported`.
- Verify context isolation: the current snapshot/restore protocol
  does not preserve the caller's pre-call `decode_ready` flag;
  callers are assumed to issue `decode_one` imminently. A v2
  extension to cover `decode_ready` is documented for the next
  contributor.

## [0.1.1] - 2026-05-12

NIF safety and SIGSEGV hardening. No public API additions; new
error tuples surface paths that previously crashed the BEAM or
raised `badarg` across a dirty scheduler.

### Fixed

- Adapter use-after-free / double-free when `free_model/1` ran
  while adapter wrappers still referenced the model. The model
  resource now tracks `active_adapters` alongside `active_contexts`
  and defers `llama_model_free` until both reach zero (#10).
- Race in `set_adapters/2` where a concurrent `adapter_free` could
  null the underlying pointer between the per-adapter mutex
  release and the `llama_set_adapters_lora` call. Locks are now
  held across the llama call in pointer-sorted order to defeat
  AB-BA between concurrent callers (#10).
- Per-message memory leak in `apply_chat_template` when the
  message list was malformed or allocation failed mid-build. The
  helper now releases its own role/content allocations on every
  error path (#10).
- `prefill/2` and `embed/2` walked past the KV slab when the
  prompt size reached `n_ctx`, and produced undefined behaviour
  when it exceeded `n_batch`. Both now bounds-check against the
  live context before touching state, returning
  `{error, context_overflow}` or `{error, batch_overflow}` (#11).
- `apply_chat_template/2` raised `badarg` across the dirty
  scheduler when `content` was a list-of-maps (Anthropic-style
  content blocks) instead of a binary. Returns
  `{error, invalid_content}` (#11).
- `load_model/2` surfaced a generic `{error, load_failed}` for
  malformed GGUF files. Now returns `{error, malformed_gguf}` on
  a best-effort basis when the captured llama log line contains
  `GGML_ASSERT`. Best-effort only: `llama_log_set` is process
  global and concurrent loads can mis-attribute classification.
  A `GGML_ASSERT` that hits `abort()` still terminates the BEAM
  process; subprocess isolation would be required for a complete
  fix and is intentionally out of scope (#11).

### Added

- New error atoms returned by the NIF: `context_overflow`,
  `batch_overflow`, `invalid_content`, `malformed_gguf`. Callers
  matching `{error, _}` are unaffected; callers that care about
  the specific reason should match the new atoms.

## [0.1.0] - 2026-05-11

Initial public release.

### Added

#### Public API

- Native Erlang/OTP wrapper around llama.cpp via a single
  dirty-scheduler NIF (`erllama_nif`) covering model load, context
  construction, tokenisation, prefill, single-token decode, and
  KV pack/unpack.
- Models are identified by `binary()` on the public API.
  `erllama:load_model/2`, `complete/2,3`, `unload/1`, `status/1`,
  `evict/1`, `shutdown/1` take `binary() | pid()`. Internal
  registration uses `{via, erllama_registry, BinaryId}` so user-
  supplied ids cannot exhaust the atom table.
- `erllama:list_models/0` returning `[model_info()]` and
  `erllama:model_info/1` keyed on a model id.
- Public `erllama:tokenize/2` and `erllama:detokenize/2` keyed on a
  model id. The low-level `erllama_nif:tokenize/3` and
  `erllama_nif:detokenize/2` remain available.
- `erllama:unload_model/1` as an alias for `erllama:unload/1`
  matching the OpenAI/Ollama-style naming downstream HTTP servers
  use.
- `erllama:infer/4` streaming inference. Returns `{ok, Ref}`;
  tokens are delivered to the caller as `{erllama_token, Ref, _}`,
  `{erllama_done, Ref, Stats}`, `{erllama_error, Ref, Reason}`.
- `erllama:cancel/1`. Idempotent and fire-and-forget; observed
  between tokens.
- `erllama:apply_chat_template/2`. Renders a normalised chat
  request (`messages`, `system`, `tools`) through the model's
  GGUF chat template and tokenises. Backed by
  `llama_chat_apply_template`.
- `erllama:embed/2`. Per-sequence pooled embedding via
  `llama_get_embeddings_seq` with last-token fallback.

#### Sampling

- Sampler parameters: `complete/3` and `infer/4` honour
  `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
  `seed`, and `grammar` via one combined chain builder
  (`erllama_nif:configure_sampler/2`). Chain order:
  `grammar -> repetition_penalty -> top_k -> top_p -> min_p ->
  (temperature > 0 ? temp -> dist(seed) : greedy)`. `set_grammar/2`
  retained as a backwards-compatible alias.
- Grammar-constrained sampling: pass `grammar => GBNF` in the
  `complete/3` Opts or `infer/4` Params; the per-model sampler
  chain is rebuilt as grammar then greedy for the duration of
  the request and reset on completion or cancellation.

#### LoRA adapters

- `erllama:load_adapter/2`, `unload_adapter/2`,
  `set_adapter_scale/3`, `list_adapters/1`. Per-adapter sha256 +
  scale fold into the cache via
  `erllama_cache_key:effective_fingerprint/2`, so rows produced
  with adapter A never collide with rows from adapter B.
  Snapshot-at-admission semantics keep in-flight requests on
  their original fingerprint even if an adapter mutation arrives
  mid-generation.

#### Concurrency model

- Concurrent request queue: a second `complete/3` or `infer/4`
  arriving while one is in flight is queued FIFO instead of
  getting `{error, busy}`. The reply `{ok, Ref}` is sent as soon
  as the call is admitted; streaming events follow once the
  queue head advances to the request.
- Decode loop schedules each step via
  `gen_statem:cast(self(), decode_step)` instead of `next_event`
  so cancel, evict, status, and queued requests interleave
  fairly between tokens.
- Seq-aware NIFs (infrastructure for 0.2 multi-seq batching):
  `nif_kv_pack/4` accepts an explicit `seq_id`; new
  `erllama_sampler_t` resource owning a standalone
  `llama_sampler*`; `nif_sampler_new/2`, `nif_sampler_free/1`.
  Cache rows stay seq-id-free: `seq_id` is a save/load call
  argument, never row metadata.

#### Internals

- `erllama_registry` module: ETS-backed `via` callback for binary
  model ids.
- `erllama_inflight` module: `Ref -> ModelPid` table so
  `cancel/1` routes to the right gen_statem.
- `erllama_model_backend` optional callbacks:
  `apply_chat_template/2`, `embed/2`, `set_grammar/2`,
  `configure_sampler/2`, `clear_sampler/1`, `load_adapter/2`,
  `unload_adapter/2`, `apply_adapters/2`. Backends that omit them
  surface `{error, not_supported}` from the public API.

#### Cache subsystem

- Token-exact KV cache with three independently-supervised tiers:
  RAM (ETS slabs), `ram_file` (`/dev/shm`), and disk (plain read
  I/O).
- Sole-writer arbitration through `erllama_cache_meta_srv`; reads
  on the hot path go to ETS directly. `lookup_exact/1` is a
  single atomic `ets:lookup` (no two-call race) and the meta
  server cancels waiter timers when an early reply lands so the
  mailbox doesn't bloat under load.
- The disk tier reads files via plain `file:read_file/1` into a
  fresh BEAM heap binary; mmap is deliberately not used. The
  process already mmaps multi-GB GGUF weights, and a region
  binary that outlived its closing NIF call would have exposed
  the BEAM to SIGBUS from any external truncation.
- Crash-safe save publish protocol: reserve, write_tmp, check,
  `link(2)`, mark_published; two-stage TTL cleanup with orphan
  adoption.
- Five save reasons (`cold`, `continued`, `finish`, `evict`,
  `shutdown`) with async/sync semantics matching their use.
- `saves_dropped` counter: bumps whenever a back-pressured writer
  pool refuses a save the model wanted to fire.
- Multi-turn warmth via explicit `parent_key` resume and
  stateless longest-prefix walk for OpenAI/Anthropic-shaped
  clients.
- `erllama_scheduler` memory-pressure poller with pluggable
  sources (`memsup`, `nvidia-smi`, custom callback). Off by
  default. Sweep timer is cancelled on `terminate/2` so a
  supervisor restart never leaves a zombie firing into a fresh
  server.
- `erllama_cache_writer` dirty-IO writer pool with a leak-proof
  reservation semaphore. `pin_and_load/2` wraps load + unpack in
  `try/after` so the holder is always checked in.
- Persisted hit counters (u32 in disk header) so popular prefixes
  survive an LRU walk after restart.
- End-to-end metrics: hits/misses/saves/evictions plus per-path
  latency totals (`pack_total_ns`, `load_total_ns`,
  `longest_prefix_ns`, `longest_prefix_probes`).

#### NIF safety

- Per-resource `pthread_mutex` and two-resource lifetime pattern
  for safe concurrent `free_*/1` plus dirty NIF ops.
- `extern "C" noexcept` shim catching every llama.cpp C++
  exception at the boundary; `decode_one` defensive guard
  against `GGML_ASSERT` aborts.
- `llama_backend_init` is deferred to the first `nif_load_model`
  via `pthread_once`, so cache-only and unit-test workloads do
  not pay the `ggml_backend_load_all` cost at NIF load.
- `nif_tokenize` and `nif_detokenize` honour `release_pending`,
  so a model returned by `free_model/1` as `{ok, deferred}`
  cannot be reused via tokenize.
- `nif_detokenize` fails closed on `n_vocab <= 0` (matches
  `nif_prefill`).
- `make_errno_atom` maps FreeBSD's `EINTEGRITY` to `eintegrity`.

#### Tooling

- `FindErlang.cmake` (adopted from erlang-rocksdb) detects
  `ERTS_INCLUDE_DIR` via the standard CMake find-module
  contract.
- Bench harness (`bench/run.sh`) with cold-vs-warm matrix and a
  4-agent shared-prefix scenario. TinyLlama and LLaMA-3 8B
  presets.

### CI

- `actions/checkout` and `actions/cache` bumped to `@v5`
  (Node.js 24).
- `xref`, `dialyzer`, `erlfmt`, `elvis` promoted to gate jobs;
  `build`, `eunit`, `proper`, `ct`, `freebsd` depend on them.
- macOS matrix is `macos-14, macos-15`.
- FreeBSD matrix added: `release: ['14.2', '14.4']`. Inside the
  VM: refresh `pcre2` so git can run, install `git`, set
  `git config --global --add safe.directory '*'` so llama.cpp's
  build-info `git rev-parse` succeeds.
- `erllama_nif_tests:load_model_rejects_non_existent_path_test` is
  now a generator with a 60 s timeout to absorb the lazy Metal
  init on macOS.

### Tests

- 211 EUnit + PropEr property tests + 7 stub Common Test cases.
  Real-model Common Test suite gated on `LLAMA_TEST_MODEL`
  (14 cases including seed determinism, grammar+sampler,
  apply_chat_template, embeddings, KV pack/unpack round-trip).
- New stub-backed coverage: sampler params (`erllama_sampler_tests`),
  LoRA adapters + cache identity (`erllama_lora_tests`), FIFO
  queueing of concurrent infers (`erllama_streaming_tests`).
- Multi-platform CI: Ubuntu 24.04 amd64, Ubuntu 24.04 arm64,
  macOS 14 + 15 (Apple Silicon), FreeBSD 14.2 + 14.4. OTP 28
  across the matrix.

### Documentation

- README rewritten as a friendly entry point with snippets.
- User guides: loading, caching, configuration, building, examples.
- Internal design notes: cache design, publish protocol, NIF safety.
- ex_doc-friendly module documentation throughout.
- `ROADMAP.md`: what 0.1 doesn't do yet (multi-seq concurrent
  decoding, speculative decoding, vision, audio, ONNX/safetensors,
  stop-sequences, telemetry hooks, multi-GPU pressure, KV
  compression, cluster).
- README closes with a teaser for the upcoming `erllama_cluster`
  application: a separate OTP project that coordinates a fleet of
  erllama nodes (request distribution, cross-node speculative
  decoding, pipeline parallelism over QUIC).

### Acknowledgements

Same idea as [antirez/ds4](https://github.com/antirez/ds4).

[Unreleased]: https://github.com/erllama/erllama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/erllama/erllama/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/erllama/erllama/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/erllama/erllama/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/erllama/erllama/releases/tag/v0.1.0
