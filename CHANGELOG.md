# Changelog

All notable changes to erllama are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.2.0] - unreleased

API reshape: `complete/2,3` now returns a map, and `prefill_only/2`
is added so sessions can warm KV without sampling a reply.

### Changed (breaking)

- `erllama:complete/2,3` and `erllama_model:complete/2,3` now return
  `{ok, completion_result()}` instead of the legacy
  `{ok, ReplyBinary, GeneratedTokens}` tuple. The map carries:
  - `reply :: binary()`
  - `generated :: [token_id()]`
  - `context_tokens :: [token_id()]` — full token list (prompt ++ generated)
  - `committed_tokens :: non_neg_integer()` — `length(context_tokens)`
  - `finish_key :: cache_key() | undefined` — token-exact key for the
    full context, suitable as `parent_key` on the next turn;
    `undefined` if the finish save was suppressed
  - `cache_hit_kind :: exact | partial | cold`
  - `finish_reason :: stop | length | cancelled`
  - `stats :: stats()`
- Streaming `{erllama_done, Ref, Stats}` (`infer/4`) `Stats` map
  gains two additive keys: `finish_key` and `committed_tokens`.
  No shape break — existing keys (`prompt_tokens`,
  `completion_tokens`, `prefill_ms`, `generation_ms`,
  `cache_hit_kind`, `finish_reason`, `cancelled`) remain.

### Added

- `erllama:prefill_only/2` and `erllama_model:prefill_only/2` —
  decode a prompt into KV state and fire a finish save without
  sampling any output tokens. Returns a `prefill_result()` map with
  `context_tokens`, `committed_tokens`, `finish_key`, and
  `cache_hit_kind`. Useful for priming the cache or holding a warm
  session across long pauses without consuming generation budget.
- New `completion_result()` and `prefill_result()` exported types in
  `erllama_model`.

### Migration

Callers matching the old 3-tuple need the map shape. Mechanical:

```erlang
%% before:
{ok, Reply, _Tokens} = erllama:complete(Model, Prompt).

%% after:
{ok, #{reply := Reply}} = erllama:complete(Model, Prompt).
```

Streaming consumers need no code change unless they want the new
`finish_key` / `committed_tokens` keys.

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

[Unreleased]: https://github.com/erllama/erllama/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/erllama/erllama/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/erllama/erllama/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/erllama/erllama/releases/tag/v0.1.0
