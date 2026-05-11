# Changelog

All notable changes to erllama are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.1.0] — 2026-05-11

Initial public release.

### Added

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
- `erllama:unload_model/1` as an alias for `erllama:unload/1` matching
  the OpenAI/Ollama-style naming downstream HTTP servers use.
- `erllama:infer/4` streaming inference. Returns `{ok, Ref}`; tokens
  are delivered to the caller as `{erllama_token, Ref, _}`,
  `{erllama_done, Ref, Stats}`, `{erllama_error, Ref, Reason}`.
- `erllama:cancel/1`. Idempotent and fire-and-forget; observed
  between tokens.
- `erllama:apply_chat_template/2`. Renders a normalised chat request
  (`messages`, `system`, `tools`) through the model's GGUF chat
  template and tokenises. Backed by `llama_chat_apply_template`.
- `erllama:embed/2`. Per-sequence pooled embedding via
  `llama_get_embeddings_seq` with last-token fallback.
- **Sampler parameters.** `complete/3` and `infer/4` honour
  `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
  `seed`, and `grammar` via one combined chain builder
  (`erllama_nif:configure_sampler/2`). Chain order:
  `grammar -> repetition_penalty -> top_k -> top_p -> min_p ->
  (temperature > 0 ? temp -> dist(seed) : greedy)`. `set_grammar/2`
  retained as a backwards-compatible alias.
- **LoRA adapters.** `erllama:load_adapter/2`,
  `unload_adapter/2`, `set_adapter_scale/3`, `list_adapters/1`.
  Per-adapter sha256 + scale fold into the cache via
  `erllama_cache_key:effective_fingerprint/2`, so rows produced
  with adapter A never collide with rows from adapter B. Snapshot-
  at-admission semantics keep in-flight requests on their original
  fingerprint even if an adapter mutation arrives mid-generation.
- **Concurrent request queue.** A second `complete/3` or `infer/4`
  arriving while one is in flight is queued FIFO instead of getting
  `{error, busy}`. The reply `{ok, Ref}` is sent as soon as the
  call is admitted; streaming events follow once the queue head
  advances to the request.
- **Seq-aware NIFs (infrastructure for 0.2 multi-seq batching).**
  `nif_kv_pack/4` accepts an explicit `seq_id`; new
  `erllama_sampler_t` resource owning a standalone
  `llama_sampler*`; `nif_sampler_new/2`, `nif_sampler_free/1`.
  Cache rows stay seq-id-free: `seq_id` is a save/load call
  argument, never row metadata.
- Grammar-constrained sampling. Pass `grammar => GBNF` in the
  `complete/3` Opts or `infer/4` Params; the per-model sampler chain
  is rebuilt as grammar then greedy for the duration of the request
  and reset on completion or cancellation.
- `erllama_registry` module: ETS-backed `via` callback for binary
  model ids.
- `erllama_inflight` module: `Ref -> ModelPid` table so `cancel/1`
  routes to the right gen_statem.
- `erllama_model_backend` optional callbacks: `apply_chat_template/2`,
  `embed/2`, `set_grammar/2`, `configure_sampler/2`,
  `clear_sampler/1`, `load_adapter/2`, `unload_adapter/2`,
  `apply_adapters/2`. Backends that omit them surface
  `{error, not_supported}` from the public API.
- Decode loop schedules each step via `gen_statem:cast(self(),
  decode_step)` instead of `next_event` so cancel, evict, status, and
  queued requests interleave fairly between tokens.
- Token-exact KV cache with three independently-supervised tiers:
  RAM (ETS slabs), `ram_file` (`/dev/shm`), and disk (plain read
  I/O).
- Sole-writer arbitration through `erllama_cache_meta_srv`; reads
  on the hot path go to ETS directly. `lookup_exact/1` is a single
  atomic `ets:lookup` (no two-call race) and the meta server cancels
  waiter timers when an early reply lands so the mailbox doesn't
  bloat under load.
- Crash-safe save publish protocol: reserve, write_tmp, check,
  `link(2)`, mark_published; two-stage TTL cleanup with orphan
  adoption.
- Five save reasons (`cold`, `continued`, `finish`, `evict`,
  `shutdown`) with async/sync semantics matching their use.
- `saves_dropped` counter: bumps whenever a back-pressured writer
  pool refuses a save the model wanted to fire.
- Multi-turn warmth via explicit `parent_key` resume and stateless
  longest-prefix walk for OpenAI/Anthropic-shaped clients.
- `erllama_scheduler` memory-pressure poller with pluggable sources
  (`memsup`, `nvidia-smi`, custom callback). Off by default.
- `erllama_cache_writer` dirty-IO writer pool with a leak-proof
  reservation semaphore.
- Persisted hit counters (u32 in disk header) so popular prefixes
  survive an LRU walk after restart.
- End-to-end metrics: hits/misses/saves/evictions plus per-path
  latency totals (`pack_total_ns`, `load_total_ns`,
  `longest_prefix_ns`, `longest_prefix_probes`).
- Per-resource `pthread_mutex` and two-resource lifetime pattern
  for safe concurrent `free_*/1` plus dirty NIF ops.
- `extern "C" noexcept` shim catching every llama.cpp C++
  exception at the boundary; `decode_one` defensive guard against
  `GGML_ASSERT` aborts.
- Bench harness (`bench/run.sh`) with cold-vs-warm matrix and a
  4-agent shared-prefix scenario. TinyLlama and LLaMA-3 8B
  presets.

### Changed

- **NIF hardening.**
  - `llama_backend_init` is deferred to the first `nif_load_model`
    via `pthread_once`. Cache-only and unit-test workloads no
    longer pay the `ggml_backend_load_all` cost at NIF load.
  - `nif_tokenize` and `nif_detokenize` honour `release_pending`,
    so a model returned by `free_model/1` as `{ok, deferred}`
    cannot be reused via tokenize.
  - `nif_detokenize` fails closed on `n_vocab <= 0` (matches
    `nif_prefill`); 16 M-token cap on the size computation
    silences gcc `-Walloc-size`.
  - Dead `atom_not_implemented` and the `_unused_anchor` hack
    removed (`-Wpedantic`).
  - `make_errno_atom` now maps FreeBSD's `EINTEGRITY` to
    `eintegrity` instead of `unknown`.
- **Cache meta server.** Sweep timer is cancelled on `terminate/2`
  so a supervisor restart never leaves a zombie firing into a fresh
  server. `pin_and_load/2` wraps load + unpack in `try/after` so
  the holder is always checked in.
- **Build.**
  - `FindErlang.cmake` adopted from erlang-rocksdb; replaces the
    inline `erl -noshell -eval` snippet that detected
    `ERTS_INCLUDE_DIR`.
  - `set(GGML_CCACHE OFF CACHE BOOL "" FORCE)` silences the
    "ccache not found" diagnostic ggml emits on every build.
- **Scheduler tests.** Three bad-config cases now call
  `erllama_scheduler:validate_config/1` directly (now exported)
  instead of spawning a `gen_server` with bad config, so eunit no
  longer prints `=CRASH REPORT=` SASL output.
- **Dialyzer.** `response_target` typed `non_neg_integer()` (idle
  state holds 0); `inet:gethostname` and `application:get_key`
  pattern-match directly. `erllama.erl` moduledoc fence closer
  fixed.

### Removed

- **mmap from the disk tier.** The cache now reads files via plain
  `file:read_file/1` into a fresh BEAM heap binary; the `iommap`
  dependency, the `disk_io` configuration option, and the
  4-arity `start_link` form of `erllama_cache_disk_srv` are gone.
  The process already mmaps multi-GB GGUF weights, and a region
  binary that outlived its closing NIF call would have exposed the
  BEAM to SIGBUS from any external truncation.

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
  application — a separate OTP project that coordinates a fleet of
  erllama nodes (request distribution, cross-node speculative
  decoding, pipeline parallelism over QUIC).

### Acknowledgements

Same idea as [antirez/ds4](https://github.com/antirez/ds4).

[Unreleased]: https://github.com/erllama/erllama/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/erllama/erllama/releases/tag/v0.1.0
