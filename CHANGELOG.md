# Changelog

All notable changes to erllama are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.1.0] — 2026-05-10

Initial public release.

### Added

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
- Grammar-constrained sampling. Pass `grammar => GBNF` in the
  `infer/4` Params map; the per-model sampler chain is rebuilt as
  grammar then greedy for the duration of the request and reset on
  completion or cancellation.
- `erllama_registry` module: ETS-backed `via` callback for binary
  model ids.
- `erllama_inflight` module: `Ref -> ModelPid` table so `cancel/1`
  routes to the right gen_statem.
- `erllama_model_backend` optional callbacks `apply_chat_template/2`,
  `embed/2`, `set_grammar/2`, `clear_sampler/1`. Backends that omit
  them surface `{error, not_supported}` from the public API.
- Decode loop schedules each step via `gen_statem:cast(self(),
  decode_step)` instead of `next_event` so cancel, evict, status, and
  busy rejection interleave fairly between tokens.
- Native Erlang/OTP wrapper around llama.cpp via a single
  dirty-scheduler NIF (`erllama_nif`) covering model load, context
  construction, tokenisation, prefill, single-token decode, and
  KV pack/unpack.
- Token-exact KV cache with three independently-supervised tiers:
  RAM (ETS slabs), `ram_file` (`/dev/shm`), and disk (plain read
  I/O).
- Sole-writer arbitration through `erllama_cache_meta_srv`; reads
  on the hot path go to ETS directly.
- Crash-safe save publish protocol: reserve, write_tmp, check,
  `link(2)`, mark_published; two-stage TTL cleanup with orphan
  adoption.
- Five save reasons (`cold`, `continued`, `finish`, `evict`,
  `shutdown`) with async/sync semantics matching their use.
- Multi-turn warmth via explicit `parent_key` resume and stateless
  longest-prefix walk for OpenAI/Anthropic-shaped clients.
- `erllama_scheduler` memory-pressure poller with pluggable sources
  (`memsup`, `nvidia-smi`, custom callback). Off by default.
- `erllama_cache_writer` poolboy-backed dirty-IO writer pool with
  leak-proof reservation semaphore.
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
- 166 EUnit + 11 PropEr + 7 stub Common Test cases. Real-model
  Common Test suite gated on `LLAMA_TEST_MODEL` (6 cases, passing
  locally with TinyLlama 1.1B Q4_K_M).
- Bench harness (`bench/run.sh`) with cold-vs-warm matrix and a
  4-agent shared-prefix scenario. TinyLlama and LLaMA-3 8B
  presets.
- Multi-platform CI: Ubuntu 24.04 amd64, Ubuntu 24.04 arm64,
  macOS 13 (Intel), macOS 14 (Apple Silicon), FreeBSD 14.4. OTP
  28 across the matrix.

### Documentation

- README rewritten as a friendly entry point with snippets.
- User guides: loading, caching, configuration.
- Internal design notes: cache design, publish protocol, NIF safety.
- ex_doc-friendly module documentation throughout.

### Acknowledgements

The on-disk `KVC` file format (48-byte header, `"KVC"` magic), the
save-reasons taxonomy, and the `boundary_trim_tokens` /
`boundary_align_tokens` defaults are direct ports from
[antirez/ds4](https://github.com/antirez/ds4). ds4 pioneered the
"disk KV cache as a first-class resume mechanism" idea for
DeepSeek V4; erllama generalises that pattern as an Erlang/OTP
library across any GGUF llama.cpp can load.

[Unreleased]: https://github.com/erllama/erllama/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/erllama/erllama/releases/tag/v0.1.0
