# Agents

Instructions for AI coding agents working on this project.

## Project Overview

erllama is a native Erlang/OTP wrapper around llama.cpp providing
OpenAI-compatible inference with full supervision and a tiered KV
cache. Requires Erlang/OTP 28 and rebar3 3.25+.

Single application, flat layout:

```
src/        Erlang sources (erllama_*, erllama_cache_*, erllama_nif)
include/    Shared headers (erllama_cache.hrl)
c_src/      C sources for the single NIF (erllama_nif.so)
test/       eunit + PropEr property tests
priv/       Build artefact: erllama_nif.so
config/     sys.config
```

The KV cache is logically a subsystem (its own supervisor, modules
prefixed `erllama_cache_*`) but lives in the same OTP application as
the rest of erllama. There is one NIF (`erllama_nif`) that holds the
entire native surface (cache pipeline + future llama.cpp wrappers).

Authoritative behaviour is encoded in the test suites under `test/`
(EUnit, PropEr, and Common Test) and the module docstrings. The
README has the public-API tables and configuration reference.

## Required Checks

Every change must be formatted and pass all checks before committing:

```bash
rebar3 fmt          # Auto-format (always run first)
rebar3 compile      # Must compile cleanly (warnings_as_errors)
rebar3 eunit        # Unit tests
rebar3 proper       # Property tests
rebar3 ct           # Common Test suites
rebar3 lint         # Elvis linter
rebar3 dialyzer     # Type checking
rebar3 xref         # Cross-reference analysis
```

## Build & Development Commands

```bash
rebar3 compile                                    # Build
rebar3 shell                                      # Boot the umbrella
rebar3 eunit                                      # All EUnit tests
rebar3 eunit --module=erllama_cache_kvc_tests     # Specific test module
rebar3 proper                                     # All PropEr property-based tests
rebar3 ct --suite=erllama_cache_meta_SUITE        # Specific Common Test suite
rebar3 fmt                                        # Auto-format (erlfmt)
rebar3 fmt --check                                # Format check, no writes
rebar3 lint                                       # Elvis linter
rebar3 dialyzer                                   # Type checking
rebar3 xref                                       # Cross-reference
rebar3 ex_doc                                     # Generate docs
```

## Architecture

### Cache subsystem (`erllama_cache_*` modules)

```
erllama_cache_sup
├── erllama_cache_meta_srv     sole writer for meta + LRU + reservations
├── erllama_cache_ram          RAM tier (ETS slab store)
├── erllama_cache_ramfile_sup
│   └── erllama_cache_ramfile_srv  per ram_file root dir
├── erllama_cache_disk_sup
│   └── erllama_cache_disk_srv     per disk root dir (plain read/write)
└── erllama_cache_writer_pool  poolboy: dirty-IO save workers
```

Public API lives in `erllama_cache.erl` (a stateless facade). Hot-path
read lookups go through ETS directly. Writes (claim, release, evict,
save announce) go through `erllama_cache_meta_srv` via
`gen_server:call`.

### Save pipeline correctness invariants (do not change without review)

- Cache hits are token-exact by construction. The cache key includes
  model fingerprint, quant byte, ctx hash, and the full token list as
  little-endian u32. Approximate match is **not** allowed at this
  layer.
- A save's payload is read from a paused live `llama_context*`. The
  context worker pauses decode for the pack window; no off-thread
  reads of the live context occur.
- Disk publication is via `link(2)` (atomic create-if-not-exists),
  preceded by a meta-server reservation and a `check_reservation`
  immediately before link to defeat stale-writer races. EEXIST is
  validated and either adopted or replaced under the current
  reservation; never silently skipped.
- Disk reads use plain `file:read_file/1` into a fresh BEAM heap
  binary. mmap is deliberately avoided: the process already mmaps
  multi-GB GGUF weights, so a second mapping per cache restore
  doubles the VM footprint, and a region binary surviving the NIF
  call would expose the BEAM to SIGBUS from any external truncation.

### Multi-turn warmth

v1 has no semantic candidate proposer. Multi-turn warmth is exact and
deterministic: the session layer holds the previous turn's cache key
in its state and passes it as `parent_key` on the next request. The
cache uses `lookup_exact_or_wait/2` (default 500 ms) to wait for an
in-flight finish save to publish before falling through to cold.

The canonical pattern is **claim, unpack, checkin** (in that order):
the holder is released before generation, so the slab returns to
`refcount=0` and is evictable while the user reads the streamed
response.

### Test Organization

- `test/<mod>_tests.erl`: EUnit unit tests
- `test/prop_<mod>.erl`: PropEr property-based tests
- `test/<feature>_SUITE.erl`: Common Test suites

### Real-model CT suite

`erllama_real_model_SUITE` exercises the llama.cpp backend against a
real GGUF file. Disabled unless `LLAMA_TEST_MODEL` points at a valid
model:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-q4_k_m.gguf rebar3 ct \
    --suite=test/erllama_real_model_SUITE
```

Without the env var the suite skips so default `rebar3 ct` runs stay
green on CI without model files.

## Linting Notes

Elvis rules are configured in `elvis.config`. The set is intentionally
lean at v0; per-module ignores will be added as the codebase grows.
The atom naming regex allows `_SUITE` suffix for CT suites:
`^[a-z](_?[a-zA-Z0-9]+)*(_SUITE)?$`. Max line length is 120.

## Coding conventions

- Default to writing no comments. Only annotate non-obvious *why* (a
  hidden constraint, an invariant, a workaround). Don't restate what
  the code does.
- Erlang/OTP idioms: `gen_statem`, `gen_server`, `supervisor`, `ETS`.
  No magic, no DSL wrappers.
- ETS reads are hot-path; ETS writes are funnelled through one owner
  process per table.
- NIFs run on dirty schedulers (CPU or IO as appropriate). No NIF
  performs file I/O directly; framing and validation live in Erlang
  code that calls a small set of pure-data NIFs (`kv_pack`,
  `kv_unpack`, `crc32c`).
- Configuration validation runs at supervisor `init/1`; misconfig is
  a hard `start_link` error, not a runtime warning.

## What to avoid

- No `iolist_to_binary` flattening of multi-GB payloads. Use iolists
  for `prim_file:write/2`.
- No `ets:select_replace/2` on the hot path; the meta server is the
  arbitration authority.
- No silent EEXIST handling on link; always validate-and-adopt or
  replace.
- No reliance on `enif_resource_refcount`; use the two-resource
  lifetime pattern (handle resource holding a mapping resource via
  `enif_keep_resource`).
- No semantic candidate proposer in v1 (deferred to v2).
- No KV state compression in v1 (TurboQuant is unproven for that;
  generic lz4/zstd is a future option).

## When in doubt

Re-read the test suite for the area you're touching — every
non-obvious invariant in the cache publish protocol, the
reservation state machine, the warm-restore logits primer, and the
NIF safety wrappers has a dedicated case. Surface tension with
existing tests to the human reviewer before changing the
behaviour.
