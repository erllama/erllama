# Configuration reference

erllama configuration lives in two places: the OTP application
environment (`config/sys.config`) and the per-model option map
passed to `erllama:load_model/1,2`. This page is the full set.

## Application environment

```erlang
{erllama, [
  %% --------------- Tiers -----------------------------------------
  {tiers, [
    #{backend => ram,                              quota_mb => 4096},
    #{backend => {ram_file, "/dev/shm/erllama"},   quota_mb => 16384},
    #{backend => {disk, "/var/lib/erllama/kvc"},   quota_mb => 65536}
  ]},

  %% --------------- Save-policy gates -----------------------------
  {min_tokens,             512},
  {cold_min_tokens,        512},
  {cold_max_tokens,      30000},
  {continued_interval,    2048},
  {boundary_trim_tokens,    32},
  {boundary_align_tokens, 2048},

  %% --------------- Cache flow tunables ---------------------------
  {evict_save_timeout_ms,  30000},
  {session_resume_wait_ms,   500},
  {fingerprint_mode,         safe},   %% safe | gguf_chunked | fast_unsafe

  %% --------------- Memory-pressure scheduler ---------------------
  {scheduler, #{
    enabled         => false,
    pressure_source => noop,
    interval_ms     => 5000,
    high_watermark  => 0.85,
    low_watermark   => 0.75,
    min_evict_bytes => 1048576,
    evict_tiers     => [ram, ram_file]
  }}
]}.
```

### `tiers`

A list of tier specs, in the order erllama should consult them. Each
entry is a map:

| Key | Type | Notes |
|---|---|---|
| `backend` | `ram` \| `{ram_file, Path}` \| `{disk, Path}` | Tier kind. |
| `quota_mb` | non_neg_integer | Soft byte budget. The tier evicts to keep itself under this. |
| `name` | atom | Optional registered name; defaults to `default_<kind>`. |

The first tier is preferred for new saves unless the per-model
`tier` option overrides.

### Save-policy gates

See the [caching guide](caching.md#save-policy-gates) for what each
threshold does. All are overridable per-model via the `policy` map.

### `evict_save_timeout_ms`

How long synchronous `evict` and `shutdown` saves wait for the
writer to finish before giving up. Defaults to 30 s. Bump for
8B-class models on slow disks.

### `session_resume_wait_ms`

When a `parent_key` is supplied and the cache sees a matching
in-flight finish save, it waits up to this long for the save to
publish before falling through to a cold prefill. 500 ms is enough
for SSD-backed deployments; bump if you observe back-to-back
multi-turn cold misses on slow storage.

### `fingerprint_mode`

How to verify the model fingerprint at load:

- `safe` — full SHA-256 over the file. Slow on multi-GB GGUFs.
- `gguf_chunked` — fingerprint metadata + first weights tensor.
  Catches accidental corruption, not malicious tampering.
- `fast_unsafe` — trust the supplied fingerprint blindly. Use only
  when you fingerprint upstream and pass the digest through.

### `scheduler`

See the [caching guide](caching.md#memory-pressure-driven-eviction).

## Per-model options

Passed to `erllama:load_model/1,2`:

```erlang
#{
  backend           => erllama_model_llama,
  model_path        => "/path/to/x.gguf",
  model_opts        => #{n_gpu_layers => 99},
  context_opts      => #{n_ctx => 4096, n_batch => 512},
  fingerprint       => <<32 bytes>>,
  fingerprint_mode  => safe,
  quant_type        => q4_k_m,
  quant_bits        => 4,
  ctx_params_hash   => <<32 bytes>>,
  context_size      => 4096,
  tier_srv          => default_disk,
  tier              => disk,
  policy            => #{
    min_tokens             => 256,
    cold_min_tokens        => 256,
    cold_max_tokens        => 8192,
    continued_interval     => 256,
    boundary_trim_tokens   => 32,
    boundary_align_tokens  => 256,
    session_resume_wait_ms => 500
  }
}
```

See [loading a model](loading.md) for the per-field walkthrough.

## Inspecting effective config

```erlang
1> application:get_env(erllama, scheduler).
{ok, #{enabled => true, ...}}

2> erllama_scheduler:status().
#{enabled => true, pressure_source => system, ...}

3> erllama_cache_meta_srv:dump().
[#{key => <<...>>, tier => disk, size => 8388608, ...}]
```
