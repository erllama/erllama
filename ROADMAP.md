# Roadmap

What erllama does not do yet, with rough scope and rationale for each
item. Issues / PRs welcome.

## Deferred from 0.1 to 0.2

### Concurrent multi-sequence decoding

Today the model gen_statem serves one request at a time; a second
`complete/3` or `infer/4` arriving while one is in flight queues
behind it (Phase 4 scoped, 0.1) and waits for the previous to
finish.

For real concurrency we need to drive multiple `seq_id`s through one
`llama_decode` call: per-request `#req{}` records (extracted from
`#data{}`), a `seq_id` allocator pool, a per-request sampler
resource (already shipping in 0.1 as `erllama_sampler_t`), and a
new `nif_decode_and_sample_batch/2` primitive.

Verification: PropEr property running N concurrent completions on
one model; cancellation evicts the right `seq_id`; counters show
batched decode steps.

## Backlog (no fixed milestone)

### Speculative decoding

Pair a small draft model with a target model; speculate-and-verify
to improve throughput. Needs a "verify N tokens at offset" NIF and
a draft-model registry. The KV cache layer is largely orthogonal
(verifications run on the target context).

### Vision / LLaVA

`llama.cpp` supports the LLaVA family via `llava_init_from_*` and
`llava_eval_image_embed`. The Erlang surface needs an
`apply_image/2` callback on the backend, an embed-cache integration
(so re-uploaded images don't re-tokenize), and chat-template
extensions for multi-modal messages.

### Audio (Whisper)

A different model class than the GGUF chat models 0.1 targets;
`whisper.cpp` has its own context shape. Could be a sister
application (`erllama_whisper`) sharing the cache subsystem.

### Non-GGUF model loading

ONNX, safetensors, raw PyTorch checkpoints. llama.cpp doesn't load
these natively, so the path is either a converter step at
`fetch`-time (the `erllama_server` repo handles fetch) or a second
backend that targets a different runtime.

### Stop-sequence support

`infer_params()` already lists `stop :: [binary()]`; the model
ignores it. Implementation needs a small stop-state machine over
emitted tokens, plus de-tokenisation of the stop strings against
the live vocab.

### Stateful streaming with bit-exact KV resume

Today's warm restore re-prefills the last KV cell to regenerate
logits, which can shift a near-tied sample. A turn-boundary save
that persists the sampler+RNG state alongside the KV cells would
make multi-turn replies bit-identical to the unbroken stream.

### Persistent sampler chain state across turns

If a chain carries internal state (e.g., `repetition_penalty`'s
sliding window), the current design rebuilds it per request.
Persisting it through the cache so a multi-turn resume picks up the
exact same sampler internal state is a 0.2 nice-to-have.

### Telemetry / OTel hooks

Counters today are bare atomics. A telemetry-style event surface
(`telemetry:execute([erllama, complete, start], ...)`) would let
operators wire Prometheus, OTel, statsd without forking the metrics
module.

### Memory-pressure NVIDIA-multi-GPU

`erllama_pressure_nvidia_smi` reads `nvidia-smi` once and sums; a
real multi-GPU deployment wants per-GPU pressure with per-context
eviction. The single-source pressure model in 0.1 collapses to "all
GPUs together".

### TurboQuant / KV state compression

KV state is bulky (~1 GB for 30k tokens on a 70B model). Generic
lz4/zstd helps a little; TurboQuant is unproven for this. We have no
benchmark data we trust enough to ship a default; both stay
deferred.

### Cluster / distributed inference

A model loaded on node A served from node B. The cache subsystem is
node-local; cross-node cache sharing (via the disk tier on a shared
filesystem, or a small announce protocol) is a 0.3+ topic.
