# Request lifecycle

This is the contributor-facing walkthrough of a single inference request.
The public usage patterns live in [Examples](../guides/examples.md); this
page explains how admission, cache restore, batching, decode, and saving fit
together inside the model process.

## Model process states

Each loaded model is a supervised `gen_statem` with one underlying
`llama_context*`. It has two broad runtime states:

- `idle` when no request is active.
- `running` when one or more requests are prefilling, decoding, or queued
  for the multi-sequence scheduler.

The default context has `n_seq_max => 1`, which keeps single-request
behavior simple and deterministic. Setting `context_opts.n_seq_max > 1`
opts into concurrent prefill/decode across multiple active sequences.

## Admission

Every request starts by taking a sequence id from the model's idle pool. If
the caller supplied a `session_id`, the model tries to reuse that session's
pinned sequence instead. If the session is already active in another
request, admission returns `{error, sticky_busy}`.

At admission, the model also prepares request-local state: token cursor,
sampling options, optional grammar, optional thinking/tool-call markers, and
the reply destination for streaming calls.

## Cache resolution

The admitted request resolves its warm prefix before any decode work starts.
The result is reported as `cache_hit_kind`:

| Kind | Meaning |
|---|---|
| `exact` | The supplied key or computed prompt key matched the full prompt. |
| `partial` | A shorter exact prefix matched and the suffix must be prefed. |
| `cold` | No usable prefix was found. |
| `sticky` | A pinned session sequence already contains the prefix live. |

The lookup order is:

1. Supplied `parent_key`, if present.
2. Resume from an in-flight finish save, waiting up to
   `session_resume_wait_ms`.
3. Longest-prefix walk over the new prompt's tokens.

The longest-prefix walk is still exact. It probes aligned token prefixes and
uses the longest row that actually exists.

## Restore or prefill

After cache resolution, the request takes one of three paths:

- **Warm restore.** `kv_unpack` restores the saved KV bytes into the
  sequence. The model then prefills the last token again as a logits primer
  so generation starts from fresh logits.
- **Cold prefill.** The sequence is cleared and the full prompt is prefed.
- **Sticky continuation.** The live sequence already holds the prefix, so
  the model truncates or extends in place and only prefills the new suffix.

Long prompts are sliced by `prefill_chunk_size`, defaulting to
`max(64, n_batch div 4)`, so one large prefill cannot monopolize every
scheduler tick when other sequences are decoding.

## Decode

The scheduler batches active rows into one `llama_decode` call per tick,
bounded by `n_batch`. A tick can contain a mix of prefill tokens and decode
tokens across different sequences.

Request-local samplers stay isolated. One request can use a grammar or a
fixed seed while another uses a different temperature. Tool-call and
thinking markers are recognized inline and surfaced on the streaming wire
without changing unrelated requests.

## Save points

The model fires cache saves at stable boundaries:

| Reason | Trigger |
|---|---|
| `cold` | After cold prefill reaches the trimmed prefix boundary. |
| `continued` | Every configured generated-token interval. |
| `finish` | When generation completes. |
| `evict` | When a holder must release a slab under pressure. |
| `shutdown` | During unload or model process stop. |

Async saves go through `erllama_cache_writer`. Synchronous saves block the
caller because the backing resource is about to be released.

Disk publication is handled by the reservation protocol described in
[Publish protocol](publish-protocol.md).

## Completion

When a request finishes, the model reports final stats, including:

- `finish_reason`
- `finish_key`
- `cache_hit_kind`
- `cache_delta`
- cancellation state, when applicable

One-shot requests release their sequence back to the idle pool. Sticky
requests keep the sequence pinned until a later request diverges or the
caller invokes `erllama:end_session/2`.

## Observability

Routers can inspect model state without crossing the model `gen_statem`:

```erlang
erllama:phase(ModelId).
erllama:pending_len(ModelId).
erllama:queue_depth(ModelId).
erllama:last_cache_hit(ModelId).
```

These read a public ETS row and are safe to call from a hot path.
