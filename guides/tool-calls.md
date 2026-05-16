# Tool calls

Some chat templates emit structured tool calls inline with normal model
text. qwen3-style `<tool_call>...</tool_call>`, DSML, and XML-shaped
templates all have the same practical problem: the HTTP layer needs to
parse a tool call for the client, then put the exact same bytes back into
the next prompt so the KV cache can still hit.

erllama handles the model-facing half of that problem. It detects
configured tool-call boundaries, streams the exact bytes the model sampled,
and can make the syntactic wrapper deterministic while leaving payload
text free to sample normally.

It does not mint tool ids, parse SDK JSON, or canonicalize tool arguments.
Those belong in the server layer above erllama.

## Declaring markers

Markers are per model because every chat template is different:

```erlang
erllama:load_model(<<"qwen3-tool">>, #{
    backend => erllama_model_llama,
    model_path => "/models/qwen3-7b.gguf",
    tool_call_markers => #{
        start => <<"<tool_call>">>,
        end => <<"</tool_call>">>
    }
}).
```

The backend tokenizes the marker strings through the model's own
vocabulary at load time. Multi-token markers work because detection is by
token sequence, not by a byte scan over detokenized text.

Omitting `tool_call_markers` keeps the model on the normal text path: no
tool-call messages and no sampler swap.

## Streaming shape

A streaming `infer/4` against a tool-aware model receives:

```erlang
{erllama_token, Ref, {tool_call_delta, Bin}}
{erllama_tool_call_end, Ref, FullBin}
```

`tool_call_delta` contains the bytes inside the tool-call span. At the end
of the span, `FullBin` is the concatenation of every delta. Store that
binary under whatever tool id your HTTP layer minted.

On the next turn, splice the stored binary back into the rendered prompt.
Do not pretty-print it. Do not canonicalize it if you still have the exact
bytes. Exact replay is what lets the next prompt match the cached token
prefix.

## Deterministic syntax

When `tool_call_markers` are configured, erllama builds a second sampler
chain for the request with `temperature = 0`. When the start marker is
sampled, the scheduler swaps the request onto that greedy chain so the
tool-call syntax stays stable from a fixed prefix.

If the template has inner markers for payload bodies, declare them too:

```erlang
tool_call_markers => #{
    start => <<"<tool_call>">>,
    end => <<"</tool_call>">>,
    payload_start => <<"<arg>">>,
    payload_end => <<"</arg>">>
}
```

With payload markers set, erllama uses the greedy sampler for the wrapper
syntax and the request's normal sampler for payload bytes. That keeps large
string arguments, file contents, or generated edits from becoming
unnecessarily deterministic.

Without payload markers, the whole tool-call span uses the greedy sampler.
That is simpler and works well for short literal arguments.

## Server responsibilities

The downstream server is responsible for:

- Minting a tool id.
- Storing `FullBin` under that id.
- Parsing `FullBin` into the SDK's `tool_use` or tool-call JSON shape.
- Falling back to deterministic rendering when the id is unknown.
- Replaying stored bytes verbatim into later prompts when the id is known.

erllama provides adjacent primitives that help compose the full flow:

- `erllama:prefill_only/3` with `parent_key` can warm a prompt prefix
  before generation.
- `session_id` on `infer/4` and `complete/3` can pin a live sequence
  across turns.
- `erllama:end_session/2` releases a pinned sequence explicitly.

See [Examples](examples.md#11-tool-call-streaming-tool_call_markers) for
copyable streaming snippets.
