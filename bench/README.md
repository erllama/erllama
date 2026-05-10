# erllama bench

Two scenarios, runnable against any GGUF llama.cpp can load:

1. **`cold_vs_warm`** — single agent, prompt lengths 512 / 1024 / 2048
   tokens. Measures total `complete/3` time on the cold path then the
   warm path; reports the speedup ratio.
2. **`multi_agent`** — pre-warm one shared system-prompt prefix (~1024
   tokens), then run 4 concurrent agents that each append a distinct
   per-agent task and complete a 1-token response. Reports min /
   p50 / p99 / max / mean latency across the agents and dumps the
   cache counters (`hits_*`, `misses`, `longest_prefix_*`,
   `load_total_ns`).

## Run

```bash
bench/run.sh              # all configured models
bench/run.sh tiny         # only the small one
bench/run.sh large        # only the large one
```

Models are selected by env var, with `tiny` defaulting to
`$HOME/Models/tinyllama-1.1b-chat.gguf`:

```bash
LLAMA_BENCH_TINY=/path/to/tinyllama-1.1b-chat.gguf \
LLAMA_BENCH_LARGE=/path/to/llama-3.1-8b-instruct.Q4_K_M.gguf \
  bench/run.sh all
```

A model whose path is unset or doesn't resolve to a file is skipped
with a stderr note. The bench creates a per-run tmp directory under
`$TMPDIR` for the disk tier and removes it on exit.

## What to compare

These are **internal** comparisons by default — they let you see
the cache speedup on a single binary. For external comparisons:

| Target | What it tells you |
|---|---|
| `bench/run.sh tiny` cold column | Equivalent to running raw `llama.cpp llama-cli` on the same prompt — that's the prefill cost the cache amortises. |
| `bench/run.sh tiny` warm column | Cache-restored next-token latency. |
| `multi_agent` p50/p99 | Latency under shared-prefix concurrency, the agent-loop scenario. |

For an apples-to-apples run against unwrapped llama.cpp, time
`./c_src/llama.cpp/build/bin/llama-cli -m $MODEL -n 1 -p '<same prompt>'`
and compare against the cold column. The warm column / cold column
ratio is the headline cache benefit.

## Caveats

- TinyLlama on CPU on M-series Macs gives a useful but not
  representative baseline; for production-shape numbers run the
  large variant on a Mac Studio or comparable.
- The first `complete/3` on a freshly-started Erlang node pays the
  llama.cpp model-load cost (seconds for a multi-GB GGUF). The
  bench's pre-warm covers that for `multi_agent`; `cold_vs_warm`'s
  cold column includes it implicitly for the first prompt-length
  bucket only.
- `n_gpu_layers => 0` keeps things on CPU for repeatability across
  machines. Override at the source if you want Metal/CUDA numbers.
- **`cold_vs_warm` reuses the same llama_context across the cold
  and warm call.** With short prompts (~512 tokens) the warm path
  is the expected ~13× faster. With longer prompts (>=1024 tokens)
  the speedup collapses to ~1× — this is *not* a cache failure
  (the `lp` column confirms the longest-prefix path is hitting),
  it's a llama-side cost: `kv_unpack` into a context that already
  has the prior call's KV cells in seq 0 isn't a clean
  reset+restore. The realistic agent-loop scenario has each agent
  on its own context — that's what `multi_agent` measures, and
  there the speedup over cold is the headline number. A
  `cold_vs_warm_fresh_context` follow-up that allocates a new
  context per warm call would isolate this further.
