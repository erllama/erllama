#!/usr/bin/env bash
# erllama bench driver.
#
# Usage:
#   bench/run.sh                   # all available models
#   bench/run.sh tiny              # only the small model
#   bench/run.sh large             # only the large model
#
# Configure model paths via env vars:
#   LLAMA_BENCH_TINY   default: $HOME/Models/tinyllama-1.1b-chat.gguf
#   LLAMA_BENCH_LARGE  default: unset (skip)
set -euo pipefail
cd "$(dirname "$0")/.."

# Build first so the NIF and modules are current.
rebar3 compile >/dev/null

MODE="${1:-all}"

# Discover ebin paths. Use the default profile rebar3 builds with `compile`.
EBIN_PATHS=$(find _build/default/lib -maxdepth 2 -name ebin -type d | xargs -I {} echo "-pa {}")

# Bench module lives in bench/ (not part of rebar3 src tree). Compile it
# on the fly into _build so its deps resolve via the same paths.
mkdir -p _build/bench/ebin
erlc -o _build/bench/ebin -I include -pa _build/default/lib/erllama/ebin \
    bench/erllama_bench.erl

# shellcheck disable=SC2086
exec erl -noinput -boot start_clean \
    $EBIN_PATHS -pa _build/bench/ebin \
    -s erllama_bench main "$MODE"
