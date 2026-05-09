# erllama

Native Erlang/OTP wrapper around llama.cpp providing OpenAI-compatible
inference with full supervision, tiered KV caching, and an Erlang-first
deployment model.

Status: scaffolding only. See the design plan referenced below for the
roadmap; no business logic has shipped yet.

## Layout

```
src/                    Erlang sources (single OTP app, flat)
include/                shared headers
c_src/                  the single NIF (erllama_nif)
test/                   eunit + PropEr tests
priv/                   build artefact: erllama_nif.so
config/sys.config       runtime configuration
IOMMAP_PREREQUISITE.md  upstream prerequisite for the iommap zero-copy primitive
```

## Build

Requires Erlang/OTP 28 and rebar3 3.25+ (see `.tool-versions`).

```
rebar3 compile
rebar3 shell
```

## Roadmap

The implementation roadmap is documented in the design plan (rev 10) at
`/Users/benoitc/.claude/plans/golden-finding-horizon.md`. Step 0 is the
current state of the tree (scaffolding). Step 1 (the iommap upstream
prerequisite) is captured as a self-contained prompt in
`IOMMAP_PREREQUISITE.md`.

## License

Apache-2.0
