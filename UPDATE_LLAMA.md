# Updating the vendored llama.cpp

erllama vendors a pinned copy of llama.cpp under `c_src/llama.cpp/`.
The current pin is **b9119**.

This file documents the bump procedure.

## Why pin

- Reproducible builds: every developer and CI run compiles the same
  source.
- Hex.pm-friendly: published packages contain the full source so no
  network access is needed at install time.
- Backend stability: llama.cpp moves fast, especially in the model
  zoo. We control when we adopt new architectures.

## What we ship

We vendor only the parts we need. Currently:

```
c_src/llama.cpp/
  CMakeLists.txt            llama.cpp's top-level CMake
  LICENSE                   MIT (dual MIT/Apache, llama.cpp pick MIT)
  cmake/                    CMake helpers (toolchain files, etc)
  include/                  public headers (llama.h, etc)
  src/                      llama core (model.cpp, context.cpp, etc)
  ggml/
    CMakeLists.txt
    cmake/                  ggml CMake helpers (common.cmake, GitVars.cmake)
    include/                public ggml headers
    src/
      CMakeLists.txt
      ggml*.c, ggml*.cpp, ggml*.h  core ggml + frontends
      gguf.cpp                     GGUF file format
      ggml-cpu/                    CPU SIMD kernels (mandatory)
      ggml-metal/                  Apple GPU backend (Apple Silicon)
      ggml-cuda/                   NVIDIA GPU backend (Linux x86-64)
      ggml-blas/                   BLAS backend (OpenBLAS / Accelerate)
```

Excluded (unused or out-of-scope for v1):

- `tools/`, `examples/`, `tests/`, `docs/`, `models/`, `gguf-py/`,
  `benches/`, `ci/`, `scripts/`, `grammars/`, `vendor/`, `.git/`,
  `.github/`, `AUTHORS`, `.devops/`
- ggml backends we do not link: Vulkan, SYCL, OpenCL, CANN, Hexagon,
  HIP, MUSA, RPC, ZDNN, ZenDNN, Virtgpu, Webgpu, OpenVINO

If a user needs one of the excluded backends they can build erllama
against an unvendored llama.cpp via `git+` rebar dep instead of the
hex package; that path is supported but unsupported in this scaffold.

## Bumping

Pick a tag from <https://github.com/ggml-org/llama.cpp/tags>. Newer
tags are usually fine; check the changelog for breaking C-API changes
to `llama_state_seq_*` (the cache layer depends on those).

```sh
# 1. Clone the new tag into a scratch directory.
cd /tmp
git clone --depth=1 --branch=<TAG> https://github.com/ggml-org/llama.cpp llama.cpp.new

# 2. Sync the parts we vendor.
cd /Users/benoitc/Projects/erllama
rm -rf c_src/llama.cpp
mkdir -p c_src/llama.cpp/ggml/src

cp -r /tmp/llama.cpp.new/{src,include,cmake,CMakeLists.txt,LICENSE} \
      c_src/llama.cpp/
cp -r /tmp/llama.cpp.new/ggml/{include,cmake,CMakeLists.txt} \
      c_src/llama.cpp/ggml/
cp /tmp/llama.cpp.new/ggml/src/CMakeLists.txt \
   c_src/llama.cpp/ggml/src/
cp /tmp/llama.cpp.new/ggml/src/ggml*.c \
   /tmp/llama.cpp.new/ggml/src/ggml*.cpp \
   /tmp/llama.cpp.new/ggml/src/ggml*.h \
   /tmp/llama.cpp.new/ggml/src/gguf.cpp \
   c_src/llama.cpp/ggml/src/
cp -r /tmp/llama.cpp.new/ggml/src/{ggml-cpu,ggml-metal,ggml-cuda,ggml-blas} \
      c_src/llama.cpp/ggml/src/

# 3. Rebuild and run the full test gauntlet.
rm -rf _build
rebar3 compile
rebar3 fmt --check && rebar3 lint && rebar3 xref \
    && rebar3 eunit && rebar3 proper && rebar3 ct

# 4. Update the pin reference in this file and in
#    c_src/llama.cpp/.version (if present).

# 5. Commit with a message naming the new tag.
```

## Configuration knobs

The CMake configure step honours these env vars (passed via
`ERLLAMA_OPTS` to `do_cmake.sh`):

```
ERLLAMA_OPTS="-DGGML_CUDA=ON"           # enable CUDA on Linux x86-64
ERLLAMA_OPTS="-DGGML_METAL=OFF"         # disable Metal on Darwin
ERLLAMA_OPTS="-DGGML_BLAS=OFF"          # disable BLAS
ERLLAMA_OPTS="-DCMAKE_BUILD_TYPE=Debug" # debug build
```

The build step honours `ERLLAMA_BUILDOPTS` (passed to `cmake --build`).

## Why we drop `common/`

llama.cpp's `common/` carries HTTP / Hugging Face download helpers
that pull in cpp-httplib (5 MB). erllama uses the public `llama.h` API
directly and provides its own thin sampling / tokenization helpers in
the NIF; nothing in `common/` is on our critical path.
