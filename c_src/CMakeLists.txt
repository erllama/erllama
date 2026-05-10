# Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
# See the LICENSE file at the project root.

cmake_minimum_required(VERSION 3.20)
project(erllama_nif C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# llama.cpp configuration
# -----------------------------------------------------------------------------
# Build llama.cpp as static libraries that we link into our NIF .so.
# Backend selection is platform-driven by default; users override via
# environment variables consumed in do_cmake.sh.

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)

# We link the ggml/llama static libs into a shared NIF .so. On Linux,
# the system libgomp.a is built without -fPIC, so an OpenMP-enabled
# static ggml-cpu drags in a non-PIC libgomp that fails to relocate
# into our .so ("R_X86_64_TPOFF32 against hidden symbol gomp_tls_data
# can not be used when making a shared object"). Disabling OpenMP at
# the ggml level avoids the libgomp dependency entirely. The cost is
# CPU-path parallelism inside ggml; the GPU paths (Metal on Apple,
# CUDA on Linux when enabled) are unaffected.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

# Silence "ccache not found" diagnostic from ggml when ccache is
# not installed on the build host (FreeBSD CI VMs, fresh dev boxes).
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

# Backends we never ship in the vendored tree (sources are not present).
set(GGML_VULKAN  OFF CACHE BOOL "" FORCE)
set(GGML_SYCL    OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL  OFF CACHE BOOL "" FORCE)
set(GGML_CANN    OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_HIP     OFF CACHE BOOL "" FORCE)
set(GGML_MUSA    OFF CACHE BOOL "" FORCE)
set(GGML_RPC     OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN    OFF CACHE BOOL "" FORCE)

# Backends that ship; defaults can be flipped via -DGGML_*=OFF on the
# CMake command line (see do_cmake.sh).
if(APPLE)
    if(NOT DEFINED GGML_METAL)
        set(GGML_METAL ON CACHE BOOL "" FORCE)
    endif()
    if(NOT DEFINED GGML_BLAS)
        set(GGML_BLAS ON CACHE BOOL "" FORCE)
        set(GGML_BLAS_VENDOR "Apple" CACHE STRING "" FORCE)
    endif()
endif()

if(NOT DEFINED GGML_CUDA)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)

# -----------------------------------------------------------------------------
# Erlang ERTS include dir (via FindErlang.cmake, copied from
# erlang-rocksdb). Honour ERTS_INCLUDE_DIR if the caller pre-set it.
# -----------------------------------------------------------------------------
if(DEFINED ENV{ERTS_INCLUDE_DIR})
    set(ERTS_INCLUDE_DIR $ENV{ERTS_INCLUDE_DIR})
else()
    include(FindErlang)
    set(ERTS_INCLUDE_DIR ${ERLANG_ERTS_INCLUDE_PATH})
endif()

# -----------------------------------------------------------------------------
# erllama_nif shared library
# -----------------------------------------------------------------------------
add_library(erllama_nif SHARED
    erllama_nif.c
    erllama_safe.cpp
    crc32c.c
)

target_compile_options(erllama_nif PRIVATE
    -O2 -g -Wall -Wextra -Wpedantic -fPIC
)

target_include_directories(erllama_nif PRIVATE
    ${ERTS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

# pthread is used by erllama_nif.c for per-resource mutexes; depend
# on the standard CMake Threads find module so toolchains that need
# -pthread (e.g. some Linux distros) get it added.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(erllama_nif PRIVATE
    llama
    ggml
    Threads::Threads
)

# Erlang NIFs need this dance for symbol resolution.
set_target_properties(erllama_nif PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "erllama_nif"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../priv"
)

if(APPLE)
    target_link_options(erllama_nif PRIVATE
        -undefined dynamic_lookup
        -flat_namespace
    )
endif()
