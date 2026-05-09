#!/bin/sh -x
# CMake build step: compile the vendored llama.cpp + erllama_nif.so.
# Mirrors erlang-rocksdb's do_rocksdb.sh pattern.

cd _build/cmake

if type cmake3 > /dev/null 2>&1 ; then
    CMAKE=cmake3
else
    CMAKE=cmake
fi

case "$@" in
    *-j*)
        ${CMAKE} --build . -- "$@" || exit 1
        ;;
    *)
        CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)

        if [ "x$CORES" = "x" ]; then
            PAR=""
        else
            PAR="-- -j $CORES"
        fi
        ${CMAKE} --build . $PAR $@ || exit 1
        ;;
esac

echo done.
