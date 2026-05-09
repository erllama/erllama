# NIF build for erllama_kv_nif (CRC32C; kv_pack/kv_unpack stubs).
#
# Invoked from the umbrella rebar.config pre/post hooks. Lands the
# shared library at apps/erllama_cache/priv/erllama_kv_nif.so so that
# `code:priv_dir(erllama_cache)` finds it after rebar3 compile.

UNAME := $(shell uname -s)

ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s~n", [filename:join([code:root_dir(), "erts-" ++ erlang:system_info(version), "include"])])' -s init stop)

CFLAGS  ?= -O2 -g -Wall -Wextra -Wpedantic -fPIC -std=c11
CFLAGS  += -I$(ERTS_INCLUDE_DIR)

LDFLAGS ?= -shared
ifeq ($(UNAME), Darwin)
LDFLAGS += -undefined dynamic_lookup -flat_namespace
endif

PRIV_DIR := ../priv
TARGET   := $(PRIV_DIR)/erllama_kv_nif.so

SOURCES := erllama_kv_nif.c crc32c.c
OBJECTS := $(SOURCES:.c=.o)
DEPS    := $(SOURCES:.c=.d)

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(PRIV_DIR)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

$(PRIV_DIR):
	mkdir -p $@

%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET)

-include $(DEPS)

.PHONY: all clean
