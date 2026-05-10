// Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
// See the LICENSE file at the project root.

/*
 * RFC 3720 / iSCSI / SCTP CRC32C (Castagnoli polynomial 0x1EDC6F41).
 *
 * Reflected form. Matches Erlang's expectation when used as the
 * payload checksum for KVC v2 cache files.
 *
 * The implementation is a straightforward 256-entry table lookup
 * (slicing-by-1). It runs at roughly 400 MB/s on a single core on
 * commodity x86-64 hardware which is fast enough for occasional
 * multi-GB checks at save/load time. Hardware acceleration
 * (PCLMULQDQ on x86-64, CRC extension on ARM64) is a future
 * optimisation if profiling shows this becomes a bottleneck.
 */
#ifndef ERLLAMA_CRC32C_H
#define ERLLAMA_CRC32C_H

#include <stddef.h>
#include <stdint.h>

/* Initialise the lookup table. Idempotent and thread-safe (uses
 * pthread_once internally). Returns the pthread_once result: 0 on
 * success, or the platform error code if the once-control could not
 * be initialised (extremely rare; only happens for an invalid
 * once-control or out-of-resources). */
int erllama_crc32c_init(void);

/* Update an existing CRC32C state with `len` bytes of `data`.
 *
 * Convention matches RFC 3720: callers pass 0 for `crc` to start a
 * new computation. Internally the function inverts on entry and
 * exit so that a single call computes the full digest. To run an
 * incremental computation, pass the previous return value as `crc`.
 */
uint32_t erllama_crc32c_update(uint32_t crc, const uint8_t *data, size_t len);

#endif
