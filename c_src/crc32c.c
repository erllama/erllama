#include "crc32c.h"

#include <pthread.h>

#define POLY 0x82F63B78u  /* reflected form of 0x1EDC6F41 */

static uint32_t crc32c_table[256];
static pthread_once_t crc32c_table_once = PTHREAD_ONCE_INIT;

static void crc32c_table_build(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1u) ? (POLY ^ (c >> 1)) : (c >> 1);
        }
        crc32c_table[i] = c;
    }
}

void erllama_crc32c_init(void) {
    /* pthread_once guarantees crc32c_table_build runs exactly once
     * across concurrent callers; subsequent calls are a single
     * relaxed atomic load on the once-control. */
    pthread_once(&crc32c_table_once, crc32c_table_build);
}

uint32_t erllama_crc32c_update(uint32_t crc, const uint8_t *data, size_t len) {
    uint32_t c = ~crc;
    for (size_t i = 0; i < len; i++) {
        c = crc32c_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return ~c;
}
