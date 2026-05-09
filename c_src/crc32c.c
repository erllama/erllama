#include "crc32c.h"

#include <stdatomic.h>

#define POLY 0x82F63B78u  /* reflected form of 0x1EDC6F41 */

static uint32_t crc32c_table[256];
static atomic_int crc32c_table_ready = 0;

void erllama_crc32c_init(void) {
    if (atomic_load_explicit(&crc32c_table_ready, memory_order_acquire)) {
        return;
    }
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1u) ? (POLY ^ (c >> 1)) : (c >> 1);
        }
        crc32c_table[i] = c;
    }
    atomic_store_explicit(&crc32c_table_ready, 1, memory_order_release);
}

uint32_t erllama_crc32c_update(uint32_t crc, const uint8_t *data, size_t len) {
    uint32_t c = ~crc;
    for (size_t i = 0; i < len; i++) {
        c = crc32c_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return ~c;
}
