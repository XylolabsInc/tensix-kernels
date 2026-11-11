// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// w4a16_wload.h: the int4 weight loads of the W4A16 matmul, shared by its reader and (with
// SPLIT) its writer, which load alternate chunks into two CBs (one request stream per RISC and NoC). A chunk is KW K rows of SBW tiles:
// each K row is one DRAM read of jn * 512 bytes into a staging buffer, then each 512-byte tile is
// copied over the core's own NoC behind the constant exponent bytes of its BFP4_b page (576 B).
// w4a16.py inlines this file.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace w4 {

constexpr uint32_t Q4_TILE = 512, BFP4_TILE = 576, EXP_BYTES = 64;

inline void fill_words(uint32_t dst, uint32_t word, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = word;
    }
}

inline void init_exponents(uint32_t cb, uint32_t pages, uint32_t exp) {
    const uint32_t base = get_write_ptr(cb);
    for (uint32_t p = 0; p < pages; ++p) {
        fill_words(base + p * BFP4_TILE, exp * 0x01010101u, EXP_BYTES);
    }
}

template <uint32_t KW, uint32_t SBW>
inline void chunk(
    uint32_t cb, uint32_t stage, uint32_t bank, uint32_t w_addr, uint32_t pitch, uint32_t c0, uint32_t jn,
    uint32_t kt0, uint32_t vc) {
    cb_reserve_back(cb, KW * SBW);
    const uint32_t lw = get_write_ptr(cb) + EXP_BYTES;
    if (jn) {
        for (uint32_t k = 0; k < KW; ++k) {
            noc_async_read(
                get_noc_addr_from_bank_id<true>(bank, w_addr + (kt0 + k) * pitch + c0 * Q4_TILE),
                stage + k * SBW * Q4_TILE,
                jn * Q4_TILE,
                noc_index,
                vc);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < KW; ++k) {
            for (uint32_t j = 0; j < jn; ++j) {
                noc_async_read(
                    get_noc_addr(stage + (k * SBW + j) * Q4_TILE),
                    lw + (k * SBW + j) * BFP4_TILE,
                    Q4_TILE,
                    noc_index,
                    vc);
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb, KW * SBW);
}

}  // namespace w4
