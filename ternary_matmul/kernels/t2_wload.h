// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// t2_wload.h: the weight loads of the ternary matmul (t2.py), shared by its reader and (with SPLIT)
// its writer, which then load alternate chunks into two CBs (two request streams per core). The Python
// side inlines this file into both data-movement kernels.
//
// t2.py lays every core's weights out contiguously in its DRAM bank: KT K rows of the core's nv
// tiles, row after row, TB bytes per tile (a BFP2_b tile of 320 bytes, or a BFP4_b one of 576,
// each with its 64 exponent bytes); no padding is stored. A chunk of KW K rows is KW reads of nv
// tiles, each into its row of KW x NC pages (the pages past nv are padding columns: stale data
// whose outputs are never written).
// DB < TB (codes only): DRAM holds DB = TB - 64 bytes per tile, the codes without their exponent
// bytes (all 127 for BFP2_b: a fifth fewer bytes). init_exponents sets every page's exponent bytes
// once; a chunk's rows land in a staging buffer in large reads (issue), then each tile's codes are
// copied behind its page's exponents by the core's own NoC (place).

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace t2 {

// The exponent bytes (64) of every page of cb, once.
inline void init_exponents(uint32_t cb, uint32_t pages, uint32_t tb, uint32_t exp) {
    const uint32_t base = get_write_ptr(cb);
    const uint32_t word = exp * 0x01010101u;
    for (uint32_t p = 0; p < pages; ++p) {
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + p * tb);
        for (uint32_t q = 0; q < 16; ++q) {
            e[q] = word;
        }
    }
}

template <uint32_t KW, uint32_t NC, uint32_t TB, uint32_t DB>
struct Chunks {
    static constexpr uint32_t PAGES = KW * NC;
    static constexpr bool CODES = DB < TB;
    uint32_t cb, bank, addr, nv, vc;  // addr: the core's first byte in the bank; nv: its tiles
    uint32_t stage = 0;               // CODES: the staging buffer (KW x nv tiles of DB bytes)
    uint32_t dst = 0;                 // the chunk's first page

    // chunk c: reserve its pages and issue its DRAM reads (the caller waits with a read barrier)
    void issue(uint32_t c) {
        cb_reserve_back(cb, PAGES);
        dst = get_write_ptr(cb);
        for (uint32_t k = 0; k < KW; ++k) {
            noc_async_read(
                get_noc_addr_from_bank_id<true>(bank, addr + (c * KW + k) * nv * DB),
                CODES ? stage + k * nv * DB : dst + k * NC * TB,
                nv * DB,
                noc_index,
                vc);
        }
    }
    // CODES, after the barrier: each tile's codes behind its page's exponents (another barrier)
    void place() const {
        if constexpr (CODES) {
            for (uint32_t k = 0; k < KW; ++k) {
                for (uint32_t j = 0; j < nv; ++j) {
                    noc_async_read<DB>(
                        get_noc_addr(stage + (k * nv + j) * DB), dst + (k * NC + j) * TB + (TB - DB), DB, noc_index, vc);
                }
            }
        }
    }
    void push() const { cb_push_back(cb, PAGES); }
};

}  // namespace t2
