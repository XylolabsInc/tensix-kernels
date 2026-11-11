// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// dense_wload.h: the weight loads of the dense reader, shared by its reader and (with SPLIT) its writer, which then
// load their chunks into two CBs (two request streams per core, one per data-movement RISC and NoC). The Python side
// inlines this file into both data-movement kernels.
//
// The weight is a tensor of block-float or bf16 tiles (bfloat8_b: 1088 B, bfloat4_b: 576 B, bfloat16: 2048 B a tile;
// TB below) whose tiles carry their own exponents: with a DRAM width-sharded weight each K row of the core's columns
// is one contiguous read straight into the CB (a CB page is a tile); with an interleaved weight a read a tile.
//
// HALF (> 0, sharded only): two weights of one sharding (the GLU's gate and up apart): each K row of a chunk is the
// gate's jn / 2 tiles at page 0 of the row and the up's (w2_addr, the same offsets) at page HALF, two contiguous reads;
// the compute pairs tile j with tile HALF + j.
// REQ (> 0, sharded only): a K row's read cut into requests of at most REQ tiles; 0: one request a row (a half row with
// HALF).

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "split.h"

namespace tk {

template <bool SHARDED, typename Acc, uint32_t TB, uint32_t HALF = 0, uint32_t REQ = 0>
struct Loader {
    uint32_t cb;
    const Acc& w_acc;
    uint32_t w_addr, bank, pitch;
    uint32_t vc;           // the read requests' virtual channel
    uint32_t w2_addr = 0;  // HALF: the up weight's address (bank-local, as w_addr)

    // One chunk of KW x W weight tiles: tile (k, j) of the chunk is weight row kt0 + k, column c0 + j of the core, at
    // page k * W + j; only the first jn columns are read.
    //   sharded: DRAM bank `bank`, tile (row, col) at w_addr + (row * pitch + col) * TB
    //   interleaved: page row * pitch + col (w_acc)
    template <uint32_t KW, uint32_t W>
    void chunk(uint32_t kt0, uint32_t c0, uint32_t jn) {
        constexpr uint32_t CH = KW * W;
        static_assert(HALF == 0 || (SHARDED && W == 2 * HALF), "dense: two weights");
        cb_reserve_back(cb, CH);
        const uint32_t start = get_write_ptr(cb);
        for (uint32_t k = 0; k < KW && jn; ++k) {
            const uint32_t row = (kt0 + k) * pitch + c0;
            if constexpr (HALF > 0) {
                for (uint32_t j0 = 0; j0 < jn / 2; j0 += (REQ ? REQ : jn / 2)) {
                    const uint32_t nj = REQ && jn / 2 - j0 > REQ ? REQ : jn / 2 - j0;
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(bank, w_addr + (row + j0) * TB),
                        start + (k * W + j0) * TB,
                        nj * TB,
                        noc_index,
                        vc);
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(bank, w2_addr + (row + j0) * TB),
                        start + (k * W + HALF + j0) * TB,
                        nj * TB,
                        noc_index,
                        vc);
                }
            } else if constexpr (SHARDED) {
                for (uint32_t j0 = 0; j0 < jn; j0 += (REQ ? REQ : jn)) {
                    const uint32_t nj = REQ && jn - j0 > REQ ? REQ : jn - j0;
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(bank, w_addr + (row + j0) * TB),
                        start + (k * W + j0) * TB,
                        nj * TB,
                        noc_index,
                        vc);
                }
            } else {
                for (uint32_t j = 0; j < jn; ++j) {
                    noc_async_read_page(row + j, w_acc, start + (k * W + j) * TB);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb, CH);
    }
};

// The chunks of x block b in the compute's order (column subblock s, then K chunk c); with split, only those whose
// running index g belongs to `part` (split.h: 0 the reader, 1 the writer, by the pattern SA). g counts from g0 and
// is returned advanced.
template <uint32_t KB, uint32_t KW, uint32_t SBW, uint32_t NS, uint32_t SA = 1, typename L>
inline uint32_t load_block(L& ld, uint32_t col0, uint32_t nvalid, uint32_t b, uint32_t g, bool split, uint32_t part) {
    for (uint32_t s = 0; s < NS; ++s) {
        const uint32_t j0 = s * SBW;
        const uint32_t jn = nvalid > j0 ? (nvalid - j0 < SBW ? nvalid - j0 : SBW) : 0;
        for (uint32_t c = 0; c < KB / KW; ++c, ++g) {
            if (!split || split_writer<SA>(g) == part) {
                ld.template chunk<KW, SBW>(b * KB + c * KW, col0 + j0, jn);
            }
        }
    }
    return g;
}

// ROWS order: the chunks of x block b are KW full rows of the core's columns (nvalid of NC read), K chunk after K
// chunk; with split only those whose running index g belongs to `part` (the pattern SA).
template <uint32_t KB, uint32_t KW, uint32_t NC, uint32_t SA = 1, typename L>
inline uint32_t load_block_rows(
    L& ld, uint32_t col0, uint32_t nvalid, uint32_t b, uint32_t g, bool split, uint32_t part) {
    for (uint32_t c = 0; c < KB / KW; ++c, ++g) {
        if (!split || split_writer<SA>(g) == part) {
            ld.template chunk<KW, NC>(b * KB + c * KW, col0, nvalid);
        }
    }
    return g;
}

}  // namespace tk
