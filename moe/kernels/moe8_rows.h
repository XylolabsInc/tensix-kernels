// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// moe8_rows.h: what the readers and writers of moe8's R-row ops (moe8.py gate_up_rows,
// down_rows) share. The ids and the routing weights are TILE tensors [1, 1, R, k] (R <= 32,
// k <= 16): one tile, element (r, s) at face (r / 16) * 2 + s / 16. Every kernel builds the same
// union of the R x k ids, the distinct experts in increasing order, from the same tile, so the
// cores agree on it without talking: union index i is expert list[i]. An id at or above E (a
// row that is not one) is left out.
// The Python side inlines this file into the dataflow kernels' source.

#pragma once

#include <cstdint>

namespace moe8r {

// element (r, c) of a 32 x 32 tile in its face order
inline uint32_t tile_at(uint32_t r, uint32_t c) {
    return ((r >> 4) << 9) + ((c >> 4) << 8) + ((r & 15) << 4) + (c & 15);
}

template <bool IDS32>
inline uint32_t id_at(uint32_t ids, uint32_t r, uint32_t s) {
    const uint32_t t = tile_at(r, s);
    return IDS32 ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids)[t]
                 : reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids)[t];
}

// The union of the ids of rows 0 .. rows - 1, slots 0 .. K - 1: list[0 .. u - 1] in increasing
// order (bits: (E + 31) / 32 words of scratch). Returns u.
template <bool IDS32, uint32_t K, uint32_t E>
inline uint32_t build_union(
    uint32_t ids, uint32_t rows, volatile tt_l1_ptr uint32_t* bits, volatile tt_l1_ptr uint16_t* list) {
    constexpr uint32_t WORDS = (E + 31) / 32;
    for (uint32_t w = 0; w < WORDS; ++w) {
        bits[w] = 0;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t e = id_at<IDS32>(ids, r, s);
            if (e < E) {
                bits[e >> 5] = bits[e >> 5] | (1u << (e & 31));
            }
        }
    }
    uint32_t u = 0;
    for (uint32_t w = 0; w < WORDS; ++w) {
        uint32_t v = bits[w];
        while (v) {
            const uint32_t b = __builtin_ctz(v);
            list[u++] = static_cast<uint16_t>(w * 32 + b);
            v &= v - 1;
        }
    }
    return u;
}

// The units of core j of a bank with CPB cores: union indices j, j + CPB, ... below u
inline uint32_t units_of(uint32_t u, uint32_t j, uint32_t cpb) { return u > j ? (u - j + cpb - 1) / cpb : 0; }

}  // namespace moe8r
