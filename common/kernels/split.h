// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// split.h: which RISC loads a weight chunk when two loaders share a core's chunks (SPLIT): the reader RISC or the
// writer RISC; the reader may take more than half of the chunks. The compile argument ARG is 0 (no split), 1 (the
// alternation: the odd chunks are the writer's), or (P << 24) | MASK: a period of P chunks (3..24) and the positions in it the writer
// loads (bit i: position i of every period, chunk g at position g % P). The host (split_arg in dense_reader/linear.py)
// never gives the writer position 0 or two positions in a row, so chunk 0 is the reader's and a reader's chunk is
// followed by at most one of the writer's: the reader's x groups of the streamed-x path hold at most two chunks, as with
// the alternation.
#ifndef TK_SPLIT_H
#define TK_SPLIT_H
#include <cstdint>

namespace tk {

template <uint32_t ARG>
constexpr uint32_t split_period() {
    return ARG == 1 ? 2 : (ARG >> 24) & 0x3F;
}

template <uint32_t ARG>
constexpr uint32_t split_mask() {
    return ARG == 1 ? 2 : ARG & 0xFFFFFF;
}

template <uint32_t ARG>
constexpr bool split_valid() {
    if constexpr (ARG <= 1) {
        return true;
    } else {
        constexpr uint32_t P = split_period<ARG>(), M = split_mask<ARG>();
        return P >= 3 && P <= 24 && (M >> P) == 0 && (M & 1) == 0 && (M & (M >> 1)) == 0 && M != 0;
    }
}

// 1 when chunk g (the running chunk index) is the writer's
template <uint32_t ARG>
inline uint32_t split_writer(uint32_t g) {
    static_assert(split_valid<ARG>(), "split: a period of 3..24, the writer never at position 0 or twice in a row");
    if constexpr (ARG == 0) {
        return 0;
    } else if constexpr (ARG == 1) {
        return g & 1;
    } else {
        return (split_mask<ARG>() >> (g % split_period<ARG>())) & 1;
    }
}

}  // namespace tk
#endif
