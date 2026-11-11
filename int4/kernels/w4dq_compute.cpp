// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Compute of the W4 dequantize (w4a16.py dequantize): every int4 weight tile (BFP4_b under the
// constant exponent, so each value is its code q) times its group scales, s[kt, n] for K tile kt:
// row kt % 32 of the pass's scale tile broadcast over the tile's rows (mul_tiles_bcast_rows with
// a row index), packed as BFP8_b. Many rows (prefill) can then run an ordinary BFP8_b matmul on it.
//
// CBs: c_1 weights (BFP4_b pages), c_2 scales (NC tiles of a pass, bf16), c_16 out (BFP8_b).

#include "dense_ops.h"

constexpr uint32_t cb_w = 1, cb_s = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t SB = get_compile_time_arg_val(2);  // tiles per dest acquire
    static_assert(NC % SB == 0, "w4dq: NC % SB");

    tk::unary_like_init(cb_w, cb_s, cb_out);
    tk::mul_bcast_rows_init(cb_w, cb_s);
    for (uint32_t kt = 0; kt < KT; ++kt) {
        if (kt % 32 == 0) {
            cb_wait_front(cb_s, NC);
        }
        cb_wait_front(cb_w, NC);
        for (uint32_t j0 = 0; j0 < NC; j0 += SB) {
            cb_reserve_back(cb_out, SB);
            tile_regs_acquire();
            for (uint32_t j = 0; j < SB; ++j) {
                mul_tiles_bcast_rows(cb_w, cb_s, j0 + j, j0 + j, j, kt % 32);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < SB; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, SB);
        }
        tk::pop(cb_w, NC);
        if (kt % 32 == 31 || kt == KT - 1) {
            tk::pop(cb_s, NC);
        }
    }
}
