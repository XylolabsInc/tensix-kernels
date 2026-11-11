// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Compute of head_pick (head_pick.py). The reader's gathered rows (c_1: DT / CW blocks of 32 row-major rows, the TOP
// candidates' bf16 head rows and zeros) are tilized into c_2 (DT tiles: candidate c in row c of each), then x's DT tiles
// (c_0) times them transposed, accumulated over the DT tiles in fp32 dest: out[row][c] = x_row . head_row(c). With
// ROUND the result is rounded to bf16 in dest (tk::round_full, to nearest: the same bits whatever the packer's own fp32
// rounding; a host head's z.to(bfloat16)) and packed to c_16 in bf16, else packed in fp32. Tilize and the
// matmul do not overlap: tilize_block takes dest a tile at a time, so all of c_2 is made first. Every input CB is bf16;
// the pack is reconfigured for c_2 (bf16) before the tilize and for c_16 before the result.
#include "dense_ops.h"
#include "api/compute/tilize.h"

constexpr uint32_t cb_x = 0, cb_rm = 1, cb_cand = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t CW = get_compile_time_arg_val(1);
    constexpr bool ROUND = get_compile_time_arg_val(2) != 0;
    static_assert(DT % CW == 0, "head_pick: whole blocks");

    // the hardware for the matmul (in0 x, in1 the candidates' tiles; the pack for c_2 first)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_cand, cb_cand);

    // the candidates' rows into tiles (each block into its place of one reservation)
    tilize_init(cb_rm, CW, cb_cand);
    cb_reserve_back(cb_cand, DT);
    for (uint32_t b = 0; b < DT / CW; ++b) {
        cb_wait_front(cb_rm, CW);
        tilize_block(cb_rm, CW, cb_cand, 0, b * CW);
        tk::pop(cb_rm, CW);
    }
    cb_push_back(cb_cand, DT);
    tilize_uninit(cb_rm, cb_cand);

    // x times the candidates, the in1 tiles transposed (their rows are the candidates)
    cb_wait_front(cb_cand, DT);
    matmul_block_init(cb_x, cb_cand, 1, 1, 1, 1);
    pack_reconfig_data_format(cb_out);
    tile_regs_acquire();
    for (uint32_t t = 0; t < DT; ++t) {
        cb_wait_front(cb_x, 1);
        matmul_block(cb_x, cb_cand, 0, t, 0, 1, 1, 1, 1);
        tk::pop(cb_x, 1);
    }
    if constexpr (ROUND) {
        silu_tile_init();  // (an SFPU init; the round is a plain unary SFPU op)
        tk::round_full(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    tk::pop(cb_cand, DT);
}
