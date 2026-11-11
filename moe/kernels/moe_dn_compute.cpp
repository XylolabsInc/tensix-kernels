// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: the matmul_block loop rewritten for the routed experts' down projection, summed
// over the slots. Adapted from tenstorrent/tt-metal v0.81.1,
// ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// Compute of moe8's down (moe8.py): y[NC tiles] = sum over the k slots' K rows of h x the chosen
// experts' down rows, all of it accumulated in dest (NC <= 4 with half sync, up to 8 with full sync), then bf16 out.
//
// SPLIT (two loaders: moe_dn2_*.cpp, moe_dnr2_writer.cpp): chunk c comes from c_4 when it is the writer's
// (split.h), else from c_1.
//
// CBs: c_0 h chunks (KW, bf16), c_1 weights (BFP4_b), c_3 the chunk count (DYN), c_4 the writer's weight chunks
//      (SPLIT), c_16 out (bf16).

#include "dense_ops.h"
#include "split.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 3, cb_w2 = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KTOT = get_compile_time_arg_val(0);  // k x I / 32
    constexpr uint32_t KW = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    // DYN: the chunk count from the reader (cb_meta, word 0: down_rows' union x I / 32 / KW)
    constexpr bool DYN = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(4);  // split.h: the writer's chunks, in c_4
    constexpr bool SPLIT = SPLIT_ARG != 0;
    // FULL: the program runs with full dest sync (moe8.py: only for NC > 4), 8 fp32 tiles of dest instead of half
    // sync's 4
    constexpr bool FULL = get_compile_time_arg_val(5) != 0;
    // KBLOCK: the pack holds each 32 x 32 block of D transposed (moe8.pack(kblock=True)); the matmul takes in1 transposed
    // within the tile.
    constexpr uint32_t KBLOCK = get_compile_time_arg_val(6);
    static_assert(KBLOCK <= 1, "moe8: KBLOCK is 0 or 1");
    static_assert(KTOT % KW == 0 && NC <= (FULL ? 8 : 4), "moe8: down's blocks");

    uint32_t chunks = KTOT / KW;
    if constexpr (DYN) {
        cb_wait_front(cb_meta, 1);
        chunks = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
    }
    tk::mm_init(cb_x, cb_w, cb_out, NC, 1, KW, KBLOCK);
    tile_regs_acquire();
    for (uint32_t c = 0; c < chunks; ++c) {
        const uint32_t cw = (SPLIT && tk::split_writer<SPLIT_ARG>(c)) ? cb_w2 : cb_w;
        cb_wait_front(cb_x, KW);
        cb_wait_front(cw, KW * NC);
        for (uint32_t k = 0; k < KW; ++k) {
            matmul_block(cb_x, cw, k, k * NC, 0, KBLOCK, NC, 1, KW);
        }
        tk::pop(cw, KW * NC);
        tk::pop(cb_x, KW);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, NC);
    for (uint32_t j = 0; j < NC; ++j) {
        pack_tile(j, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, NC);
}
