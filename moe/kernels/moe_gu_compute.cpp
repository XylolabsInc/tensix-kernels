// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: the matmul_block loop rewritten for the routed experts' gate/up pairs with the GLU
// and routing weight in dest. Adapted from tenstorrent/tt-metal v0.81.1,
// ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// Compute of moe8's gate_up (moe8.py): per slot (unit) the expert's NC columns of gate and up
// tiles in turn (tile 2p gate, 2p + 1 up), accumulated in dest over all of K (full dest sync,
// NC <= 8), then in dest (SFPU): silu(gate) * up (gelu_tanh with ACT 1; ACT 2 gpt-oss's clamped SwiGLU,
// min(g, 7) * sigmoid(1.702 min(g, 7)) * (clamp(up, -7, 7) + 1)), times the slot's routing weight (its tile holds
// w in all of row 0, copied into the free dest NC - 1), packed in bf16: NC / 2 tiles per unit (the
// writer keeps the valid ones).
//
// gate_up_rows (DYN): the weight's tile holds each row's coefficient in its row; ITER is 0.
//
// SPLIT (two loaders: moe_gu2_*.cpp, moe_gur2_writer.cpp): chunk g of the running chunk index (over the units) comes
// from c_4 when it is the writer's (split.h), else from c_1.
//
// BIAS (a gate/up bias, as gpt-oss's experts have): each unit's NC bias tiles (c_5, bf16: row 0, or every row for gate_up_rows) are
// copied into dest before its K loop, so the matmul accumulates x G + b and x U + b there.
//
// CBs: c_0 x (KT, bf16), c_1 weights (BFP4_b), c_2 the weight's tile (bf16), c_3 the unit count
//      (DYN), c_4 the writer's weight chunks (SPLIT), c_5 the bias tiles (BIAS), c_16 out (bf16).

#include "dense_ops.h"
#include "split.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ws = 2, cb_meta = 3, cb_w2 = 4, cb_b = 5, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KW = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    // ITER > 0 (at most 16 rows of x): the SFPU stages on the tiles' top half only, silu over ITER
    // iterations (dense_ops.h silu_rows; two cover four rows)
    constexpr int ITER = get_compile_time_arg_val(3);
    // DYN: the unit count from the reader (cb_meta, word 0: gate_up_rows' union), else the
    // runtime argument
    constexpr bool DYN = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(5);  // split.h: the writer's chunks, in c_4
    constexpr bool SPLIT = SPLIT_ARG != 0;
    // ACT: the gate's activation, 0 silu (SwiGLU), 1 gelu_tanh (GeGLU, as Gemma's experts; HF's gelu_pytorch_tanh,
    // dense_ops.h tk_gelu_tanh, as the dense reader's GeGLU stage), 2 swiglu_oai (gpt-oss's clamped SwiGLU, tk_oai_gate on
    // the gate and tk_oai_up on the up tile).
    constexpr uint32_t ACT = get_compile_time_arg_val(6);
    static_assert(ACT <= 2, "moe8: the activation is silu (0), gelu_tanh (1) or swiglu_oai (2)");
    constexpr bool BIAS = get_compile_time_arg_val(7) != 0;
    // KBLOCK: the pack holds each 32 x 32 block of G and U transposed (moe8.pack(kblock=True): BFP4 exponent blocks along
    // K, as GPTQ along K exports them); the matmul takes in1 transposed within the tile.
    constexpr uint32_t KBLOCK = get_compile_time_arg_val(8);
    static_assert(KBLOCK <= 1, "moe8: KBLOCK is 0 or 1");
    constexpr uint32_t H = NC / 2;
    static_assert(NC % 2 == 0 && NC <= 8 && KT % KW == 0, "moe8: gate/up pairs in dest");
    uint32_t units;
    if constexpr (DYN) {
        cb_wait_front(cb_meta, 1);
        units = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
    } else {
        units = get_arg_val<uint32_t>(0);
    }

    tk::mm_init(cb_x, cb_w, cb_out, NC, 1, KT, KBLOCK);
    cb_wait_front(cb_x, KT);
    uint32_t g = 0;  // the running chunk index (SPLIT: the writer's in c_4)
    for (uint32_t u = 0; u < units; ++u) {
        tile_regs_acquire();
        if constexpr (BIAS) {
            // the unit's bias into dest; the K loop's matmul accumulates on it
            reconfig_data_format_srca(cb_w, cb_b);
            tk::copy_init(cb_b);
            cb_wait_front(cb_b, NC);
            for (uint32_t t = 0; t < NC; ++t) {
                copy_tile(cb_b, t, t);
            }
            tk::pop(cb_b, NC);
            reconfig_data_format_srca(cb_b, cb_w);
            tk::mm_init_short(cb_x, cb_w, NC, 1, KT, KBLOCK);
        }
        for (uint32_t c = 0; c < KT / KW; ++c, ++g) {
            const uint32_t cw = (SPLIT && tk::split_writer<SPLIT_ARG>(g)) ? cb_w2 : cb_w;
            cb_wait_front(cw, KW * NC);
            for (uint32_t k = 0; k < KW; ++k) {
                matmul_block(cb_x, cw, c * KW + k, k * NC, 0, KBLOCK, NC, 1, KT);
            }
            tk::pop(cw, KW * NC);
        }
        // silu(gate) * up (gelu_tanh with ACT 1; swiglu_oai's halves with ACT 2) into dest 0 .. H - 1
        silu_tile_init();  // (gelu_tanh's and swiglu_oai's sigmoid is silu's)
        for (uint32_t p = 0; p < H; ++p) {
            if constexpr (ACT == 2) {
                if constexpr (ITER > 0) {
                    tk::oai_gate_rows<ITER>(2 * p);
                    tk::oai_up_rows<ITER>(2 * p + 1);
                } else {
                    tk::oai_gate_full(2 * p);
                    tk::oai_up_full(2 * p + 1);
                }
            } else if constexpr (ACT == 1) {
                if constexpr (ITER > 0) {
                    tk::gelu_rows<ITER>(2 * p);
                } else {
                    tk::gelu_full(2 * p);
                }
            } else if constexpr (ITER > 0) {
                tk::silu_rows<ITER>(2 * p);
            } else {
                silu_tile(2 * p);
            }
        }
        mul_binary_tile_init();
        for (uint32_t p = 0; p < H; ++p) {
            if constexpr (ITER > 0) {
                tk::mul_rows(2 * p, 2 * p + 1, p);
            } else {
                mul_binary_tile(2 * p, 2 * p + 1, p);
            }
        }
        // times the routing weight: its tile (w in all of row 0) into the free dest NC - 1
        reconfig_data_format_srca(cb_w, cb_ws);
        tk::copy_init(cb_ws);
        cb_wait_front(cb_ws, 1);
        copy_tile(cb_ws, 0, NC - 1);
        mul_binary_tile_init();
        for (uint32_t p = 0; p < H; ++p) {
            if constexpr (ITER > 0) {
                tk::mul_rows(p, NC - 1, p);
            } else {
                mul_binary_tile(p, NC - 1, p);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, H);
        for (uint32_t p = 0; p < H; ++p) {
            pack_tile(p, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, H);
        tk::pop(cb_ws, 1);

        reconfig_data_format_srca(cb_ws, cb_w);
        tk::mm_init_short(cb_x, cb_w, NC, 1, KT, KBLOCK);
    }
    tk::pop(cb_x, KT);
}
