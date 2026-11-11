// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: the matmul_block loop rewritten for streamed block-float weight chunks, split
// loaders and fused output stages. Adapted from tenstorrent/tt-metal v0.81.1,
// ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// Compute of the dense reader's matmul (linear.py): y[MT x NC tiles] = x[MT x KT] @ w[KT x NC] (* s[NC]), x bf16, w
// block-float or bf16 tiles, fp32 accumulation in dest, then an output stage: the per-output-channel scale s (a bf16
// tile per 32 columns, row 0 used, broadcast over the rows), or none (unit scales), the GLU act(gate) * up, or a
// residual added in fp32; bf16 out (or the fp32 accumulator as it is).
//
// The loop follows tt-metal's matmul_block pattern (ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/
// bmm_large_block_zm_fused_bias_activation.cpp, Apache-2.0; see NOTICE): matmul_block over output subblocks, partial
// sums packed with the packer's L1 accumulation, a row-broadcast stage after the matmul.
//
// Order (the reader produces in the same order):
//   for each x block b of KB tiles (NKB = KT / KB blocks):
//     for each column subblock s of SBW tiles (NS = NC / SBW):
//       dest[MT x SBW] = sum over the block's KB/KW weight chunks of KW x SBW tiles
//       packed into cb_acc at (m, s*SBW + j); blocks after the first add to it (packer L1 acc)
//   then the output stage over the acc tiles -> cb_out, MT x NS groups of SBW.
// (ROWS: a chunk is KW full rows of the core's columns, K outermost, every chunk's partial sums added in cb_acc.)
//
// CBs: c_0 x (bf16), c_1 weights (the weight's own tiles), c_2 scales (NC tiles, bf16), c_3 acc (MT*NC tiles, fp32),
//      c_4 the writer's weight chunks with SPLIT, c_8 the residual, c_16 out.

#include "dense_ops.h"
#include "split.h"
#include "api/compute/eltwise_binary_sfpu.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_acc = 3, cb_w2 = 4, cb_res = 8, cb_out = 16;

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t KB = get_compile_time_arg_val(2);
    constexpr uint32_t KW = get_compile_time_arg_val(3);
    constexpr uint32_t SBW = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    // SPLIT: the writer RISC loads some of the chunks into cb_w2 (split.h's pattern: 1 the odd ones, else a period)
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(6);
    constexpr bool SPLIT = SPLIT_ARG != 0;
    // ROWS: the weight chunks are KW full rows of the core's NC columns (contiguous in DRAM), K outermost, every
    // chunk's partial sums added up in cb_acc by the packer (L1 accumulation); else a chunk is KW x SBW tiles of one
    // column subblock, K inner, dest accumulating the whole block
    constexpr bool ROWS = get_compile_time_arg_val(7) != 0;
    // XS (ROWS only, one x block): x streams with the weights, MT x KW tiles per chunk, so no x block waits in front of
    // the first chunk and L1 holds a few chunks of x instead of all of it
    constexpr bool XS = get_compile_time_arg_val(8) != 0;
    // NOSCALE: unit scales: the accumulator is copied out, not multiplied
    constexpr bool NOSCALE = get_compile_time_arg_val(9) != 0;
    // GLU (with NOSCALE): the weight's columns are gate and up tiles in turn (tile 2j gate j, tile 2j + 1 up j), and
    // the output is act(gate) * up: NC / 2 tiles per row (2: gate and up as two weights: the gate's columns first in
    // each row, the up's at NC / 2, so the pairs are tiles j and NC / 2 + j of the row; dense_wload.h HALF)
    constexpr uint32_t GLU_MODE = get_compile_time_arg_val(10);
    constexpr bool GLU = GLU_MODE != 0;
    constexpr bool GLU2 = GLU_MODE == 2;
    // GLU_ITER > 0 (one tile row of x with at most 16 valid rows): the activation and the multiply on the tiles' top
    // half only, the activation over GLU_ITER SFPU iterations (dense_ops.h silu_rows; an even count: two iterations cover
    // four rows)
    constexpr int GLU_ITER = get_compile_time_arg_val(11);
    // GLU_ACT: the activation of the gate: 0 silu, 1 gelu_tanh (dense_ops.h gelu_rows)
    constexpr uint32_t GLU_ACT = get_compile_time_arg_val(12);
    // RES_MODE (with NOSCALE, no GLU): the residual stream's tiles of the core's columns (c_8, loaded by the writer).
    // 1: added to the accumulator in the output stage (fp32, then one bf16 pack); 2: preloaded into dest before the
    // first K chunk's matmul (bf16 to fp32 is exact), so the packer's fp32 L1 accumulation carries it (XS only); the
    // output stage is then the plain one with a round to bf16
    constexpr uint32_t RES_MODE = get_compile_time_arg_val(13);
    constexpr bool HAS_RES = RES_MODE != 0;
    // SEGS (> 1, a block list; XS, unit scales, no residual): the core's columns of SEGS blocks as segments in
    // sequence, every K chunk of segment 0, then of segment 1, ..., each accumulated over K as without segments; the
    // accumulator holds them side by side (segment s at column s * NC of each tile row), the output stage runs over all
    // of them (GLU pairs within a segment) and the writer puts each segment's columns at its block's place
    constexpr uint32_t SEGS = get_compile_time_arg_val(14);
    constexpr uint32_t NCT = NC * SEGS;  // the accumulator's tiles a row
    static_assert(!GLU || (NOSCALE && SBW % 2 == 0), "dense: GLU needs NOSCALE and an even SBW");
    static_assert(!HAS_RES || (NOSCALE && !GLU), "dense: a residual needs unit scales, no GLU stage");
    static_assert(RES_MODE != 2 || XS, "dense: the residual preload needs x streaming");
    constexpr uint32_t NKB = KT / KB;
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NKW = KB / KW;
    constexpr uint32_t CH = KW * (ROWS ? NC : SBW);
    static_assert(KT % KB == 0 && KB % KW == 0 && NC % SBW == 0, "dense: blocks must divide");
    static_assert(!XS || (ROWS && NKB == 1), "dense: XS needs ROWS and one x block");
    static_assert(SEGS >= 1 && SEGS <= 4, "dense: 1 to 4 segments");
    static_assert(SEGS == 1 || (XS && NOSCALE && !HAS_RES), "dense: segments: x streamed, unit scales, no residual");
    constexpr uint32_t NST = NS * SEGS;  // the output stage's subblocks a row

    tk::mm_init(cb_x, cb_w, cb_acc, SBW, MT, XS ? KW : KB);

    cb_reserve_back(cb_acc, MT * NCT);
    uint32_t g = 0;  // running chunk index: with SPLIT the writer's (split.h) come from cb_w2 (same format)
    // XS: each segment's K chunks in turn (SEGS 1: one pass); without XS no pass here
    for (uint32_t sg = 0; sg < (XS ? SEGS : 0); ++sg) {
        for (uint32_t c = 0; c < NKW; ++c, ++g) {
            const uint32_t cw = (SPLIT && tk::split_writer<SPLIT_ARG>(g)) ? cb_w2 : cb_w;
            cb_wait_front(cb_x, MT * KW);
            cb_wait_front(cw, CH);
            if constexpr (RES_MODE == 2) {
                if (c == 0) {
                    cb_wait_front(cb_res, MT * NC);
                }
            }
            for (uint32_t s = 0; s < NS; ++s) {
                tile_regs_acquire();
                if constexpr (RES_MODE == 2) {
                    if (c == 0) {
                        // dst[r * SBW + j] = the residual tile (r, s * SBW + j), then the matmul adds on top
                        reconfig_data_format_srca(cw, cb_res);
                        tk::copy_init(cb_res);
                        for (uint32_t m = 0; m < MT; ++m) {
                            for (uint32_t j = 0; j < SBW; ++j) {
                                copy_tile(cb_res, m * NC + s * SBW + j, m * SBW + j);
                            }
                        }
                        reconfig_data_format_srca(cb_res, cw);
                        tk::mm_init_short(cb_x, cw, SBW, MT, KW);
                    }
                }
                for (uint32_t k = 0; k < KW; ++k) {
                    // dst[r * SBW + j] += x[r * KW + k] * w[k * NC + s*SBW + j]
                    matmul_block(cb_x, cw, k, k * NC + s * SBW, 0, 0, SBW, MT, KW);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(c > 0 ? 1 : 0);  // (each segment's first chunk starts its own sums)
                for (uint32_t m = 0; m < MT; ++m) {
                    for (uint32_t j = 0; j < SBW; ++j) {
                        pack_tile<true>(m * SBW + j, cb_acc, m * NCT + sg * NC + s * SBW + j);
                    }
                }
                tile_regs_release();
            }
            tk::pop(cw, CH);
            tk::pop(cb_x, MT * KW);
            if constexpr (RES_MODE == 2) {
                if (c == 0) {
                    tk::pop(cb_res, MT * NC);
                }
            }
        }
    }
    for (uint32_t b = 0; b < (XS ? 0 : NKB); ++b) {
        cb_wait_front(cb_x, MT * KB);
        if constexpr (ROWS) {
            for (uint32_t c = 0; c < NKW; ++c, ++g) {
                const uint32_t cw = (SPLIT && tk::split_writer<SPLIT_ARG>(g)) ? cb_w2 : cb_w;
                cb_wait_front(cw, CH);
                for (uint32_t s = 0; s < NS; ++s) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < KW; ++k) {
                        // dst[r * SBW + j] += x[r * KB + c*KW + k] * w[k * NC + s*SBW + j]
                        matmul_block(cb_x, cw, c * KW + k, k * NC + s * SBW, 0, 0, SBW, MT, KB);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_l1_acc((b > 0 || c > 0) ? 1 : 0);
                    for (uint32_t m = 0; m < MT; ++m) {
                        for (uint32_t j = 0; j < SBW; ++j) {
                            pack_tile<true>(m * SBW + j, cb_acc, m * NC + s * SBW + j);
                        }
                    }
                    tile_regs_release();
                }
                tk::pop(cw, CH);
            }
        } else {
            for (uint32_t s = 0; s < NS; ++s) {
                tile_regs_acquire();
                for (uint32_t c = 0; c < NKW; ++c, ++g) {
                    const uint32_t cw = (SPLIT && tk::split_writer<SPLIT_ARG>(g)) ? cb_w2 : cb_w;
                    cb_wait_front(cw, CH);
                    for (uint32_t k = 0; k < KW; ++k) {
                        // dst[r * SBW + j] += x[r * KB + c*KW + k] * w[k * SBW + j]
                        matmul_block(cb_x, cw, c * KW + k, k * SBW, 0, 0, SBW, MT, KB);
                    }
                    tk::pop(cw, CH);
                }
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (NKB > 1) {
                    pack_reconfig_l1_acc(b > 0 ? 1 : 0);
                }
                for (uint32_t m = 0; m < MT; ++m) {
                    for (uint32_t j = 0; j < SBW; ++j) {
                        pack_tile<true>(m * SBW + j, cb_acc, m * NC + s * SBW + j);
                    }
                }
                tile_regs_release();
            }
        }
        tk::pop(cb_x, MT * KB);
    }
    if constexpr (ROWS || NKB > 1) {
        pack_reconfig_l1_acc(0);
    }
    cb_push_back(cb_acc, MT * NCT);

    if constexpr (GLU) {
        // per subblock: gate and up tiles to dest, act(gate) (SFPU), times up (SFPU binary), the SBW / 2 products
        // packed in bf16
        reconfig_data_format_srca(cb_w, cb_acc);
        pack_reconfig_data_format(cb_out);
        cb_wait_front(cb_acc, MT * NCT);
        constexpr uint32_t H = SBW / 2;
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t s = 0; s < NST; ++s) {
                cb_reserve_back(cb_out, H);
                tile_regs_acquire();
                tk::copy_init(cb_acc);
                if constexpr (GLU2) {
                    // pair p of this group: gate tile s * H + p, up tile NC / 2 + s * H + p (of its segment)
                    const uint32_t sg = SEGS == 1 ? 0 : s / NS, s1 = s - sg * NS, a0 = m * NCT + sg * NC;
                    for (uint32_t p = 0; p < H; ++p) {
                        copy_tile(cb_acc, a0 + s1 * H + p, 2 * p);
                        copy_tile(cb_acc, a0 + NC / 2 + s1 * H + p, 2 * p + 1);
                    }
                } else {
                    for (uint32_t j = 0; j < SBW; ++j) {
                        copy_tile(cb_acc, m * NCT + s * SBW + j, j);
                    }
                }
                silu_tile_init();  // (gelu_tanh's sigmoid is silu's)
                for (uint32_t p = 0; p < H; ++p) {
                    if constexpr (GLU_ACT == 1) {
                        if constexpr (GLU_ITER > 0) {
                            tk::gelu_rows<GLU_ITER>(2 * p);
                        } else {
                            tk::gelu_full(2 * p);
                        }
                    } else if constexpr (GLU_ITER > 0) {
                        tk::silu_rows<GLU_ITER>(2 * p);
                    } else {
                        silu_tile(2 * p);
                    }
                }
                mul_binary_tile_init();
                for (uint32_t p = 0; p < H; ++p) {
                    if constexpr (GLU_ITER > 0) {
                        tk::mul_rows(2 * p, 2 * p + 1, p);
                    } else {
                        mul_binary_tile(2 * p, 2 * p + 1, p);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t p = 0; p < H; ++p) {
                    pack_tile(p, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, H);
            }
        }
        tk::pop(cb_acc, MT * NCT);
        return;
    }
    if constexpr (NOSCALE && RES_MODE == 1) {
        // the accumulator (fp32) plus the residual tile (bf16), added in fp32 dest, packed to bf16: chunks of CS tiles
        // (CS of the accumulator, CS of the residual in dest; at most 4 fp32 tiles with half-sync dest)
        constexpr uint32_t CS = SBW % 2 == 0 ? 2 : 1;
        pack_reconfig_data_format(cb_out);
        cb_wait_front(cb_acc, MT * NC);
        cb_wait_front(cb_res, MT * NC);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t s = 0; s < NS; ++s) {
                cb_reserve_back(cb_out, SBW);
                for (uint32_t j0 = 0; j0 < SBW; j0 += CS) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_res, cb_acc);
                    tk::copy_init(cb_acc);
                    for (uint32_t j = 0; j < CS; ++j) {
                        copy_tile(cb_acc, m * NC + s * SBW + j0 + j, j);
                    }
                    reconfig_data_format_srca(cb_acc, cb_res);
                    tk::copy_init(cb_res);
                    for (uint32_t j = 0; j < CS; ++j) {
                        copy_tile(cb_res, m * NC + s * SBW + j0 + j, CS + j);
                    }
                    add_binary_tile_init();
                    for (uint32_t j = 0; j < CS; ++j) {
                        add_binary_tile(j, CS + j, j);
                    }
                    silu_tile_init();  // (the unary SFPU set-up; the round below is a plain unary SFPU op)
                    for (uint32_t j = 0; j < CS; ++j) {
                        tk::round_full(j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < CS; ++j) {
                        pack_tile(j, cb_out);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_out, SBW);
            }
        }
        tk::pop(cb_acc, MT * NC);
        tk::pop(cb_res, MT * NC);
        return;
    }
    if constexpr (NOSCALE) {
        // the accumulator (fp32) to the output: unpack to dest, pack
        reconfig_data_format_srca(cb_w, cb_acc);
        pack_reconfig_data_format(cb_out);
        tk::copy_init(cb_acc);
        cb_wait_front(cb_acc, MT * NCT);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t s = 0; s < NST; ++s) {
                cb_reserve_back(cb_out, SBW);
                tile_regs_acquire();
                for (uint32_t j = 0; j < SBW; ++j) {
                    copy_tile(cb_acc, m * NCT + s * SBW + j, j);
                }
                if constexpr (RES_MODE == 2) {
                    silu_tile_init();
                    for (uint32_t j = 0; j < SBW; ++j) {
                        tk::round_full(j);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile(j, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, SBW);
            }
        }
        tk::pop(cb_acc, MT * NCT);
        return;
    }
    // the per-output-channel scales: acc (srcA) times the scale tile's row 0 (srcB, broadcast)
    reconfig_data_format(cb_w, cb_acc, cb_x, cb_s);
    pack_reconfig_data_format(cb_out);
    tk::mul_bcast_rows_init(cb_acc, cb_s);
    cb_wait_front(cb_acc, MT * NC);
    cb_wait_front(cb_s, NC);
    for (uint32_t m = 0; m < MT; ++m) {
        for (uint32_t s = 0; s < NS; ++s) {
            cb_reserve_back(cb_out, SBW);
            tile_regs_acquire();
            for (uint32_t j = 0; j < SBW; ++j) {
                mul_tiles_bcast_rows(cb_acc, cb_s, m * NC + s * SBW + j, s * SBW + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < SBW; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, SBW);
        }
    }
    tk::pop(cb_acc, MT * NC);
    tk::pop(cb_s, NC);
}
