// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: the matmul_block loop rewritten for many rows over ternary tiles dequantized in L1.
// Adapted from tenstorrent/tt-metal v0.81.1,
// ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// Compute of the multi-row ternary matmul (t2.py t2m_linear): y[MT x NC tiles] = x[MT x KT] @ W,
// W = q * s with the trits q in BFP2_b (or BFP4_b) tiles and s one bf16 scale per output channel
// and group of GT K tiles (t2_compute.cpp has the format). For prefill and batches, where the
// one-token kernel's row spread does not apply: each weight tile is dequantized in L1 first,
//   Wd = q * s, s the group's row of the pass's scale tile broadcast over the tile's rows
//   (mul_tiles_bcast_rows with a row index; exact: s is bf16, q is -1, 0 or +1),
// packed in bf16, then the matmul with x (bf16), fp32 accumulation, every chunk's partial sums
// added up in the accumulator by the packer (L1 accumulation), as the dense reader's ROWS/XS order.
// Per chunk (KW K rows of the core's NC columns) and column subblock (SBW tiles): the KW x SBW
// tiles dequantized, then MT x SBW output tiles accumulated over the KW K tiles.
//
// CBs: c_0 x (MT x KW tiles per chunk, bf16), c_1 weights, c_2 scales (NC per pass, bf16), c_3
//      the accumulator (MT x NC, fp32), c_4 the odd weight chunks with SPLIT, c_6 the dequantized
//      subblock (KW x SBW, bf16), c_16 out (bf16).

#include "dense_ops.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_acc = 3, cb_w2 = 4, cb_wd = 6, cb_out = 16;

// srcA's unpacker reads BFP2_b (t2_compute.cpp)
// (tt-llk's internal unpacker reconfiguration: the compute API has no BFP2_b CB format to reconfigure to; not a
// documented kernel API, so it may change between releases: validated on tt-metal v0.81.x)
ALWI void srca_bfp2(uint32_t cb) {
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
        (uint32_t)DataFormat::Bfp2_b,
        (uint32_t)DataFormat::Bfp2_b,
        get_local_cb_interface(get_operand_id(cb)).fifo_page_size)));
}

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t KW = get_compile_time_arg_val(2);
    constexpr uint32_t GT = get_compile_time_arg_val(3);
    constexpr uint32_t SBW = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr bool SPLIT = get_compile_time_arg_val(6) != 0;
    constexpr bool BFP2 = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t PT = 32 * GT;
    constexpr uint32_t NCH = KT / KW;
    constexpr uint32_t CH = KW * NC;
    static_assert(KT % KW == 0 && KW % GT == 0 && PT % KW == 0 && NC % SBW == 0, "t2m: blocks");

    tk::mm_init(cb_x, cb_wd, cb_acc, SBW, MT, KW);
    cb_reserve_back(cb_acc, MT * NC);
    for (uint32_t c = 0; c < NCH; ++c) {
        const uint32_t kt0 = c * KW;
        if (kt0 % PT == 0) {
            cb_wait_front(cb_s, NC);
        }
        const uint32_t cw = (SPLIT && (c & 1)) ? cb_w2 : cb_w;
        cb_wait_front(cb_x, MT * KW);
        cb_wait_front(cw, CH);
        for (uint32_t s = 0; s < NS; ++s) {
            // the subblock's KW x SBW weight tiles times their group's scale row, in bf16
            reconfig_data_format(cw, cb_s);
            if constexpr (BFP2) {
                srca_bfp2(cw);
            }
            pack_reconfig_data_format(cb_wd);
            tk::mul_bcast_rows_init(cw, cb_s);
            cb_reserve_back(cb_wd, KW * SBW);
            for (uint32_t k = 0; k < KW; ++k) {
                const uint32_t row = ((kt0 + k) / GT) & 31;
                tile_regs_acquire();
                for (uint32_t j = 0; j < SBW; ++j) {
                    mul_tiles_bcast_rows(cw, cb_s, k * NC + s * SBW + j, s * SBW + j, j, row);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile(j, cb_wd);
                }
                tile_regs_release();
            }
            cb_push_back(cb_wd, KW * SBW);

            // the matmul: dst[m * SBW + j] += x[m * KW + k] * wd[k * SBW + j]
            reconfig_data_format(cb_wd, cb_x);
            pack_reconfig_data_format(cb_acc);
            tk::mm_init_short(cb_x, cb_wd, SBW, MT, KW);
            cb_wait_front(cb_wd, KW * SBW);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KW; ++k) {
                matmul_block(cb_x, cb_wd, k, k * SBW, 0, 0, SBW, MT, KW);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(c > 0 ? 1 : 0);
            for (uint32_t m = 0; m < MT; ++m) {
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile<true>(m * SBW + j, cb_acc, m * NC + s * SBW + j);
                }
            }
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            tk::pop(cb_wd, KW * SBW);
        }
        tk::pop(cw, CH);
        tk::pop(cb_x, MT * KW);
        if ((kt0 + KW) % PT == 0 || c == NCH - 1) {
            tk::pop(cb_s, NC);
        }
    }
    cb_push_back(cb_acc, MT * NC);

    // the output: the accumulator (fp32) in bf16
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    tk::copy_init(cb_acc);
    cb_wait_front(cb_acc, MT * NC);
    for (uint32_t m = 0; m < MT; ++m) {
        for (uint32_t s = 0; s < NS; ++s) {
            cb_reserve_back(cb_out, SBW);
            tile_regs_acquire();
            for (uint32_t j = 0; j < SBW; ++j) {
                copy_tile(cb_acc, m * NC + s * SBW + j, j);
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
}
