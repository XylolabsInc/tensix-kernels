// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: the matmul_block loop rewritten for int4 tiles with group scales along K (one
// token spread over the tile rows). Adapted from tenstorrent/tt-metal v0.81.1,
// ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// Compute of the W4A16 matmul with group scales along K (w4a16.py): one token, y[n] = sum over K
// tiles t of s[t, n] * (x_t . q_t[:, n]), q the int4 codes (-7..7) read as BFP4_b tiles under a
// constant exponent (the reader; each value is the integer exactly), s one bf16 scale per output
// channel and 32-row K tile (G = 32).
//
// The group scale varies along K, so it cannot be applied after the K sum. One token uses one row
// of a matmul's 32, so the reader hands over x row-spread: activation tile t holds x's K tile t
// in row t % 32 and zeros elsewhere. Then 32 consecutive K tiles (a pass) accumulate in dest with
// row r holding exactly K tile 32p + r's partial product, D[r, n] = x_t . q_t[:, n]. Per pass and
// column subblock: D times the pass's scale tile S_p (S_p[r, n] = s[32p + r, n]: the scale
// tensor's own tile), then the column sum (a matmul with a tile of ones in row 0) added to the
// output in L1 (packer L1 accumulation). Three extra packs per 32 weight tiles.
//
// CBs: c_0 x row-spread (KB tiles, bf16), c_1 weights (BFP4_b pages), c_2 scales (2 SBW, bf16),
//      c_3 acc (NC tiles, fp32, unpacked to dest), c_5 ones (1, bf16), c_6 D (SBW, fp32),
//      c_7 D * S (SBW, fp32), c_4 the odd weight chunks with SPLIT, c_16 out (bf16).

#include "dense_ops.h"
#include "api/compute/eltwise_binary.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_acc = 3, cb_w2 = 4, cb_one = 5, cb_d = 6, cb_ds = 7;
constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t KW = get_compile_time_arg_val(2);
    constexpr uint32_t SBW = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr bool SPLIT = get_compile_time_arg_val(5) != 0;  // odd chunks in cb_w2 (the writer's)
    constexpr uint32_t NKB = KT / KB;
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NP = KB / 32;  // passes per x block
    constexpr uint32_t NKW = 32 / KW;  // weight chunks per pass
    constexpr uint32_t CH = KW * SBW;
    static_assert(KT % KB == 0 && KB % 32 == 0 && 32 % KW == 0 && NC % SBW == 0, "w4a16: blocks");

    tk::mm_init(cb_x, cb_w, cb_d, SBW, 1, KB);
    cb_wait_front(cb_one, 1);
    cb_reserve_back(cb_acc, NC);
    uint32_t g = 0;
    for (uint32_t b = 0; b < NKB; ++b) {
        cb_wait_front(cb_x, KB);
        for (uint32_t s = 0; s < NS; ++s) {
            for (uint32_t p = 0; p < NP; ++p) {
                // D: the pass's 32 K tiles, row r = K tile 32p + r
                cb_reserve_back(cb_d, SBW);
                tile_regs_acquire();
                for (uint32_t c = 0; c < NKW; ++c, ++g) {
                    const uint32_t cw = (SPLIT && (g & 1)) ? cb_w2 : cb_w;
                    cb_wait_front(cw, CH);
                    for (uint32_t k = 0; k < KW; ++k) {
                        matmul_block(cb_x, cw, p * 32 + c * KW + k, k * SBW, 0, 0, SBW, 1, KB);
                    }
                    tk::pop(cw, CH);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_d);
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile(j, cb_d);
                }
                tile_regs_release();
                cb_push_back(cb_d, SBW);

                // D * S_p
                reconfig_data_format(cb_w, cb_d, cb_x, cb_s);
                tk::mul_init(cb_d, cb_s);
                cb_wait_front(cb_d, SBW);
                cb_wait_front(cb_s, SBW);
                cb_reserve_back(cb_ds, SBW);
                tile_regs_acquire();
                for (uint32_t j = 0; j < SBW; ++j) {
                    mul_tiles(cb_d, cb_s, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_ds);
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile(j, cb_ds);
                }
                tile_regs_release();
                cb_push_back(cb_ds, SBW);
                tk::pop(cb_d, SBW);
                tk::pop(cb_s, SBW);

                // the column sums (ones in row 0 times D * S) into the output, L1 accumulated
                reconfig_data_format(cb_d, cb_ds, cb_s, cb_one);
                tk::mm_init_short(cb_one, cb_ds, 1, 1, 1);
                cb_wait_front(cb_ds, SBW);
                tile_regs_acquire();
                for (uint32_t j = 0; j < SBW; ++j) {
                    matmul_block(cb_one, cb_ds, 0, j, j, 0, 1, 1, 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_acc);
                pack_reconfig_l1_acc((b > 0 || p > 0) ? 1 : 0);
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile<true>(j, cb_acc, s * SBW + j);
                }
                pack_reconfig_l1_acc(0);
                tile_regs_release();
                tk::pop(cb_ds, SBW);

                // back to the main matmul
                reconfig_data_format(cb_ds, cb_w, cb_one, cb_x);
                tk::mm_init_short(cb_x, cb_w, SBW, 1, KB);
            }
        }
        tk::pop(cb_x, KB);
    }
    cb_push_back(cb_acc, NC);

    // the output: acc (fp32, row 0 holds y, rows 1-31 zero) in bf16
    reconfig_data_format_srca(cb_w, cb_acc);
    pack_reconfig_data_format(cb_out);
    tk::copy_init(cb_acc);
    cb_wait_front(cb_acc, NC);
    for (uint32_t s = 0; s < NS; ++s) {
        cb_reserve_back(cb_out, SBW);
        tile_regs_acquire();
        for (uint32_t j = 0; j < SBW; ++j) {
            copy_tile(cb_acc, s * SBW + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < SBW; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, SBW);
    }
    tk::pop(cb_acc, NC);
    tk::pop(cb_one, 1);
}
