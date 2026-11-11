// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: the matmul_block loop rewritten for ternary BFP2_b tiles with group scales and a
// fused block Hadamard. Adapted from tenstorrent/tt-metal v0.81.1,
// ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// Compute of the ternary matmul with group scales along K (t2.py): one token,
// y[n] = sum over groups g of s[g, n] * (x_g . q_g[:, n]), q the trits (-1, 0, +1) as BFP2_b tiles
// whose exponents read every value exactly (or, the fallback, the same values as BFP4_b tiles), s
// one bf16 scale per output channel and group of G = 32 * GT inputs.
//
// As in w4a16_compute.cpp, the token is spread over the activation tile's rows: x tile kt holds x's
// K tile kt in row (kt / GT) % 32 and zeros elsewhere, so the products of the 32 groups of a pass
// land in 32 different rows, D[r, n] = x_g . q_g[:, n] for group g = 32p + r. Here the weights
// stream in chunks of KW full K rows of the core's NC columns (KW a multiple of GT), each chunk's
// product packed into D with packer L1 accumulation (the first chunk of a pass overwrites). Per
// pass: D times the scale tensor's tile S_p (S_p[r, n] = s[32p + r, n]), then its column sum (a
// matmul with ones in row 0) added to the output accumulator (L1 accumulation), then that in bf16.
//
// BFP2 (1): the weight CBs are declared BFP4_b (ttnn has no BFP2 dtype) with 320-byte pages; the
// srcA unpacker is switched to Bfp2_b after every format change that touches it (srca_bfp2). The
// matmul unpacker steps through a CB by its page size, so nothing else changes.
//
// NB > 0: first the block Hadamard of x (t2_reader.cpp): per block Z = X_b H_32, then
// Y_b = H_32 Z (= 32 x'_b; the scales carry the 1/32), packed in bf16 for the reader, which spreads
// x' from them.
//
// CBs: c_0 x ring (bf16), c_1 weights, c_2 scales (bf16), c_3 the output accumulator (NC, fp32),
//      c_4 the odd weight chunks with SPLIT, c_5 ones (bf16), c_6 D (NC, fp32), c_7 D * S (SBW,
//      fp32), c_16 out (bf16); with NB: c_8 X_b (bf16), c_10 Z (fp32), c_11 Y (NB, bf16), c_12 H_32.

#include "dense_ops.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_acc = 3, cb_w2 = 4, cb_one = 5, cb_d = 6, cb_ds = 7;
constexpr uint32_t cb_xb = 8, cb_z = 10, cb_y = 11, cb_h = 12, cb_out = 16;

// srcA's unpacker reads BFP2_b from now on, into srcA as BFP2_b (a block-float CB unpacks to its
// own format; Blackhole's ALU infers the format), with cb's page size
// (tt-llk's internal unpacker reconfiguration: the compute API has no BFP2_b CB format to reconfigure to; not a
// documented kernel API, so it may change between releases: validated on tt-metal v0.81.x)
ALWI void srca_bfp2(uint32_t cb) {
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
        (uint32_t)DataFormat::Bfp2_b,
        (uint32_t)DataFormat::Bfp2_b,
        get_local_cb_interface(get_operand_id(cb)).fifo_page_size)));
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KW = get_compile_time_arg_val(1);
    constexpr uint32_t GT = get_compile_time_arg_val(2);
    constexpr uint32_t SBW = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr bool SPLIT = get_compile_time_arg_val(5) != 0;  // odd chunks in cb_w2 (the writer's)
    constexpr bool BFP2 = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t NB = get_compile_time_arg_val(7);  // Hadamard blocks (0: none)
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t PT = 32 * GT;   // K tiles per pass
    constexpr uint32_t CPP = PT / KW;  // chunks per full pass
    constexpr uint32_t NCH = KT / KW;
    constexpr uint32_t NP = (KT + PT - 1) / PT;
    constexpr uint32_t CH = KW * NC;
    static_assert(KT % KW == 0 && KW % GT == 0 && PT % KW == 0 && NC % SBW == 0, "t2: blocks");

    if constexpr (NB > 0) {
        // Z = X_b H (X in srcB, H in srcA), Y_b = H Z (H in srcB, Z in srcA)
        tk::mm_init(cb_xb, cb_h, cb_z, 1, 1, 1);
        cb_wait_front(cb_h, 1);
        for (uint32_t b = 0; b < NB; ++b) {
            cb_wait_front(cb_xb, 1);
            cb_reserve_back(cb_z, 1);
            tile_regs_acquire();
            matmul_block(cb_xb, cb_h, 0, 0, 0, 0, 1, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_z);
            tile_regs_release();
            cb_push_back(cb_z, 1);
            tk::pop(cb_xb, 1);

            reconfig_data_format(cb_z, cb_h);
            pack_reconfig_data_format(cb_y);
            tk::mm_init_short(cb_h, cb_z, 1, 1, 1);
            cb_wait_front(cb_z, 1);
            cb_reserve_back(cb_y, 1);
            tile_regs_acquire();
            matmul_block(cb_h, cb_z, 0, 0, 0, 0, 1, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
            cb_push_back(cb_y, 1);
            tk::pop(cb_z, 1);

            reconfig_data_format(cb_h, cb_xb);
            pack_reconfig_data_format(cb_z);
            tk::mm_init_short(cb_xb, cb_h, 1, 1, 1);
        }
        tk::pop(cb_h, 1);
        reconfig_data_format(cb_w, cb_x);
        pack_reconfig_data_format(cb_d);
        tk::mm_init_short(cb_x, cb_w, SBW, 1, KW);
    } else {
        tk::mm_init(cb_x, cb_w, cb_d, SBW, 1, KW);
    }
    if constexpr (BFP2) {
        srca_bfp2(cb_w);
    }
    cb_wait_front(cb_one, 1);
    cb_reserve_back(cb_acc, NC);
    uint32_t c = 0;  // running chunk index: with SPLIT the odd ones come from cb_w2
    for (uint32_t p = 0; p < NP; ++p) {
        const uint32_t c1 = (c + CPP < NCH) ? c + CPP : NCH;
        // D: the pass's chunks, row r = group 32p + r
        cb_reserve_back(cb_d, NC);
        for (uint32_t first = c; c < c1; ++c) {
            const uint32_t cw = (SPLIT && (c & 1)) ? cb_w2 : cb_w;
            cb_wait_front(cb_x, KW);
            cb_wait_front(cw, CH);
            for (uint32_t s = 0; s < NS; ++s) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < KW; ++k) {
                    // dst[j] += x[k] * w[k * NC + s*SBW + j]
                    matmul_block(cb_x, cw, k, k * NC + s * SBW, 0, 0, SBW, 1, KW);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(c > first ? 1 : 0);
                for (uint32_t j = 0; j < SBW; ++j) {
                    pack_tile<true>(j, cb_d, s * SBW + j);
                }
                tile_regs_release();
            }
            tk::pop(cw, CH);
            tk::pop(cb_x, KW);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_d, NC);

        cb_wait_front(cb_d, NC);
        cb_wait_front(cb_s, NC);
        for (uint32_t s = 0; s < NS; ++s) {
            // D * S_p (D in srcA, S in srcB)
            reconfig_data_format(cb_d, cb_s);
            pack_reconfig_data_format(cb_ds);
            tk::mul_init(cb_d, cb_s);
            cb_reserve_back(cb_ds, SBW);
            tile_regs_acquire();
            for (uint32_t j = 0; j < SBW; ++j) {
                mul_tiles(cb_d, cb_s, s * SBW + j, s * SBW + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < SBW; ++j) {
                pack_tile(j, cb_ds);
            }
            tile_regs_release();
            cb_push_back(cb_ds, SBW);

            // the column sums (ones in row 0 times D * S) into the output, L1 accumulated
            reconfig_data_format(cb_ds, cb_one);
            tk::mm_init_short(cb_one, cb_ds, 1, 1, 1);
            cb_wait_front(cb_ds, SBW);
            tile_regs_acquire();
            for (uint32_t j = 0; j < SBW; ++j) {
                matmul_block(cb_one, cb_ds, 0, j, j, 0, 1, 1, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_acc);
            pack_reconfig_l1_acc(p > 0 ? 1 : 0);
            for (uint32_t j = 0; j < SBW; ++j) {
                pack_tile<true>(j, cb_acc, s * SBW + j);
            }
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            tk::pop(cb_ds, SBW);
        }
        tk::pop(cb_d, NC);
        tk::pop(cb_s, NC);

        // back to the main matmul
        reconfig_data_format(cb_w, cb_x);
        pack_reconfig_data_format(cb_d);
        tk::mm_init_short(cb_x, cb_w, SBW, 1, KW);
        if constexpr (BFP2) {
            srca_bfp2(cb_w);
        }
    }
    cb_push_back(cb_acc, NC);

    // the output: acc (fp32, row 0 holds y, rows 1-31 zero) in bf16
    reconfig_data_format_srca(cb_acc);
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
