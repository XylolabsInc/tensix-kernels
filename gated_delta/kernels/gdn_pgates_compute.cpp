// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// The prompt's gates of the DeltaNet, as the stock gate ops compute them (tenstorrent/tt-metal v0.81.1's
// models/experimental/gated_attention_gated_deltanet, Apache-2.0), bit for bit, per row tile and gate tile c:
//   beta = sigmoid(b)                        (sigmoid_tile)
//   g    = A_neg * softplus(a + dt_bias)     (add_tiles on the FPU, softplus_tile, mul_binary_tile)
// with Dst in 16-bit mode as each stock op runs. a is the fused projection output's tiles from a's first column as they are (a's
// first column sits on a tile, as the stock slice keeps whole tiles); b, which need not start on a tile, is moved to
// column 0 by exact 0/1 matmuls (S @ M + next S @ M2) instead of the stock untilize, slice and tilize. dt_bias and A_neg
// come row-replicated (the stock ops broadcast their row the same way). Compile-time args: NT (gate tiles a row tile),
// SHIFT (b's column within its tile is not 0), NB (b's source tiles a row tile). Runtime args: this core's row tiles.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "gdn_compat.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/softplus.h"

namespace {

constexpr uint32_t cb_a = 0;    // NT tiles a row tile: a's tiles
constexpr uint32_t cb_b = 1;    // NB tiles a row tile: the tiles b starts in and after
constexpr uint32_t cb_db = 2;   // NT tiles, once: dt_bias, row-replicated
constexpr uint32_t cb_an = 3;   // NT tiles, once: A_neg, row-replicated
constexpr uint32_t cb_k = 4;    // 2 tiles, once: M (column c -> c - off), M2 (c -> c + 32 - off)
constexpr uint32_t cb_g = 16, cb_beta = 17;
constexpr uint32_t ONE = 0x3f800000u, TWENTY = 0x41a00000u;  // softplus: beta 1, 1 / beta 1, threshold 20

}  // namespace

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t SHIFT = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    const uint32_t n = get_arg_val<uint32_t>(0);  // row tiles on this core

    compute_kernel_hw_startup(cb_a, cb_db, cb_g);
    cb_wait_front(cb_db, NT);
    cb_wait_front(cb_an, NT);
    cb_wait_front(cb_k, 2);
    for (uint32_t i = 0; i < n; i++) {
        cb_wait_front(cb_a, NT);
        cb_wait_front(cb_b, NB);
        for (uint32_t c = 0; c < NT; c++) {
            DeviceZoneScopedN("G-tile");
            cb_reserve_back(cb_g, 1);
            cb_reserve_back(cb_beta, 1);
            tile_regs_acquire();
            // the FPU's part: A_neg into Dst 0, a + dt_bias into Dst 1, b's tile into Dst 2
            gdn::copy_init(cb_an);
            copy_tile(cb_an, c, 0);
            reconfig_data_format(cb_a, cb_db);
            gdn::add_init(cb_a, cb_db);
            add_tiles(cb_a, cb_db, c, c, 1);
            reconfig_data_format(cb_k, cb_b);  // matmul: in0 (b's source) -> srcB, in1 (M) -> srcA
            gdn::mm_init(cb_b, cb_k, 0);
            matmul_tiles(cb_b, cb_k, c, 0, 2);
            if constexpr (SHIFT) {
                if (c + 1 < NB) {
                    matmul_tiles(cb_b, cb_k, c + 1, 1, 2);
                }
            }
            // the SFPU's, in the stock ops' order and operand order
            softplus_tile_init();
            softplus_tile(1, ONE, ONE, TWENTY);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            sigmoid_tile_init();
            sigmoid_tile(2);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_g);
            pack_tile(2, cb_beta);
            tile_regs_release();
            cb_push_back(cb_g, 1);
            cb_push_back(cb_beta, 1);
        }
        gdn::pop(cb_a, NT);
        gdn::pop(cb_b, NB);
    }
}
