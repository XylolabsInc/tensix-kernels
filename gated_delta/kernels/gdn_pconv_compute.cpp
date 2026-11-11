// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// The prompt's causal conv1d (size 4, no bias) and SiLU over the DeltaNet's fused q|k|v channels, as
// upstream's _causal_conv1d_fir computes them (tenstorrent/tt-metal v0.81.1, models/experimental/
// gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py, Apache-2.0), bit for bit: per token t
//   out = x(t-3) * w0                       (mul_binary_tile: the SFPU's product, rounded to bf16)
//   out = addcmul(out, x(t-2), w1)          (addcmul_tile: out + x * w in one MAD, rounded to bf16)
//   out = addcmul(out, x(t-1), w2)
//   out = addcmul(out, x(t), w3)
//   out = silu(out)                         (silu_tile)
// with Dst in 16-bit mode, as each of those ops runs alone (fp32_dest_acc_en off), so every step
// stores bf16 into Dst as the op's own pack would. x(t - s) for row tile rt comes from the FPU:
// L_s @ X(rt) + C_s @ X(rt - 1) (or P_s @ the conv state for rt 0), 0/1 matrices, exact at HiFi4 for finite bf16 with
// |x| >= 2^-121 (one nonzero product an element; below that the low mantissa bits, denormals and -0 become +0, NaN
// becomes Inf: where the stock path's copy keeps them), written straight into the Dst slot the SFPU reads.
// Then the new conv state: rows 0..2 = x(T-3 .. T-1), by the selection matrices NS (host-made per T).
// Per channel tile (ct) the reader pushes the RT row tiles of x, the state tile and the four taps,
// each tap's row 0 copied to all 32 rows (the stock ops broadcast the row the same way).
// Compile-time args: RT (row tiles), NS_PREV, NS_STATE (the new state needs tile RT - 2, the old
// state). Runtime args: the channel tiles on this core.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "gdn_compat.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"

namespace {

constexpr uint32_t cb_x = 0;    // RT tiles a channel tile (double-buffered: 2 RT)
constexpr uint32_t cb_s = 1;    // the conv state's tile (zeros without a state)
constexpr uint32_t cb_w = 2;    // w0..w3, each row-replicated
constexpr uint32_t cb_k = 3;    // the constant 0/1 tiles (KI below), pushed once
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_ns = 17;  // the new state's tile
// cb_k's tiles: L_s (row r <- row r - s), C_s (rows r < s <- the previous tile's row 32 + r - s), P_s (rows r < s <-
// the state's row 3 + r - s), s = 1, 2, 3 at index s - 1; then NS_LAST, NS_PREV, NS_STATE.
constexpr uint32_t KL = 0, KC = 3, KP = 6, KNL = 9, KNP = 10, KNS = 11;
constexpr uint32_t ONE_F32 = 0x3f800000u;  // addcmul's value: 1.0 (the stock op's default)

}  // namespace

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr uint32_t NS_PREV = get_compile_time_arg_val(1);
    constexpr uint32_t NS_STATE = get_compile_time_arg_val(2);
    const uint32_t n = get_arg_val<uint32_t>(0);  // channel tiles on this core
    static_assert(RT >= 1, "pconv: at least one row tile");
    static_assert(!DST_ACCUM_MODE, "pconv: Dst in 16-bit mode, as the stock ops (Dst 0-7 is one half of it)");

    compute_kernel_hw_startup(cb_x, cb_k, cb_out);  // Initialize srcA, srcB, and the packer.
    cb_wait_front(cb_k, 12);
    for (uint32_t i = 0; i < n; i++) {
        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_w, 4);
        cb_wait_front(cb_x, RT);
        for (uint32_t rt = 0; rt < RT; rt++) {
            DeviceZoneScopedN("P-tile");
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            // x(t-3), x(t-2), x(t-1) into Dst 0, 2, 4 (zeroed by tile_regs_release): the matmul accumulates
            reconfig_data_format(cb_x, cb_k);  // matmul: in0 (the 0/1 tile) -> srcB, in1 (x) -> srcA
            gdn::mm_init(cb_k, cb_x, 0);
            for (uint32_t s = 3; s >= 1; s--) {
                const uint32_t d = 2 * (3 - s);
                matmul_tiles(cb_k, cb_x, KL + s - 1, rt, d);
                if (rt == 0) {
                    matmul_tiles(cb_k, cb_s, KP + s - 1, 0, d);
                } else {
                    matmul_tiles(cb_k, cb_x, KC + s - 1, rt - 1, d);
                }
            }
            // the taps into Dst 1, 3, 5, 7 and x(t) into Dst 6
            gdn::copy_init(cb_w);
            copy_tile(cb_w, 0, 1);
            copy_tile(cb_w, 1, 3);
            copy_tile(cb_w, 2, 5);
            copy_tile(cb_w, 3, 7);
            gdn::copy_init(cb_x);
            copy_tile(cb_x, rt, 6);
            // the stock ops' SFPU steps, in their order and operand order
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            addcmul_tile_init();
            addcmul_tile<DataFormat::Float16_b>(0, 2, 3, 0, ONE_F32);
            addcmul_tile<DataFormat::Float16_b>(0, 4, 5, 0, ONE_F32);
            addcmul_tile<DataFormat::Float16_b>(0, 6, 7, 0, ONE_F32);
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        {
            DeviceZoneScopedN("P-state");  // rows 0..2 = x(T-3 .. T-1)
            cb_reserve_back(cb_ns, 1);
            tile_regs_acquire();
            reconfig_data_format(cb_x, cb_k);
            gdn::mm_init(cb_k, cb_x, 0);
            matmul_tiles(cb_k, cb_x, KNL, RT - 1, 0);
            if constexpr (NS_PREV) {
                matmul_tiles(cb_k, cb_x, KNP, RT - 2, 0);
            }
            if constexpr (NS_STATE) {
                matmul_tiles(cb_k, cb_s, KNS, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_ns);
            tile_regs_release();
            cb_push_back(cb_ns, 1);
        }
        gdn::pop(cb_x, RT);
        gdn::pop(cb_s, 1);
        gdn::pop(cb_w, 4);
    }
}
