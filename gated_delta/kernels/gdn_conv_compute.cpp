// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// The DeltaNet's causal conv1d (size 4) for T new tokens, on packed tiles (gdn_conv_reader.cpp puts
// up to 32 items, 32 channels each, into the rows of one tile per window row: w(0..2) the conv state,
// w(3 + t) token t). Per token t, in this order, accumulated in fp32:
//   out_t = silu(((w3 * w(t+3) + w0 * w(t)) + w1 * w(t+1)) + w2 * w(t+2) + bias)
// The order is the same for every t and every T, so a verify's row t (T > 1) is bit for bit the
// decode step (T = 1) that sees the same window. Upstream's _causal_conv1d_decode_t1 rounds every
// step to bf16 (tenstorrent/tt-metal v0.81.1, Apache-2.0). Compile-time args: T, BIAS (1: add the
// bias). Runtime args: the groups on this core.

#include <cstdint>
// DeviceZoneScopedN marks profiler zones and has no effect when profiling is disabled.
#include "api/debug/kernel_profiler.hpp"
#include "gdn_compat.h"

namespace {

constexpr uint32_t cb_in = 0;  // the window: T + 3 tiles
constexpr uint32_t cb_w = 1;   // w0, w1, w2, w3, bias
constexpr uint32_t cb_m0 = 2, cb_m1 = 3, cb_m2 = 4, cb_m3 = 5, cb_m4 = 6, cb_m5 = 7, cb_m6 = 8, cb_m7 = 9;
constexpr uint32_t cb_out = 16;
constexpr int ADD = 0, MUL = 2;

// o = a[ia] (op) b[ib], one tile (the inputs available; o pushed).
__attribute__((noinline)) void binop(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o, int op) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == MUL) {
        gdn::mul_init(a, b);
    } else {
        gdn::add_init(a, b);
    }
    tile_regs_acquire();
    if (op == MUL) {
        mul_tiles(a, b, ia, ib, 0);
    } else {
        add_tiles(a, b, ia, ib, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
    cb_push_back(o, 1);
}

__attribute__((noinline)) void silu_out(uint32_t in) {
    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    reconfig_data_format_srca(in);
    gdn::copy_init(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

// out_t for window rows t..t+3 of cb_in.
template <bool bias>
__attribute__((noinline)) void token(uint32_t t) {
    binop(cb_w, 3, cb_in, t + 3, cb_m0, MUL);  // w3 * x_t
    binop(cb_w, 0, cb_in, t, cb_m1, MUL);      // w0 * w(t)
    cb_wait_front(cb_m0, 1);
    cb_wait_front(cb_m1, 1);
    binop(cb_m0, 0, cb_m1, 0, cb_m2, ADD);
    gdn::pop(cb_m0, 1);
    gdn::pop(cb_m1, 1);
    binop(cb_w, 1, cb_in, t + 1, cb_m3, MUL);  // w1 * w(t+1)
    cb_wait_front(cb_m2, 1);
    cb_wait_front(cb_m3, 1);
    binop(cb_m2, 0, cb_m3, 0, cb_m4, ADD);
    gdn::pop(cb_m2, 1);
    gdn::pop(cb_m3, 1);
    binop(cb_w, 2, cb_in, t + 2, cb_m5, MUL);  // w2 * w(t+2)
    cb_wait_front(cb_m4, 1);
    cb_wait_front(cb_m5, 1);
    binop(cb_m4, 0, cb_m5, 0, cb_m6, ADD);
    gdn::pop(cb_m4, 1);
    gdn::pop(cb_m5, 1);
    cb_wait_front(cb_m6, 1);
    uint32_t acc = cb_m6;
    if constexpr (bias) {
        binop(cb_m6, 0, cb_w, 4, cb_m7, ADD);
        gdn::pop(cb_m6, 1);
        cb_wait_front(cb_m7, 1);
        acc = cb_m7;
    }
    silu_out(acc);
    gdn::pop(acc, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t BIAS = get_compile_time_arg_val(1);
    const uint32_t groups = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_w, cb_out);

    for (uint32_t g = 0; g < groups; g++) {
        {
            DeviceZoneScopedN("V-cwait");
            cb_wait_front(cb_in, T + 3);
            cb_wait_front(cb_w, 5);
        }
        for (uint32_t t = 0; t < T; t++) {
            DeviceZoneScopedN("V-ctok");
            token<BIAS != 0>(t);
        }
        gdn::pop(cb_in, T + 3);
        gdn::pop(cb_w, 5);
    }
}
