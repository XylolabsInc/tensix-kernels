// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the prompt's causal conv (gdn_pconv_compute.cpp): the constant 0/1 tiles once, then per
// channel tile ct of this core's run: the conv state's tile (or zeros), the four row-replicated taps
// and the RT row tiles of x (the fused projection output's q|k|v columns, read in place: no slice), in one batch
// with one barrier.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_s = 1, cb_w = 2, cb_k = 3;

// bytes at addr zeroed by NOC reads of the core's zeros region (bytes a multiple of MEM_ZEROS_SIZE)
void noc_zero(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(z, addr + off, MEM_ZEROS_SIZE);
    }
}

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr uint32_t STATE = get_compile_time_arg_val(1);  // 0: no conv state (zeros)
    constexpr auto x_a = TensorAccessorArgs<2>();
    constexpr auto s_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto k_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t ct0 = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t CT = get_arg_val<uint32_t>(i++);       // channel tiles in all (the taps' pitch)
    const uint32_t x_pitch = get_arg_val<uint32_t>(i++);  // x's tiles a row tile
    const uint32_t x_col = get_arg_val<uint32_t>(i++);    // x's first channel tile
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t k_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tb = get_tile_size(cb_x);  // bf16 tiles throughout
    const auto x_acc = TensorAccessor(x_a, x_addr, tb);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);
    const auto w_acc = TensorAccessor(w_a, w_addr, tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb);

    {
        cb_reserve_back(cb_k, 12);
        const uint32_t kw = get_write_ptr(cb_k);
        for (uint32_t t = 0; t < 12; t++) {
            noc_async_read_page(t, k_acc, kw + t * tb);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k, 12);
    }
    for (uint32_t j = 0; j < n; j++) {
        DeviceZoneScopedN("R-ct");
        const uint32_t ct = ct0 + j;
        cb_reserve_back(cb_s, 1);
        cb_reserve_back(cb_w, 4);
        cb_reserve_back(cb_x, RT);
        const uint32_t sw = get_write_ptr(cb_s), ww = get_write_ptr(cb_w), xw = get_write_ptr(cb_x);
        if constexpr (STATE) {
            noc_async_read_page(ct, s_acc, sw);
        } else {
            noc_zero(sw, tb);
        }
        for (uint32_t k = 0; k < 4; k++) {
            noc_async_read_page(k * CT + ct, w_acc, ww + k * tb);
        }
        for (uint32_t rt = 0; rt < RT; rt++) {
            noc_async_read_page(rt * x_pitch + x_col + ct, x_acc, xw + rt * tb);
        }
        noc_async_read_barrier();
        cb_push_back(cb_s, 1);
        cb_push_back(cb_w, 4);
        cb_push_back(cb_x, RT);
    }
}
