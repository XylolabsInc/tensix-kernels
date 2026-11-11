// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Write prompt convolution results from gdn_pconv_compute.cpp. For each channel tile ct,
// write RT output row tiles ([1, T, C], page rt * CT + ct). Write the new state tile to
// fused state [1, 3, C] at page ct and to its q, k, or v state ([1, 3, q_dim], etc.).

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16, cb_ns = 17;

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    constexpr auto f_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto q_a = TensorAccessorArgs<f_a.next_compile_time_args_offset()>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t ct0 = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t CT = get_arg_val<uint32_t>(i++);
    const uint32_t QT = get_arg_val<uint32_t>(i++);  // q's channel tiles
    const uint32_t KT = get_arg_val<uint32_t>(i++);  // k's channel tiles
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t f_addr = get_arg_val<uint32_t>(i++);
    const uint32_t q_addr = get_arg_val<uint32_t>(i++);
    const uint32_t k_addr = get_arg_val<uint32_t>(i++);
    const uint32_t v_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    const auto f_acc = TensorAccessor(f_a, f_addr, tb);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb);

    for (uint32_t j = 0; j < n; j++) {
        const uint32_t ct = ct0 + j;
        for (uint32_t rt = 0; rt < RT; rt++) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(rt * CT + ct, o_acc, get_read_ptr(cb_out));
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
        cb_wait_front(cb_ns, 1);
        const uint32_t r = get_read_ptr(cb_ns);
        noc_async_write_page(ct, f_acc, r);
        if (ct < QT) {
            noc_async_write_page(ct, q_acc, r);
        } else if (ct < QT + KT) {
            noc_async_write_page(ct - QT, k_acc, r);
        } else {
            noc_async_write_page(ct - QT - KT, v_acc, r);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_ns, 1);
    }
    noc_async_write_barrier();
}
