// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Write prompt gates from gdn_pgates_compute.cpp. For each assigned row tile, write
// NT tiles of g and beta ([1, T, HV], page rt * NT + c).

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_g = 16, cb_beta = 17;

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr auto g_a = TensorAccessorArgs<1>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t rt0 = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t g_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tb = get_tile_size(cb_g);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb);
    for (uint32_t j = 0; j < n; j++) {
        const uint32_t rt = rt0 + j;
        for (uint32_t c = 0; c < NT; c++) {
            cb_wait_front(cb_g, 1);
            cb_wait_front(cb_beta, 1);
            noc_async_write_page(rt * NT + c, g_acc, get_read_ptr(cb_g));
            noc_async_write_page(rt * NT + c, b_acc, get_read_ptr(cb_beta));
            noc_async_writes_flushed();
            cb_pop_front(cb_g, 1);
            cb_pop_front(cb_beta, 1);
        }
    }
    noc_async_write_barrier();
}
