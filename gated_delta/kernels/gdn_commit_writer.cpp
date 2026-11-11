// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Write committed pages from gdn_commit_reader.cpp to their destinations in batches.
// Use the same pair table and page traversal as the reader.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_c = 0, cb_t = 2;  // the writer's own copy of the table

void kernel_main() {
    constexpr uint32_t BATCH = get_compile_time_arg_val(0);
    constexpr uint32_t TBYTES = get_compile_time_arg_val(1);
    constexpr auto d_a = TensorAccessorArgs<3>();
    constexpr auto t_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    const uint32_t t_addr = get_arg_val<uint32_t>(0);
    uint32_t p = get_arg_val<uint32_t>(1);
    uint32_t off = get_arg_val<uint32_t>(2);
    uint32_t left = get_arg_val<uint32_t>(3);
    constexpr uint32_t PB = get_compile_time_arg_val(2);  // a slot: the largest pair page (not the CB's tile size)

    const auto t_acc = TensorAccessor(t_a, t_addr, TBYTES);
    const uint32_t tw = get_write_ptr(cb_t);
    noc_async_read_page(0, t_acc, tw);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tw);
    while (left > 0) {
        DeviceZoneScopedN("C-write");
        cb_wait_front(cb_c, BATCH);
        const uint32_t r = get_read_ptr(cb_c);
        uint32_t k = 0;
        while (k < BATCH && left > 0) {
            const uint32_t dst = tab[4 * p + 1], np = tab[4 * p + 2], pb = tab[4 * p + 3];
            const auto d_acc = TensorAccessor(d_a, dst, pb);
            noc_async_write_page(off, d_acc, r + k * PB);
            k++;
            left--;
            if (++off == np) {
                off = 0;
                p++;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_c, BATCH);
    }
    noc_async_write_barrier();
}
