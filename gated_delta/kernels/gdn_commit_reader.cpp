// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the verify's commit (gdn_commit.py): many page-for-page copies in one launch. The pair table (a DRAM
// page of P records: src address, dst address, pages, page bytes) is read once; this core takes `count` pages of the
// pairs' pages laid end to end, from pair p0's page off0, and reads them into cb_c in batches of up to BATCH pages; the
// writer (gdn_commit_writer.cpp) writes each batch to its pair's dst. Every pair is interleaved in DRAM (one accessor
// layout, with its own base address and page size).

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_c = 0, cb_t = 1;

void kernel_main() {
    constexpr uint32_t BATCH = get_compile_time_arg_val(0);
    constexpr uint32_t TBYTES = get_compile_time_arg_val(1);  // the table page's bytes
    constexpr auto d_a = TensorAccessorArgs<3>();               // the pairs' layout (interleaved DRAM)
    constexpr auto t_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();  // the table
    const uint32_t t_addr = get_arg_val<uint32_t>(0);
    uint32_t p = get_arg_val<uint32_t>(1);    // the first pair
    uint32_t off = get_arg_val<uint32_t>(2);  // its first page
    uint32_t left = get_arg_val<uint32_t>(3); // pages on this core
    constexpr uint32_t PB = get_compile_time_arg_val(2);  // a slot: the largest pair page (not the CB's tile size)

    const auto t_acc = TensorAccessor(t_a, t_addr, TBYTES);
    const uint32_t tw = get_write_ptr(cb_t);
    noc_async_read_page(0, t_acc, tw);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tw);
    while (left > 0) {
        DeviceZoneScopedN("C-read");
        cb_reserve_back(cb_c, BATCH);
        const uint32_t w = get_write_ptr(cb_c);
        uint32_t k = 0;
        while (k < BATCH && left > 0) {
            const uint32_t src = tab[4 * p], np = tab[4 * p + 2], pb = tab[4 * p + 3];
            const auto s_acc = TensorAccessor(d_a, src, pb);
            noc_async_read_page(off, s_acc, w + k * PB);
            k++;
            left--;
            if (++off == np) {
                off = 0;
                p++;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_c, BATCH);  // a short last batch: the writer counts its own pages
    }
}
