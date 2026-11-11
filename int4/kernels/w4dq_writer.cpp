// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of the W4 dequantize (w4dq_compute.cpp): the core's BFP8_b tiles of each K row (NC
// pages, the first nvalid real) into the BFP8 scratch weight, DRAM width-sharded over the banks
// (common/device.py sharded_memcfg: bank `bank`, tile (kt, col) at o_addr + (kt * pitch + col) *
// 1088): one contiguous write per K row.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16;
constexpr uint32_t BFP8_TILE = 1088;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t SB = get_compile_time_arg_val(2);

    uint32_t i = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t pitch = get_arg_val<uint32_t>(i++);  // tiles per row of the scratch's shard
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);

    for (uint32_t kt = 0; kt < KT; ++kt) {
        // the row's NC tiles arrive in groups of SB; they are contiguous in the CB (NC pages,
        // a multiple of SB, so a row never wraps)
        cb_wait_front(cb_out, NC);
        if (nvalid) {
            noc_async_write(
                get_read_ptr(cb_out),
                get_noc_addr_from_bank_id<true>(bank, o_addr + (kt * pitch + col0) * BFP8_TILE),
                nvalid * BFP8_TILE);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, NC);
    }
}
