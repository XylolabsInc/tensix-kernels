// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's gate_up: per slot s of this core the valid products (the expert's pairs held
// by this bank) to h's tiles s * PAIRS + first pair + p (h [1, k * I], interleaved).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_scr = 10, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t CPB = get_compile_time_arg_val(2);
    constexpr uint32_t BANKS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4);
    constexpr bool IDS32 = get_compile_time_arg_val(5) != 0;
    constexpr auto h_args = TensorAccessorArgs<6>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr uint32_t H = NC / 2;

    uint32_t i = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);

    const uint32_t ty = get_tile_size(cb_out);
    const auto h_acc = TensorAccessor(h_args, h_addr, ty);
    const auto i_acc = TensorAccessor(i_args, ids_addr, 64);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    volatile tt_l1_ptr uint32_t* ids32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);

    constexpr uint32_t Q = PAIRS / BANKS, R = PAIRS % BANKS;
    for (uint32_t s = j; s < K; s += CPB) {
        const uint32_t e = IDS32 ? ids32[s] : ids16[s];
        const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
        const uint32_t pc = Q + (rank < R ? 1 : 0);
        const uint32_t p0 = rank * Q + (rank < R ? rank : R);
        cb_wait_front(cb_out, H);
        const uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t p = 0; p < pc; ++p) {
            noc_async_write_page(s * PAIRS + p0 + p, h_acc, rd + p * ty);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, H);
    }
}
