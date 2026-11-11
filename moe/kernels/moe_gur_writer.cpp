// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's gate_up_rows: per unit (union index q of this core, moe8_rows.h) the valid
// products (expert list[q]'s pairs held by this bank) to h's tiles q * PAIRS + first pair + p
// (h [1, 1, 32, u_max * I], interleaved: union index q's I columns at q * I). With G column groups each union index
// is G units, group gi's products the pairs gi * NC / 2 .. of the expert's here.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe8_rows.h"

constexpr uint32_t cb_scr = 10, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t CPB = get_compile_time_arg_val(2);
    constexpr uint32_t BANKS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4);
    constexpr bool IDS32 = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t ROWS = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);  // column groups per union index
    constexpr auto h_args = TensorAccessorArgs<9>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr uint32_t H = NC / 2;
    constexpr uint32_t OFF_BITS = 4096, OFF_LIST = OFF_BITS + 128;

    uint32_t i = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);

    const uint32_t ty = get_tile_size(cb_out);
    const auto h_acc = TensorAccessor(h_args, h_addr, ty);
    const auto i_acc = TensorAccessor(i_args, ids_addr, IDS32 ? 4096 : 2048);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, IDS32 ? 4096 : 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_BITS);
    volatile tt_l1_ptr uint16_t* list = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_LIST);
    const uint32_t u = moe8r::build_union<IDS32, K, E>(scr, ROWS, bits, list);

    constexpr uint32_t Q = PAIRS / BANKS, R = PAIRS % BANKS;
    for (uint32_t q = j; q < u; q += CPB) {
        const uint32_t e = list[q];
        const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
        const uint32_t pc = Q + (rank < R ? 1 : 0);
        const uint32_t p0 = rank * Q + (rank < R ? rank : R);
        for (uint32_t gi = 0; gi < G; ++gi) {
            const uint32_t gp = pc > gi * H ? (pc - gi * H < H ? pc - gi * H : H) : 0;
            cb_wait_front(cb_out, H);
            const uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t p = 0; p < gp; ++p) {
                noc_async_write_page(q * PAIRS + p0 + gi * H + p, h_acc, rd + p * ty);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, H);
        }
    }
}
