// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the prompt's gates (gdn_pgates_compute.cpp): once, dt_bias's and A_neg's row-replicated tiles and the two
// 0/1 column-shift tiles; then per row tile of this core's run, a's NT tiles and b's NB source tiles, read in place
// from the fused projection output, with one barrier per row tile.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_a = 0, cb_b = 1, cb_db = 2, cb_an = 3, cb_k = 4;

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr auto x_a = TensorAccessorArgs<2>();
    constexpr auto c_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();  // dt_bias, A_neg, M, M2
    uint32_t i = 0;
    const uint32_t rt0 = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t x_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t a_t0 = get_arg_val<uint32_t>(i++);  // a's first tile column in the fused projection output
    const uint32_t b_t0 = get_arg_val<uint32_t>(i++);  // the tile column b starts in
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t c_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tb = get_tile_size(cb_a);
    const auto x_acc = TensorAccessor(x_a, x_addr, tb);
    const auto c_acc = TensorAccessor(c_a, c_addr, tb);

    cb_reserve_back(cb_db, NT);
    cb_reserve_back(cb_an, NT);
    cb_reserve_back(cb_k, 2);
    for (uint32_t t = 0; t < NT; t++) {
        noc_async_read_page(t, c_acc, get_write_ptr(cb_db) + t * tb);
        noc_async_read_page(NT + t, c_acc, get_write_ptr(cb_an) + t * tb);
    }
    noc_async_read_page(2 * NT, c_acc, get_write_ptr(cb_k));
    noc_async_read_page(2 * NT + 1, c_acc, get_write_ptr(cb_k) + tb);
    noc_async_read_barrier();
    cb_push_back(cb_db, NT);
    cb_push_back(cb_an, NT);
    cb_push_back(cb_k, 2);
    for (uint32_t j = 0; j < n; j++) {
        DeviceZoneScopedN("R-rt");
        const uint32_t rt = rt0 + j;
        cb_reserve_back(cb_a, NT);
        cb_reserve_back(cb_b, NB);
        const uint32_t wa = get_write_ptr(cb_a), wb = get_write_ptr(cb_b);
        for (uint32_t t = 0; t < NT; t++) {
            noc_async_read_page(rt * x_pitch + a_t0 + t, x_acc, wa + t * tb);
        }
        for (uint32_t t = 0; t < NB; t++) {
            noc_async_read_page(rt * x_pitch + b_t0 + t, x_acc, wb + t * tb);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, NT);
        cb_push_back(cb_b, NB);
    }
}
