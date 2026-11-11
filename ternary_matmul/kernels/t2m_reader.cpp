// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of the multi-row ternary matmul (t2m_compute.cpp; t2.py t2m_linear): MT tile rows of x
// (already in the weights' basis: t2.rotate_rows) streamed with the weights.
//   - per pass of 32 groups: the scale tiles of the core's columns (as t2_reader.cpp);
//   - per chunk of KW K rows: x's MT x KW tiles (tile rows mt0 .. mt0 + MT - 1, page m * KW + k)
//     and the chunk's weights (t2_wload.h; with SPLIT the even chunks only).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "t2_wload.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2;

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t KW = get_compile_time_arg_val(2);
    constexpr uint32_t GT = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr bool SPLIT = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t TB = get_compile_time_arg_val(6);
    constexpr uint32_t DB = get_compile_time_arg_val(7);
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(8);
    constexpr uint32_t EXP = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t PT = 32 * GT;
    constexpr uint32_t NCH = KT / KW;
    static_assert(KT % KW == 0 && KW % GT == 0 && PT % KW == 0, "t2m: chunks of whole groups");

    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t core_off = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t nst = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);
    const uint32_t mt0 = get_arg_val<uint32_t>(i++);  // x's first tile row of this call

    const uint32_t tx = get_tile_size(cb_x);
    const uint32_t ts = get_tile_size(cb_s);
    const auto x_acc = TensorAccessor(x_args, x_addr, tx);
    const auto s_acc = TensorAccessor(s_args, s_addr, ts);
    t2::Chunks<KW, NC, TB, DB> ld{cb_w, bank, w_addr + core_off, nvalid, vc};
    if constexpr (DB < TB) {
        t2::init_exponents(cb_w, W_PAGES, TB, EXP);
        cb_reserve_back(14, 1);  // the staging buffer: scratch, never pushed
        ld.stage = get_write_ptr(14);
    }

    for (uint32_t c = 0; c < NCH; ++c) {
        const uint32_t kt0 = c * KW;
        if (kt0 % PT == 0) {
            cb_reserve_back(cb_s, NC);
            const uint32_t ls = get_write_ptr(cb_s);
            for (uint32_t j = 0; j < nvalid; ++j) {
                noc_async_read_page((kt0 / PT) * nst + n0 + j, s_acc, ls + j * ts);
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, NC);
        }
        cb_reserve_back(cb_x, MT * KW);
        const uint32_t lx = get_write_ptr(cb_x);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t k = 0; k < KW; ++k) {
                noc_async_read_page((mt0 + m) * KT + kt0 + k, x_acc, lx + (m * KW + k) * tx);
            }
        }
        const bool mine = !SPLIT || (c & 1) == 0;
        if (mine) {
            ld.issue(c);
        }
        noc_async_read_barrier();
        if (mine) {
            if constexpr (DB < TB) {
                ld.place();
                noc_async_read_barrier();
            }
            ld.push();
        }
        cb_push_back(cb_x, MT * KW);
    }
}
