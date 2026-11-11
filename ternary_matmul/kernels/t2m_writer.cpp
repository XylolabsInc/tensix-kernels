// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of the multi-row ternary matmul (t2m_compute.cpp). With SPLIT it first loads the odd
// weight chunks into the second weight CB (c_4; t2_wload.h). Then the core's output tiles, tile
// row by tile row in groups of SBW, into y [MT_y x nt tiles] (interleaved, page row * nt + column)
// from tile row mt0; of the core's NC columns only the first nvalid are written.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "t2_wload.h"

constexpr uint32_t cb_w2 = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t KW = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t SBW = get_compile_time_arg_val(4);
    constexpr bool SPLIT = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t TB = get_compile_time_arg_val(6);
    constexpr uint32_t DB = get_compile_time_arg_val(7);
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(8);
    constexpr uint32_t EXP = get_compile_time_arg_val(9);
    constexpr auto y_args = TensorAccessorArgs<10>();
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NCH = KT / KW;

    uint32_t i = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(i++);
    const uint32_t nt = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t core_off = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);
    const uint32_t mt0 = get_arg_val<uint32_t>(i++);

    if constexpr (SPLIT) {
        t2::Chunks<KW, NC, TB, DB> ld{cb_w2, bank, w_addr + core_off, nvalid, vc};
        if constexpr (DB < TB) {
            t2::init_exponents(cb_w2, W_PAGES, TB, EXP);
            cb_reserve_back(15, 1);  // the staging buffer: scratch, never pushed
            ld.stage = get_write_ptr(15);
        }
        for (uint32_t c = 1; c < NCH; c += 2) {
            ld.issue(c);
            noc_async_read_barrier();
            if constexpr (DB < TB) {
                ld.place();
                noc_async_read_barrier();
            }
            ld.push();
        }
    }

    const uint32_t ty = get_tile_size(cb_out);
    const auto y_acc = TensorAccessor(y_args, y_addr, ty);
    for (uint32_t m = 0; m < MT; ++m) {
        for (uint32_t s = 0; s < NS; ++s) {
            cb_wait_front(cb_out, SBW);
            const uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < SBW; ++j) {
                if (s * SBW + j < nvalid) {
                    noc_async_write_page((mt0 + m) * nt + n0 + s * SBW + j, y_acc, rd + j * ty);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, SBW);
        }
    }
}
