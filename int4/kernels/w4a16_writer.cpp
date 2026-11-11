// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of the W4A16 group-scale matmul (w4a16_compute.cpp). With SPLIT it first loads the odd
// weight chunks into the second weight CB (c_4; w4a16_wload.h), a second request stream next to
// the reader's, in the compute's order (x block, column subblock, pass, chunk). Then the core's
// output tiles (one token: tile row 0) into y [1 x nt tiles]; of the core's NC columns only the
// first nvalid are written.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "w4a16_wload.h"

constexpr uint32_t cb_w2 = 4, cb_stage2 = 10, cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t SBW = get_compile_time_arg_val(1);
    constexpr uint32_t KT = get_compile_time_arg_val(2);
    constexpr uint32_t KB = get_compile_time_arg_val(3);
    constexpr uint32_t KW = get_compile_time_arg_val(4);
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(5);
    constexpr uint32_t EXP = get_compile_time_arg_val(6);
    constexpr bool SPLIT = get_compile_time_arg_val(7) != 0;
    constexpr auto y_args = TensorAccessorArgs<8>();
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NKB = KT / KB;
    constexpr uint32_t NP = KB / 32;

    uint32_t i = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(i++);
    const uint32_t nt = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t pitch = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    if constexpr (SPLIT) {
        w4::init_exponents(cb_w2, W_PAGES, EXP);
        cb_reserve_back(cb_stage2, 1);  // scratch, never pushed
        const uint32_t stage = get_write_ptr(cb_stage2);
        uint32_t g = 0;
        for (uint32_t b = 0; b < NKB; ++b) {
            for (uint32_t s = 0; s < NS; ++s) {
                const uint32_t j0 = s * SBW;
                const uint32_t jn = nvalid > j0 ? (nvalid - j0 < SBW ? nvalid - j0 : SBW) : 0;
                for (uint32_t p = 0; p < NP; ++p) {
                    for (uint32_t c = 0; c < 32 / KW; ++c, ++g) {
                        if (g & 1) {
                            w4::chunk<KW, SBW>(
                                cb_w2, stage, bank, w_addr, pitch, col0 + j0, jn, b * KB + p * 32 + c * KW, vc);
                        }
                    }
                }
            }
        }
    }

    const uint32_t ty = get_tile_size(cb_out);
    const auto y_acc = TensorAccessor(y_args, y_addr, ty);
    for (uint32_t s = 0; s < NS; ++s) {
        cb_wait_front(cb_out, SBW);
        const uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < SBW; ++j) {
            if (s * SBW + j < nvalid) {
                noc_async_write_page(n0 + s * SBW + j, y_acc, rd + j * ty);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, SBW);
    }
}
