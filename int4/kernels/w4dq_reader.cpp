// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of the W4 dequantize (w4dq_compute.cpp): the core's int4 weight rows (w4a16_wload.h:
// one large read per K row into staging, then the 512-byte tiles into BFP4_b pages under the
// constant exponent) and, once per pass of 32 K tiles, the pass's scale tiles of the core's
// columns (scale tensor [K/32, N] bf16 TILE: tile (pass, column) holds the scales of K tiles
// 32 pass .. 32 pass + 31 in its rows).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "w4a16_wload.h"

constexpr uint32_t cb_w = 1, cb_s = 2, cb_stage = 8;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);  // columns (tiles) per K row chunk
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(2);
    constexpr uint32_t EXP = get_compile_time_arg_val(3);
    constexpr auto s_args = TensorAccessorArgs<4>();

    uint32_t i = 0;
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t pitch = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t nt = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    const uint32_t ts = get_tile_size(cb_s);
    const auto s_acc = TensorAccessor(s_args, s_addr, ts);
    w4::init_exponents(cb_w, W_PAGES, EXP);
    cb_reserve_back(cb_stage, 1);  // scratch, never pushed
    const uint32_t stage = get_write_ptr(cb_stage);

    for (uint32_t kt = 0; kt < KT; ++kt) {
        if (kt % 32 == 0) {
            cb_reserve_back(cb_s, NC);
            const uint32_t ls = get_write_ptr(cb_s);
            for (uint32_t j = 0; j < nvalid; ++j) {
                noc_async_read_page((kt / 32) * nt + n0 + j, s_acc, ls + j * ts);
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, NC);
        }
        w4::chunk<1, NC>(cb_w, stage, bank, w_addr, pitch, col0, nvalid, kt, vc);
    }
}
