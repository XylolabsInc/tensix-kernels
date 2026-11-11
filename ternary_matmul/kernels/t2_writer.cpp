// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of the ternary matmul (t2_compute.cpp). With SPLIT it first loads the odd weight chunks
// into the second weight CB (c_4; t2_wload.h), a second request stream next to the reader's. Then
// the core's output tiles in groups of SBW into y [1 x nt tiles] (interleaved, DRAM or L1, page
// n0 + column); of the core's NC columns only the first nvalid are written (the rest are padding).
// TOPK > 0 (an LM head, as the dense reader's dense_writer.cpp): a running top-TOPK of the token's row over the
// core's output columns, written as one record (page `slot`) of the result tensor r; WRITE_Y = 0
// skips y.
// Output column tile c (n0 + the core's column) goes to page c + col0, plus shift from tile split
// on: a call writes into a wider preallocated y at a column offset, its columns possibly in two
// runs (a fused projection whose output leaves a gap for another op's columns).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "t2_wload.h"

constexpr uint32_t cb_w2 = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KW = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t SBW = get_compile_time_arg_val(3);
    constexpr bool SPLIT = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t TB = get_compile_time_arg_val(5);
    constexpr uint32_t TOPK = get_compile_time_arg_val(6);
    constexpr bool WRITE_Y = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t DB = get_compile_time_arg_val(8);  // weight tile bytes in DRAM
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(9);
    constexpr uint32_t EXP = get_compile_time_arg_val(10);
    constexpr auto y_args = TensorAccessorArgs<11>();
    constexpr auto r_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    static_assert(TOPK <= 64, "t2: the top-k state lives on the writer's stack");
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NCH = KT / KW;

    uint32_t i = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t core_off = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t split = get_arg_val<uint32_t>(i++);
    const uint32_t shift = get_arg_val<uint32_t>(i++);
    // TOPK: the result tensor, this core's record, the global index of output element 0 of this
    // call (a vocabulary block's offset) and the number of valid output elements of the call
    const uint32_t r_addr = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    const uint32_t slot = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    const uint32_t base = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    const uint32_t n_elems = TOPK ? get_arg_val<uint32_t>(i++) : 0;

    // keys: a bf16 as a monotonic uint16 (sign-magnitude to offset order), 0 below everything
    constexpr uint32_t KR = TOPK ? TOPK : 1;
    uint32_t key[KR], idx[KR], low = 0;  // low: the slot of the smallest key
    for (uint32_t q = 0; q < KR; ++q) {
        key[q] = 0;
        idx[q] = 0xFFFFFFFFu;
    }

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
    for (uint32_t s = 0; s < NS; ++s) {
        cb_wait_front(cb_out, SBW);
        const uint32_t rd = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < SBW; ++j) {
            const uint32_t col = s * SBW + j;
            if (col >= nvalid) {
                continue;
            }
            if constexpr (WRITE_Y) {
                const uint32_t c = n0 + col;
                noc_async_write_page(c + col0 + (c >= split ? shift : 0), y_acc, rd + j * ty);
            }
            if constexpr (TOPK > 0) {
                // row 0 of a bf16 tile in face order: element c of face c / 16 at (c / 16) * 256 + c % 16
                volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rd + j * ty);
                const uint32_t g0 = (n0 + col) * 32;
                for (uint32_t c = 0; c < 32 && g0 + c < n_elems; ++c) {
                    const uint32_t b = e[(c >> 4) * 256 + (c & 15)];
                    const uint32_t k = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
                    if (k > key[low]) {
                        key[low] = k;
                        idx[low] = base + g0 + c;
                        uint32_t lo = 0;
                        for (uint32_t q = 1; q < TOPK; ++q) {
                            if (key[q] < key[lo]) {
                                lo = q;
                            }
                        }
                        low = lo;
                    }
                }
            }
        }
        if constexpr (WRITE_Y) {
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, SBW);
    }
    if constexpr (TOPK > 0) {
        // the record: TOPK keys, then TOPK indices (uint32 each), staged in c_13
        constexpr uint32_t RBYTES = 2 * TOPK * 4;
        constexpr uint32_t cb_rec = 13;
        const auto r_acc = TensorAccessor(r_args, r_addr, RBYTES);
        cb_reserve_back(cb_rec, 1);
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_rec));
        for (uint32_t q = 0; q < TOPK; ++q) {
            rec[q] = key[q];
            rec[TOPK + q] = idx[q];
        }
        noc_async_write_page(slot, r_acc, get_write_ptr(cb_rec));
        noc_async_write_barrier();
    }
}
