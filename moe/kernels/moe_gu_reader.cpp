// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of moe8's gate_up (moe_gu_compute.cpp; moe8.py): core j of bank `bank` serves the slots
// s = j, j + CPB, ... of the k chosen experts. Per slot: its expert id from ids' row 0, the
// expert's byte offset in this bank from the table, its pair count from its rank in the bank
// rotation, the routing weight as a tile (bf16, all of row 0), then the expert's block in chunks
// of KW K rows: each row's 2 * pairs tiles in one read into a row of NC pages, the pages past them
// zero (a narrower bank). x is read whole, once.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ws = 2, cb_scr = 9;
constexpr uint32_t TB = 576;

inline void zero(uint32_t dst, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        const uint32_t n = bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE;
        noc_async_read(z, dst + off, n);
    }
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KW = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);  // chosen experts
    constexpr uint32_t CPB = get_compile_time_arg_val(4);
    constexpr uint32_t BANKS = get_compile_time_arg_val(5);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(6);  // gate/up tile pairs per expert
    constexpr bool IDS32 = get_compile_time_arg_val(7) != 0;
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto i_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    static_assert(KT % KW == 0 && K <= 16, "moe8: chunks; at most 16 experts");

    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gu_addr = get_arg_val<uint32_t>(i++);
    const uint32_t t_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    const auto i_acc = TensorAccessor(i_args, ids_addr, 64);
    const auto w_acc = TensorAccessor(w_args, w_addr, 64);
    const auto t_acc = TensorAccessor(t_args, t_addr, 64);

    // ids and routing weights: the first 32 bytes of each tensor's first page (row 0); the table rows
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, 64);
    noc_async_read(w_acc.get_noc_addr(0), scr + 64, 64);
    // x, whole
    cb_reserve_back(cb_x, KT);
    const uint32_t lx = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < KT; ++k) {
        noc_async_read_page(k, x_acc, lx + k * 2048);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, KT);

    volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    volatile tt_l1_ptr uint32_t* ids32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 64);
    for (uint32_t s = j; s < K; s += CPB) {
        const uint32_t e = IDS32 ? ids32[s] : ids16[s];
        // the expert's table row (16 words: its byte offset in each bank)
        noc_async_read_page(e, t_acc, scr + 128);
        noc_async_read_barrier();
        const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128)[bank];
        const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
        const uint32_t cnt = 2 * (PAIRS / BANKS + (rank < PAIRS % BANKS ? 1 : 0));
        // the routing weight as a scalar tile
        cb_reserve_back(cb_ws, 1);
        {
            const uint32_t t = get_write_ptr(cb_ws);
            zero(t, 2048);
            noc_async_read_barrier();
            // in every column of row 0 (faces 0 and 1): the compute multiplies the products by
            // it in dest (SFPU, row 0), so no scalar broadcast and no fp32 round trip through L1
            volatile tt_l1_ptr uint16_t* r0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t);
            const uint16_t wv = wts[s];
            for (uint32_t c = 0; c < 16; ++c) {
                r0[c] = wv;
                r0[256 + c] = wv;
            }
        }
        cb_push_back(cb_ws, 1);
        // the expert's block in this bank, KW K rows at a time
        for (uint32_t c = 0; c < KT / KW; ++c) {
            cb_reserve_back(cb_w, KW * NC);
            const uint32_t dst = get_write_ptr(cb_w);
            for (uint32_t k = 0; k < KW; ++k) {
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, gu_addr + off + (c * KW + k) * cnt * TB),
                    dst + k * NC * TB,
                    cnt * TB,
                    noc_index,
                    vc);
                if (cnt < NC) {
                    zero(dst + (k * NC + cnt) * TB, (NC - cnt) * TB);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, KW * NC);
        }
    }
}
