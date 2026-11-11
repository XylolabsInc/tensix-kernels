// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of moe8's gate_up with two loaders (moe_gu_reader.cpp's work; moe8.gate_up(loaders=2)): core j of bank
// `bank` serves the slots s = j, j + CPB, ... of the k chosen experts.
//   - ids, routing weights and x are read together (one round trip); then the table rows of all of
//     this core's slots (one more), not one barrier per slot;
//   - the routing weight's tile pages are zeroed on their first use only (row 0 is rewritten per
//     slot, the rest stays zero);
//   - SPLIT: chunk g of the core's running chunk index (over its slots) is loaded here when it is the reader's
//     (split.h); the writer loads the others into c_4 (moe_gu2_writer.cpp), a second request stream.
//   - G column groups (moe8.gu_groups: more of an expert's tiles in a bank than dest holds): each slot is G
//     units, group gi the NC tiles from gi * NC of the expert's block in this bank (its groups lie one after the
//     other, each KT x its tiles); a group past the expert's tiles here is zero-filled.
//   - BIAS (a gate/up bias, moe8.pack(gu_bias=)): per unit the group's bias tiles into c_5, row 0 (bf16, 64
//     bytes a tile, right after the expert's tiles in this bank), for the compute to put in dest first.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "split.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ws = 2, cb_b = 5, cb_bst = 6, cb_scr = 9;
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
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(8);  // split.h: 0 none, 1 every other chunk, else a pattern
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr uint32_t G = get_compile_time_arg_val(9);  // column groups per slot
    constexpr bool BIAS = get_compile_time_arg_val(10) != 0;
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto i_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    static_assert(KT % KW == 0 && K <= 16, "moe8: chunks; at most 16 experts");
    static_assert(G >= 1 && NC % 2 == 0, "moe8: column groups of gate/up pairs");
    constexpr uint32_t CHUNKS = KT / KW;

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

    // ids and routing weights (row 0 of each tensor's first page) and x, whole, in one round trip
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, 64);
    noc_async_read(w_acc.get_noc_addr(0), scr + 64, 64);
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
    // the table rows of this core's slots (16 words each: the expert's byte offset in each bank)
    const uint32_t rows = scr + 128;
    for (uint32_t s = j, u = 0; s < K; s += CPB, ++u) {
        noc_async_read_page(IDS32 ? ids32[s] : ids16[s], t_acc, rows + u * 64);
    }
    noc_async_read_barrier();

    uint32_t stg = 0;  // BIAS: the staging of a group's bias row
    if constexpr (BIAS) {
        cb_reserve_back(cb_bst, 1);
        stg = get_write_ptr(cb_bst);
    }
    uint32_t g = 0;   // the core's running chunk index (SPLIT: the writer's by split.h)
    uint32_t un = 0;  // the core's running unit index (slot x group)
    for (uint32_t s = j, u = 0; s < K; s += CPB, ++u) {
        const uint32_t e = IDS32 ? ids32[s] : ids16[s];
        const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + u * 64)[bank];
        const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
        const uint32_t cnt = 2 * (PAIRS / BANKS + (rank < PAIRS % BANKS ? 1 : 0));
        for (uint32_t gi = 0; gi < G; ++gi, ++un) {
            // this group's tiles of the expert here (G 1: all cnt) and its block (G 1: the expert's)
            const uint32_t gcnt = cnt > gi * NC ? (cnt - gi * NC < NC ? cnt - gi * NC : NC) : 0;
            const uint32_t goff = off + gi * KT * NC * TB;
            // the routing weight in every column of row 0 (faces 0 and 1) of a scalar tile; its two
            // pages are zeroed on their first use
            cb_reserve_back(cb_ws, 1);
            {
                const uint32_t t = get_write_ptr(cb_ws);
                if (un < 2) {
                    zero(t, 2048);
                    noc_async_read_barrier();
                }
                volatile tt_l1_ptr uint16_t* r0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t);
                const uint16_t wv = wts[s];
                for (uint32_t c = 0; c < 16; ++c) {
                    r0[c] = wv;
                    r0[256 + c] = wv;
                }
            }
            cb_push_back(cb_ws, 1);
            if constexpr (BIAS) {
                // the group's bias: the 64 bytes a tile after the expert's tiles here into the staging, then each tile's
                // 32 values into row 0 of its tile (faces 0 and 1); the pages' other rows are zeroed on first use
                if (gcnt > 0) {
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(bank, gu_addr + off + KT * cnt * TB + gi * NC * 64),
                        stg,
                        gcnt * 64,
                        noc_index,
                        vc);
                }
                cb_reserve_back(cb_b, NC);
                const uint32_t bt = get_write_ptr(cb_b);
                if (un < 2) {
                    zero(bt, NC * 2048);
                }
                noc_async_read_barrier();
                for (uint32_t t = 0; t < NC; ++t) {
                    if (t < gcnt) {
                        noc_async_read(get_noc_addr(stg + t * 64), bt + t * 2048, 32);
                        noc_async_read(get_noc_addr(stg + t * 64 + 32), bt + t * 2048 + 512, 32);
                    } else {
                        zero(bt + t * 2048, 32);
                        zero(bt + t * 2048 + 512, 32);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_b, NC);
            }
            // the group's block in this bank, KW K rows at a time
            for (uint32_t c = 0; c < CHUNKS; ++c, ++g) {
                if (SPLIT && tk::split_writer<SPLIT_ARG>(g)) {
                    continue;
                }
                cb_reserve_back(cb_w, KW * NC);
                const uint32_t dst = get_write_ptr(cb_w);
                const uint32_t src = gu_addr + goff + c * KW * gcnt * TB;
                for (uint32_t k = 0; k < KW; ++k) {
                    if (gcnt > 0) {
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(bank, src + k * gcnt * TB),
                            dst + k * NC * TB,
                            gcnt * TB,
                            noc_index,
                            vc);
                    }
                    if (gcnt < NC) {
                        zero(dst + (k * NC + gcnt) * TB, (NC - gcnt) * TB);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_w, KW * NC);
            }
        }
    }
}
