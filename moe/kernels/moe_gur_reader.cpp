// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of moe8's gate_up_rows (moe_gu_compute.cpp with DYN; moe8.py): R rows of x, each with its
// own k experts. The work is by expert, not by slot: the union of the R x k ids (moe8_rows.h),
// union index i = j, j + CPB, ... on core j of bank `bank`. Per unit: the expert's byte offset in
// this bank from the table, its pair count from its rank in the bank rotation, the coefficient
// tile (row r: row r's routing weight for this expert, 0 where row r did not choose it, in all 32
// columns), then the expert's block in chunks of KW K rows as moe_gu_reader.cpp reads it. The
// unit count goes to the compute first (cb_meta, word 0). x is read whole, once. With G column groups
// (moe8.gu_groups) each union index is G units, group gi the NC tiles from gi * NC of the expert's block here
// (zero-filled past its tiles); the coefficient tile is made once per union index (RISC stores), copied into the
// other page of c_2 for its second group (a local NoC read), and pushed as it stands for the rest, as both pages then
// hold it. BIAS (a gate/up bias, moe8.pack(gu_bias=)): per unit the group's bias tiles into c_5, every row (bf16, 64
// bytes a tile right after the expert's tiles in this bank), for the compute to put in dest first.
// SPLIT (the second loader, moe8.gate_up_rows(loaders=2)): chunk g of the core's running chunk index (over its units)
// is loaded here when it is the reader's (split.h); the writer loads the others into c_4 (moe_gur2_writer.cpp), a
// second request stream, as moe_gu2_*.cpp do for decode.
//
// Scratch (cb_scr): the ids tile, the weights tile, the table row, the union's bits, its list, each
// expert's union index, and C [u][32]: the coefficients as bf16 bits.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe8_rows.h"
#include "split.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_ws = 2, cb_meta = 3, cb_b = 5, cb_bst = 6, cb_scr = 9;
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
    constexpr uint32_t K = get_compile_time_arg_val(3);  // experts per row
    constexpr uint32_t CPB = get_compile_time_arg_val(4);
    constexpr uint32_t BANKS = get_compile_time_arg_val(5);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(6);  // gate/up tile pairs per expert
    constexpr bool IDS32 = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t ROWS = get_compile_time_arg_val(8);
    constexpr uint32_t E = get_compile_time_arg_val(9);
    constexpr uint32_t G = get_compile_time_arg_val(10);  // column groups per union index
    constexpr bool BIAS = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(12);  // split.h: 0 none, 1 every other chunk, else a pattern
    constexpr bool SPLIT = SPLIT_ARG != 0;  // the writer loads some chunks (SPLIT_ARG's pattern)
    constexpr auto x_args = TensorAccessorArgs<13>();
    constexpr auto i_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    static_assert(KT % KW == 0 && K <= 16 && ROWS >= 1 && ROWS <= 32 && E <= 1024, "moe8 rows");
    constexpr uint32_t OFF_W = 4096, OFF_T = 6144, OFF_BITS = 6208, OFF_LIST = OFF_BITS + 128;
    constexpr uint32_t OFF_POS = OFF_LIST + 2 * E;
    constexpr uint32_t OFF_C = (OFF_POS + 2 * E + 15) / 16 * 16;

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
    const auto i_acc = TensorAccessor(i_args, ids_addr, IDS32 ? 4096 : 2048);
    const auto w_acc = TensorAccessor(w_args, w_addr, 2048);
    const auto t_acc = TensorAccessor(t_args, t_addr, 64);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, IDS32 ? 4096 : 2048);
    noc_async_read(w_acc.get_noc_addr(0), scr + OFF_W, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_BITS);
    volatile tt_l1_ptr uint16_t* list = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_LIST);
    volatile tt_l1_ptr uint16_t* pos = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_POS);
    volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_C);
    volatile tt_l1_ptr uint16_t* ch = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_C);
    volatile tt_l1_ptr uint16_t* wts = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_W);
    const uint32_t u = moe8r::build_union<IDS32, K, E>(scr, ROWS, bits, list);
    const uint32_t units = moe8r::units_of(u, j, CPB) * G;

    // the unit count, for the compute
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = units;
    cb_push_back(cb_meta, 1);

    // x, whole (in flight while the coefficients are made)
    cb_reserve_back(cb_x, KT);
    const uint32_t lx = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < KT; ++k) {
        noc_async_read_page(k, x_acc, lx + k * 2048);
    }
    // C [u][32]: row r's weight for union index i (bf16 bits), 0 where row r did not choose it
    for (uint32_t q = 0; q < u; ++q) {
        pos[list[q]] = static_cast<uint16_t>(q);
    }
    for (uint32_t q = 0; q < u * 16; ++q) {
        cw[q] = 0;
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t e = moe8r::id_at<IDS32>(scr, r, s);
            if (e < E) {
                ch[pos[e] * 32 + r] = wts[moe8r::tile_at(r, s)];
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, KT);

    uint32_t stg = 0;  // BIAS: the staging of a group's bias row
    if constexpr (BIAS) {
        cb_reserve_back(cb_bst, 1);
        stg = get_write_ptr(cb_bst);
    }
    [[maybe_unused]] uint32_t g = 0;  // the core's running chunk index (SPLIT: the writer's by split.h)
    for (uint32_t q = j; q < u; q += CPB) {
        const uint32_t e = list[q];
        // the expert's table row (16 words: its byte offset in each bank)
        noc_async_read_page(e, t_acc, scr + OFF_T);
        noc_async_read_barrier();
        const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_T)[bank];
        const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
        const uint32_t cnt = 2 * (PAIRS / BANKS + (rank < PAIRS % BANKS ? 1 : 0));
        uint32_t made = 0;  // the c_2 page holding this union index's coefficient tile
        for (uint32_t gi = 0; gi < G; ++gi) {
            const uint32_t gcnt = cnt > gi * NC ? (cnt - gi * NC < NC ? cnt - gi * NC : NC) : 0;
            const uint32_t goff = off + gi * KT * NC * TB;
            // the coefficient tile: row r all c[r] (faces 0 and 1 for rows 0-15, 2 and 3 for 16-31)
            cb_reserve_back(cb_ws, 1);
            if (gi == 0) {
                made = get_write_ptr(cb_ws);
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(made);
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t v = ch[q * 32 + r];
                    const uint32_t vv = v | (v << 16);
                    const uint32_t b = moe8r::tile_at(r, 0) / 2;  // in words
                    for (uint32_t c = 0; c < 8; ++c) {
                        t[b + c] = vv;
                        t[b + 128 + c] = vv;
                    }
                }
            } else if (gi == 1) {
                noc_async_read(get_noc_addr(made), get_write_ptr(cb_ws), 2048);
                noc_async_read_barrier();
            }
            cb_push_back(cb_ws, 1);
            if constexpr (BIAS) {
                // the group's bias: the 64 bytes a tile after the expert's tiles here into the staging, each tile's
                // 32 values into its row 0 (faces 0 and 1), then into every row: doubled within faces 0 and 1, which
                // then fill faces 2 and 3 (local NoC copies); a tile past the expert's here is zero
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
                noc_async_read_barrier();
                for (uint32_t t = 0; t < NC; ++t) {
                    if (t < gcnt) {
                        noc_async_read(get_noc_addr(stg + t * 64), bt + t * 2048, 32);
                        noc_async_read(get_noc_addr(stg + t * 64 + 32), bt + t * 2048 + 512, 32);
                    } else {
                        zero(bt + t * 2048, 2048);
                    }
                }
                noc_async_read_barrier();
                for (uint32_t w = 32; w < 512; w *= 2) {
                    for (uint32_t t = 0; t < gcnt && t < NC; ++t) {
                        noc_async_read(get_noc_addr(bt + t * 2048), bt + t * 2048 + w, w);
                        noc_async_read(get_noc_addr(bt + t * 2048 + 512), bt + t * 2048 + 512 + w, w);
                    }
                    noc_async_read_barrier();
                }
                for (uint32_t t = 0; t < gcnt && t < NC; ++t) {
                    noc_async_read(get_noc_addr(bt + t * 2048), bt + t * 2048 + 1024, 1024);
                }
                noc_async_read_barrier();
                cb_push_back(cb_b, NC);
            }
            // the group's block in this bank, KW K rows at a time
            for (uint32_t c = 0; c < KT / KW; ++c, ++g) {
                if constexpr (SPLIT) {
                    if (tk::split_writer<SPLIT_ARG>(g)) {
                        continue;
                    }
                }
                cb_reserve_back(cb_w, KW * NC);
                const uint32_t dst = get_write_ptr(cb_w);
                for (uint32_t k = 0; k < KW; ++k) {
                    if (gcnt > 0) {
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(bank, gu_addr + goff + (c * KW + k) * gcnt * TB),
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
