// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// head_pick's records-only pick (head_pick.py: pick_records), one core for every row. Per row, the top-k result's best
// candidate by the device's own logit, with no rescore, then its token from a map. It serves a draft head: a sub-head of
// the most frequent tokens. A draft is a guess a verify step checks, so the device's argmax is enough.
//   1. The records (linear.matmul's topk: NREC row-major pages; in each page the keys of every row, ROWS x K words, then
//      their ids, ROWS x K words), read whole.
//   2. Per row r, the largest key over the records in order, record by record and slot by slot. An equal key keeps the
//      earlier one: torch.argmax over linear.records_host's values, as lmhead.DeviceHead's collect does without a rescore.
//      A key is the bf16 logit's bits in an order-preserving 16-bit code, 0 for an empty slot. A NaN counts as the
//      largest whatever its sign, and -0 equals +0 (torch's compare). An id at or past VOCAB is an empty slot.
//   3. With MAPPED, the best id's token is word id % 32 of the map's page id / 32 (a uint32 row-major map, 32 ids a
//      128-byte page). Without MAPPED, the token is the id itself.
//   4. Row r's output page (128 bytes, uint32): [the token, the best id, the valid slots' count, 0 ...]. A row with no
//      valid slot gets [0, 0xFFFFFFFF, 0].
//   5. IDS (ids_out): row 0's token also into page 0 of a one-element uint32 tensor (an embedding's index for a draft's
//      next step), as 32 bytes [token, 0, ...] within the page's DRAM-aligned slot.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_stg = 4, cb_map = 5, cb_scr = 9;

// a record key's rank: the key, a NaN's above every key (0x10000), -0's (0x7FFF) equal to +0's (0x8000); key != 0
inline uint32_t rank16(uint32_t kk) {
    const uint32_t mag = (kk >= 0x8000u ? kk : ~kk) & 0x7FFFu;  // the bf16's bits without the sign
    if ((mag & 0x7F80u) == 0x7F80u && (mag & 0x007Fu) != 0) {
        return 0x10000u;
    }
    return kk == 0x7FFFu ? 0x8000u : kk;
}

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);       // candidates a record holds for a row
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);    // rows of the top-k result (all picked here)
    constexpr uint32_t NREC = get_compile_time_arg_val(2);    // records (pages) of the top-k result
    constexpr uint32_t VOCAB = get_compile_time_arg_val(3);   // an id at or past it is an empty slot
    constexpr bool MAPPED = get_compile_time_arg_val(4) != 0;  // the token from the map (else the id)
    constexpr bool IDS = get_compile_time_arg_val(5) != 0;     // row 0's token into ids_out too
    constexpr auto r_args = TensorAccessorArgs<6>();
    constexpr auto m_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    static_assert(K >= 1 && ROWS >= 1 && ROWS <= 32 && NREC >= 1, "head_pick_records");
    constexpr uint32_t PAGE = 2 * ROWS * K * 4;  // a record's bytes (the result tensor's page)
    // the records' stride in L1: a page rounded up to 64 bytes, so each record lands at its DRAM page's offset
    // modulo 64 (a DRAM page starts 64-byte aligned where the DRAM alignment is 64, as on Blackhole, and a NoC
    // read needs its source and destination alike modulo 64)
    constexpr uint32_t STRIDE = (PAGE + 63) / 64 * 64;
    // DRAM reads land in L1 at the same offset modulo 64: the records a STRIDE apart from c_9's base (64-byte aligned,
    // as in head_pick_reader.cpp); the map's and the output's pages are 128 bytes at their CBs' bases.
    static_assert((PAGE & 31) == 0, "head_pick_records: 32-byte aligned records");

    uint32_t i = 0;
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t map_addr = get_arg_val<uint32_t>(i++);
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t d_addr = get_arg_val<uint32_t>(i++);  // ids_out (IDS)
    const auto ra = TensorAccessor(r_args, rec_addr, PAGE);
    const auto da = TensorAccessor(d_args, d_addr, 32);
    const auto ma = TensorAccessor(m_args, map_addr, 128);
    const auto oa = TensorAccessor(o_args, o_addr, 128);

    // 1. the records, whole
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t p = 0; p < NREC; ++p) {
        noc_async_read_page(p, ra, scr + p * STRIDE);
    }
    noc_async_read_barrier();

    cb_reserve_back(cb_map, 1);
    const uint32_t mp = get_write_ptr(cb_map);
    cb_reserve_back(cb_stg, ROWS);  // the output pages' staging, 128 bytes a row
    const uint32_t stg = get_write_ptr(cb_stg);
    for (uint32_t r = 0; r < ROWS; ++r) {
        // 2. row r's best: the first slot of the largest rank
        uint32_t best = 0xFFFFFFFFu, best_rank = 0, n = 0;
        for (uint32_t p = 0; p < NREC; ++p) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + p * STRIDE);
            for (uint32_t s = 0; s < K; ++s) {
                const uint32_t kk = w[r * K + s] & 0xFFFFu;
                const uint32_t id = w[ROWS * K + r * K + s];
                if (kk == 0 || id >= VOCAB) {
                    continue;  // an empty slot (or an id past the table)
                }
                const uint32_t rk = rank16(kk);
                if (n++ == 0 || rk > best_rank) {
                    best_rank = rk;
                    best = id;
                }
            }
        }
        // 3. the token
        uint32_t tok = 0;
        if (n > 0) {
            if constexpr (MAPPED) {
                noc_async_read_page(best >> 5, ma, mp);
                noc_async_read_barrier();
                tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp)[best & 31];
            } else {
                tok = best;
            }
        }
        // 4. row r's page
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg + r * 128);
        for (uint32_t w = 0; w < 32; ++w) {
            o[w] = 0;
        }
        o[0] = tok;
        o[1] = best;
        o[2] = n;
        noc_async_write(stg + r * 128, oa.get_noc_addr(r), 128);
    }
    noc_async_write_barrier();
    if constexpr (IDS) {
        // row 0's page staged again as [token, 0 x 7] (its write is done), then 32 bytes to ids_out
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg);
        o[1] = 0;
        o[2] = 0;
        noc_async_write(stg, da.get_noc_addr(0), 32);
        noc_async_write_barrier();
    }
}
