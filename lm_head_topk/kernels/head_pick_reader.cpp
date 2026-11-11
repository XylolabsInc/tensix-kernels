// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of head_pick (head_pick.py: the top-k LM head's exact rescore and pick on the device). Row r of the step is
// this core's.
//   1. The top-k result's records (linear.matmul's topk: one row-major page a record, NREC of them; in each page the keys
//      of every row, ROWS x K words, then their ids, ROWS x K words), read whole, then row r's K keys and ids of each
//      record scanned for the best TOP by key. A key is the bf16 logit's bits in an order-preserving 16-bit code
//      (linear.records_host), 0 for an empty slot, so an unsigned compare orders them; an equal key keeps the earlier.
//   2. The TOP candidates (their ids, their keys, and how many are valid) to the writer (c_3).
//   3. Their bf16 head rows, gathered from the row-major head (one DRAM page a vocabulary row) in DT / CW blocks of CW
//      tiles' columns: candidate i's columns of block b into row i of the block's 32 rows (row-major, CW * 64 bytes a
//      row, the compute tilizes them); rows TOP .. 31, and the rows of empty candidates, zero. c_1 holds two blocks; each
//      page is zeroed on its first use, after which only rows of valid candidates are written (the zeros stay).
//   4. x's DT tiles (the matmul's in0), streamed after the blocks.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_rm = 1, cb_ids = 3, cb_scr = 9;

inline void zero(uint32_t dst, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        const uint32_t n = bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE;
        noc_async_read(z, dst + off, n);
    }
}

void kernel_main() {
    constexpr uint32_t DT = get_compile_time_arg_val(0);    // tiles of a head row (dim / 32)
    constexpr uint32_t CW = get_compile_time_arg_val(1);    // tiles a gathered block
    constexpr uint32_t TOP = get_compile_time_arg_val(2);   // candidates rescored (<= 32)
    constexpr uint32_t K = get_compile_time_arg_val(3);     // candidates a record holds for a row
    constexpr uint32_t ROWS = get_compile_time_arg_val(4);  // rows of the top-k result
    constexpr uint32_t NREC = get_compile_time_arg_val(5);  // records (pages) of the top-k result
    constexpr uint32_t VOCAB = get_compile_time_arg_val(6);  // the head table's rows: an id at or past it is an empty slot
    constexpr auto r_args = TensorAccessorArgs<7>();
    constexpr auto h_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    static_assert(DT % CW == 0 && TOP >= 1 && TOP <= 32 && K >= 1, "head_pick");
    constexpr uint32_t PAGE = 2 * ROWS * K * 4;  // a record's bytes (the result tensor's page)
    // the records' stride in L1: a page rounded up to 64 bytes, so each record lands at its DRAM page's offset
    // modulo 64 (a DRAM page starts 64-byte aligned where the DRAM alignment is 64, as on Blackhole, and a NoC
    // read needs its source and destination alike modulo 64)
    constexpr uint32_t STRIDE = (PAGE + 63) / 64 * 64;
    constexpr uint32_t BW = CW * 64;              // a block row's bytes (CW tiles' 32 bf16 columns)
    // DRAM reads land in L1 at the same offset modulo 64: the records a STRIDE apart from c_9's base, the blocks' rows a BW
    // apart from c_1's. Both rest on the CB bases being 64-byte aligned (every CB size here is a multiple of 64 and the
    // first CB's base is the allocator's aligned L1 base: assumed, as in the other kernels here).
    static_assert((PAGE & 31) == 0 && (BW & 31) == 0, "head_pick: 32-byte aligned records and blocks");

    uint32_t i = 0;
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t head_addr = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t r = get_arg_val<uint32_t>(i++);  // this core's row
    const uint32_t row_bytes = get_arg_val<uint32_t>(i++);  // a head row's bytes (the head's page)

    const auto ra = TensorAccessor(r_args, rec_addr, PAGE);
    const auto ha = TensorAccessor(h_args, head_addr, row_bytes);
    const auto xa = TensorAccessor(x_args, x_addr, get_tile_size(cb_x));

    // 1. the records, whole, then the best TOP of row r
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t p = 0; p < NREC; ++p) {
        noc_async_read_page(p, ra, scr + p * STRIDE);
    }
    noc_async_read_barrier();
    uint32_t key[TOP], cid[TOP];
    uint32_t n = 0;  // candidates held, in decreasing key order
    for (uint32_t p = 0; p < NREC; ++p) {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + p * STRIDE);
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t kk = w[r * K + s] & 0xFFFFu;
            const uint32_t id = w[ROWS * K + r * K + s];
            if (kk == 0 || id >= VOCAB) {
                continue;  // an empty slot (or an id past the table: never read)
            }
            if (n == TOP && kk <= key[TOP - 1]) {
                continue;
            }
            uint32_t at = n < TOP ? n++ : TOP - 1;  // the slot it takes, then moved up past the smaller keys
            while (at > 0 && key[at - 1] < kk) {
                key[at] = key[at - 1];
                cid[at] = cid[at - 1];
                --at;
            }
            key[at] = kk;
            cid[at] = id;
        }
    }

    // 2. the candidates to the writer: [valid count, keys[TOP], ids[TOP]]
    cb_reserve_back(cb_ids, 1);
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ids));
        o[0] = n;
        for (uint32_t c = 0; c < TOP; ++c) {
            o[1 + c] = c < n ? key[c] : 0;
            o[1 + TOP + c] = c < n ? cid[c] : 0xFFFFFFFFu;
        }
    }
    cb_push_back(cb_ids, 1);

    // 3. the candidates' rows, block by block
    for (uint32_t b = 0; b < DT / CW; ++b) {
        cb_reserve_back(cb_rm, CW);
        const uint32_t dst = get_write_ptr(cb_rm);
        if (b < 2) {
            zero(dst, 32 * BW);  // this page's first use: every row zero (rows TOP .. 31 and empty candidates stay so)
            noc_async_read_barrier();
        }
        for (uint32_t c = 0; c < n; ++c) {
            noc_async_read(ha.get_noc_addr(cid[c], b * BW), dst + c * BW, BW);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm, CW);
    }

    // 4. x's tiles, for the matmul
    for (uint32_t t = 0; t < DT; ++t) {
        cb_reserve_back(cb_x, 1);
        noc_async_read_page(t, xa, get_write_ptr(cb_x));
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
    }
}
