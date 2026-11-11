// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of the dense reader's matmul (dense_compute.cpp). With SPLIT it first loads its share of the weight chunks
// into the second weight CB (c_4; dense_wload.h), a second request stream next to the reader's. Then the core's output
// tiles, row by row in groups of SBW, into y [MT x nt tiles] (interleaved, DRAM or L1, page id m * nt + n); of the
// core's NC columns only the first nvalid are written (the rest are padding of the subblocks); the rows from tile row
// mt0 (row slices of prefill). With TOPK it keeps a running top-k of each output row over the core's columns and writes
// it as one record.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dense_wload.h"

constexpr uint32_t cb_w2 = 4, cb_out = 16;

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t SBW = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t KB = get_compile_time_arg_val(4);
    constexpr uint32_t KW = get_compile_time_arg_val(5);
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(6);
    constexpr bool W_SHARDED = get_compile_time_arg_val(7) != 0;
    // SPLIT: this RISC loads some of the weight chunks into cb_w2 (split.h's pattern: 1 the odd ones, else a period)
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(8);
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr bool ROWS = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t TB = get_compile_time_arg_val(10);  // a weight tile's bytes
    // TOPK (> 0): a running top-TOPK of each of the first TROWS rows over the core's output columns, written as one
    // record (page `slot`) of the result tensor r; WRITE_Y = 0 skips y
    constexpr uint32_t TOPK = get_compile_time_arg_val(11);
    constexpr uint32_t TROWS = get_compile_time_arg_val(12);
    constexpr bool WRITE_Y = get_compile_time_arg_val(13) != 0;
    // GLU: the compute packs act(gate) * up, one tile per pair of the core's columns (SBW / 2 per subblock); output
    // tile (n0 + col) / 2 of a y of nt = NT / 2 tiles per row
    constexpr bool GLU = get_compile_time_arg_val(14) != 0;
    // GLU2 (> 0): gate and up as two weights (the reader's GLU2), for the SPLIT loads
    constexpr uint32_t GLU2 = get_compile_time_arg_val(15);
    // HAS_RES: the residual's tiles of this core's columns (same page ids as y) into c_8 before anything else
    constexpr bool HAS_RES = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t REQ = get_compile_time_arg_val(17);  // a row's requests at most REQ tiles (0: one a row)
    constexpr auto y_args = TensorAccessorArgs<18>();
    constexpr auto w_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    // SEGS (> 1, a block list): the core's columns of SEGS blocks as segments in sequence (dense_reader.cpp): this RISC
    // loads its chunks of each in turn, and writes each segment's output columns at that block's place in y
    constexpr uint32_t SEGS = get_compile_time_arg_val(res_args.next_compile_time_args_offset());
    static_assert(TOPK == 0 || (MT == 1 && TROWS >= 1 && TROWS <= 16), "dense: top-k scans tile row 0");
    static_assert(TROWS * TOPK <= 64, "dense: the top-k state lives on the writer's stack");
    // at function scope: inside the GLU branch it would fire for every top-k build (a static_assert in a discarded
    // if-constexpr branch of a non-template function is still evaluated)
    static_assert(!GLU || TOPK == 0, "dense: GLU without top-k");
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NKB = KT / KB;
    static_assert(SEGS >= 1 && SEGS <= 4, "dense: 1 to 4 segments");
    static_assert(
        SEGS == 1 || (W_SHARDED && ROWS && NKB == 1 && TOPK == 0 && !HAS_RES),
        "dense: segments: sharded weights, x streamed, no top-k or residual");

    uint32_t i = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(i++);
    const uint32_t nt = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t pitch = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t mt0 = get_arg_val<uint32_t>(i++);  // first tile row of y (row slices of prefill)
    const uint32_t vc = get_arg_val<uint32_t>(i++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(i++);   // GLU2: the up weight (else unused)
    const uint32_t res_addr = get_arg_val<uint32_t>(i++);  // HAS_RES: the residual (else unused)
    // TOPK: the result tensor, this core's record, the global index of output element 0 of this call (a vocabulary
    // block's offset) and the number of valid output elements of the call
    const uint32_t r_addr = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    const uint32_t slot = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    const uint32_t base = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    const uint32_t n_elems = TOPK ? get_arg_val<uint32_t>(i++) : 0;
    // each segment's first output tile (its block's col0 applied), valid columns, weight, pitch, first column and up
    // weight (segment 0's above; the others' here, six each)
    uint32_t seg_n0[SEGS], seg_nvalid[SEGS], seg_w[SEGS], seg_pitch[SEGS], seg_col0[SEGS], seg_w2[SEGS];
    seg_n0[0] = n0;
    seg_nvalid[0] = nvalid;
    seg_w[0] = w_addr;
    seg_pitch[0] = pitch;
    seg_col0[0] = col0;
    seg_w2[0] = w2_addr;
    for (uint32_t s = 1; s < SEGS; ++s) {
        seg_n0[s] = get_arg_val<uint32_t>(i++);
        seg_nvalid[s] = get_arg_val<uint32_t>(i++);
        seg_w[s] = get_arg_val<uint32_t>(i++);
        seg_pitch[s] = get_arg_val<uint32_t>(i++);
        seg_col0[s] = get_arg_val<uint32_t>(i++);
        seg_w2[s] = get_arg_val<uint32_t>(i++);
    }

    // keys: a bf16 as a monotonic uint16 (sign-magnitude to offset order), 0 below everything
    constexpr uint32_t KR = TOPK ? TOPK : 1, TR = TOPK ? TROWS : 1;
    uint32_t key[TR][KR], idx[TR][KR], low[TR];  // low: the slot of the row's smallest key
    for (uint32_t t = 0; t < TR; ++t) {
        for (uint32_t q = 0; q < KR; ++q) {
            key[t][q] = 0;
            idx[t][q] = 0xFFFFFFFFu;
        }
        low[t] = 0;
    }

    if constexpr (HAS_RES) {
        constexpr uint32_t cb_res = 8;
        const uint32_t tr = get_tile_size(cb_res);
        const auto res_acc = TensorAccessor(res_args, res_addr, tr);
        cb_reserve_back(cb_res, MT * NC);
        const uint32_t rb = get_write_ptr(cb_res);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t col = 0; col < NC && col < nvalid; ++col) {
                noc_async_read_page((mt0 + m) * nt + n0 + col, res_acc, rb + (m * NC + col) * tr);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_res, MT * NC);
    }
    if constexpr (SPLIT) {
        const auto w_acc = TensorAccessor(w_args, w_addr, TB);
        tk::Loader<W_SHARDED, decltype(w_acc), TB, GLU2, REQ> ld{cb_w2, w_acc, w_addr, bank, pitch, vc};
        ld.w2_addr = w2_addr;
        uint32_t g = 0;
        for (uint32_t b = 0; b < NKB; ++b) {
            // (SEGS > 1: one x block; g runs on across the segments, as the reader's and the compute's chunk count)
            for (uint32_t sg = 0; sg < SEGS; ++sg) {
                if constexpr (SEGS > 1) {
                    ld.w_addr = seg_w[sg];
                    ld.pitch = seg_pitch[sg];
                    ld.w2_addr = seg_w2[sg];
                }
                if constexpr (ROWS) {
                    g = tk::load_block_rows<KB, KW, NC, SPLIT_ARG>(ld, seg_col0[sg], seg_nvalid[sg], b, g, true, 1);
                } else {
                    g = tk::load_block<KB, KW, SBW, NS, SPLIT_ARG>(ld, seg_col0[sg], seg_nvalid[sg], b, g, true, 1);
                }
            }
        }
    }

    const uint32_t ty = get_tile_size(cb_out);
    const auto y_acc = TensorAccessor(y_args, y_addr, ty);
    if constexpr (GLU) {
        constexpr uint32_t H = SBW / 2;
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t s = 0; s < NS * SEGS; ++s) {
                // (SEGS > 1: the subblocks of segment 0, then of segment 1, ...)
                const uint32_t sg = SEGS == 1 ? 0 : s / NS;
                cb_wait_front(cb_out, H);
                const uint32_t rd = get_read_ptr(cb_out);
                if constexpr (WRITE_Y) {
                    for (uint32_t j = 0; j < H; ++j) {
                        const uint32_t pair = (s - sg * NS) * H + j;
                        if (2 * pair < seg_nvalid[sg]) {
                            noc_async_write_page((mt0 + m) * nt + seg_n0[sg] / 2 + pair, y_acc, rd + j * ty);
                        }
                    }
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_out, H);
            }
        }
        noc_async_write_barrier();  // every output acknowledged before the kernel ends
        return;
    }
    for (uint32_t m = 0; m < MT; ++m) {
        for (uint32_t s = 0; s < NS * SEGS; ++s) {
            const uint32_t sg = SEGS == 1 ? 0 : s / NS;  // (SEGS > 1: segment 0's subblocks, then segment 1's, ...)
            cb_wait_front(cb_out, SBW);
            const uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < SBW; ++j) {
                const uint32_t col = (s - sg * NS) * SBW + j;
                if (col >= seg_nvalid[sg]) {
                    continue;
                }
                if constexpr (WRITE_Y) {
                    noc_async_write_page((mt0 + m) * nt + seg_n0[sg] + col, y_acc, rd + j * ty);
                }
                if constexpr (TOPK > 0) {
                    // a bf16 tile in face order: element (r, c) of face (r / 16) * 2 + c / 16 at face * 256 + (r % 16) *
                    // 16 + c % 16, so row t's 32 values are 16 words, two values each (the even column in the low
                    // half); the key flip is branchless and the row's smallest kept key is held in a register
                    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd + j * ty);
                    const uint32_t g0 = (n0 + col) * 32;
                    // the tile's valid columns (none past the head's last element)
                    const uint32_t cn = g0 >= n_elems ? 0 : (n_elems - g0 < 32 ? n_elems - g0 : 32);
                    for (uint32_t t = 0; t < TROWS; ++t) {
                        uint32_t floor = key[t][low[t]];
                        for (uint32_t h = 0; h < 2; ++h) {        // face 0 then face 1
                            const uint32_t w0 = h * 128 + t * 8;  // the row's first word in the face
                            for (uint32_t q2 = 0; q2 < 8; ++q2) {
                                const uint32_t word = e[w0 + q2];
                                // a quick reject: once the row's smallest kept key is a positive value's (key >=
                                // 0x8000, raw bits key - 0x8000), a value beats it only if its bf16, as a signed int16,
                                // is above those raw bits (a negative bf16 is negative as int16). Both halves at or
                                // below: nothing of this word can enter, as the full test below would find; it only
                                // rejects, so the kept set is the same.
                                if (floor >= 0x8000u) {
                                    const int32_t fr = static_cast<int32_t>(floor - 0x8000u);
                                    const int32_t lo16 = static_cast<int32_t>(word << 16) >> 16;
                                    const int32_t hi16 = static_cast<int32_t>(word) >> 16;
                                    if (lo16 <= fr && hi16 <= fr) {
                                        continue;
                                    }
                                }
                                for (uint32_t half = 0; half < 2; ++half) {
                                    const uint32_t c = h * 16 + q2 * 2 + half;
                                    const uint32_t b = half ? (word >> 16) : (word & 0xFFFFu);
                                    const uint32_t k = b ^ (0x8000u | (0x7FFFu & (0u - (b >> 15))));
                                    if (k > floor && c < cn) {
                                        key[t][low[t]] = k;
                                        idx[t][low[t]] = base + g0 + c;
                                        uint32_t lo = 0;
                                        for (uint32_t q = 1; q < TOPK; ++q) {
                                            if (key[t][q] < key[t][lo]) {
                                                lo = q;
                                            }
                                        }
                                        low[t] = lo;
                                        floor = key[t][lo];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (WRITE_Y) {
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, SBW);
        }
    }
    if constexpr (WRITE_Y) {
        noc_async_write_barrier();  // every output acknowledged before the kernel ends
    }
    if constexpr (TOPK > 0) {
        // the record: TROWS x TOPK keys, then TROWS x TOPK indices (uint32 each), staged in c_7
        constexpr uint32_t RBYTES = 2 * TROWS * TOPK * 4;
        constexpr uint32_t cb_rec = 7;
        const auto r_acc = TensorAccessor(r_args, r_addr, RBYTES);
        cb_reserve_back(cb_rec, 1);
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_rec));
        for (uint32_t t = 0; t < TROWS; ++t) {
            for (uint32_t q = 0; q < TOPK; ++q) {
                rec[t * TOPK + q] = key[t][q];
                rec[TROWS * TOPK + t * TOPK + q] = idx[t][q];
            }
        }
        noc_async_write_page(slot, r_acc, get_write_ptr(cb_rec));
        noc_async_write_barrier();
    }
}
