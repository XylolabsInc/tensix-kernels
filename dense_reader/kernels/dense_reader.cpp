// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of the dense reader's matmul (dense_compute.cpp): the core's scale tiles once (unless NOSCALE), then per x
// block its x tiles and the weight chunks in the order the compute consumes them (with SPLIT only the reader's share
// of the chunks; the writer loads the others into the second weight CB). The weight loads are in dense_wload.h.
//
// Weight layouts:
//   W_SHARDED = 1: DRAM width-sharded: bank `bank` holds a shard of KT x pitch tiles, row-major, at the buffer
//     address; this core reads columns col0.. of it.
//   W_SHARDED = 0: interleaved, KT x pitch tiles row-major (pitch = N tiles), page ids.
// x: tile rows mt0 .. mt0 + MT of [* x KT] tiles, interleaved (DRAM or L1), page id m * KT + k.
// scale: [1, 1, 1, N] bf16 TILE, interleaved: page n is output column tile n.
// Columns j >= nvalid of the core's NC are not read (their slots keep whatever they held; the writer drops those
// outputs), so no read leaves the shard.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dense_wload.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2;

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t KB = get_compile_time_arg_val(2);
    constexpr uint32_t KW = get_compile_time_arg_val(3);
    constexpr uint32_t SBW = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(6);
    constexpr bool W_SHARDED = get_compile_time_arg_val(7) != 0;
    // SPLIT: the writer RISC loads some of the weight chunks (split.h's pattern: 1 every other one, else a period)
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(8);
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr bool ROWS = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t TB = get_compile_time_arg_val(10);  // a weight tile's bytes
    // XS (ROWS, one x block): x streams with the weights, MT x KW tiles per chunk (the compute's XS); with SPLIT the
    // reader reads the x of its chunk and of the writer's chunk after it together, under its own chunk's barrier
    constexpr bool XS = get_compile_time_arg_val(11) != 0;
    // NOSCALE: no scale tensor (unit scales); its accessor slot holds x's
    constexpr bool NOSCALE = get_compile_time_arg_val(12) != 0;
    // GLU2 (> 0): the GLU's gate and up as two weights (dense_wload.h HALF): the up's columns at row page GLU2 of each
    // chunk row, its address after vc
    constexpr uint32_t GLU2 = get_compile_time_arg_val(13);
    constexpr uint32_t REQ = get_compile_time_arg_val(14);  // a row's requests at most REQ tiles (0: one a row)
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    // SEGS (> 1, a block list; XS with sharded weights): the core's columns of SEGS blocks, each DRAM-sharded on its
    // own, as segments in sequence: every chunk of segment 0, then of segment 1, ... (x streamed again for each), one
    // chunk stream across them (the split pattern included)
    constexpr uint32_t SEGS = get_compile_time_arg_val(w_args.next_compile_time_args_offset());
    constexpr uint32_t NKB = KT / KB;
    constexpr uint32_t NS = NC / SBW;
    static_assert(W_PAGES % (KW * (ROWS ? NC : SBW)) == 0, "dense: the weight CB must hold whole chunks");
    // (at function scope: a static_assert in a discarded if-constexpr branch of a non-template function is still
    // checked)
    static_assert(!XS || (ROWS && NKB == 1), "dense: XS needs ROWS and one x block");
    static_assert(GLU2 == 0 || (ROWS && W_SHARDED && NC == 2 * GLU2), "dense: two weights");
    static_assert(SEGS >= 1 && SEGS <= 4, "dense: 1 to 4 segments");
    static_assert(SEGS == 1 || (XS && W_SHARDED && NOSCALE), "dense: segments stream x with sharded weights");

    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t pitch = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);
    const uint32_t mt0 = get_arg_val<uint32_t>(i++);  // first tile row of x (row slices of prefill)
    const uint32_t vc = get_arg_val<uint32_t>(i++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(i++);  // GLU2: the up weight (else unused)
    // each segment's weight, pitch, the core's first column and valid columns in it, up weight (segment 0's above;
    // the others' after them, five each)
    uint32_t seg_w[SEGS], seg_pitch[SEGS], seg_col0[SEGS], seg_nvalid[SEGS], seg_w2[SEGS];
    seg_w[0] = w_addr;
    seg_pitch[0] = pitch;
    seg_col0[0] = col0;
    seg_nvalid[0] = nvalid;
    seg_w2[0] = w2_addr;
    for (uint32_t s = 1; s < SEGS; ++s) {
        seg_w[s] = get_arg_val<uint32_t>(i++);
        seg_pitch[s] = get_arg_val<uint32_t>(i++);
        seg_col0[s] = get_arg_val<uint32_t>(i++);
        seg_nvalid[s] = get_arg_val<uint32_t>(i++);
        seg_w2[s] = get_arg_val<uint32_t>(i++);
    }

    const uint32_t tx = get_tile_size(cb_x);
    const auto x_acc = TensorAccessor(x_args, x_addr, tx);
    const auto w_acc = TensorAccessor(w_args, w_addr, TB);

    tk::Loader<W_SHARDED, decltype(w_acc), TB, GLU2, REQ> ld{cb_w, w_acc, w_addr, bank, pitch, vc};
    ld.w2_addr = w2_addr;

    // the scales of the core's columns
    if constexpr (!NOSCALE) {
        const uint32_t ts = get_tile_size(cb_s);
        const auto s_acc = TensorAccessor(s_args, s_addr, ts);
        cb_reserve_back(cb_s, NC);
        const uint32_t ls = get_write_ptr(cb_s);
        for (uint32_t j = 0; j < nvalid; ++j) {
            noc_async_read_page(n0 + j, s_acc, ls + j * ts);
        }
        noc_async_read_barrier();
        cb_push_back(cb_s, NC);
    }

    // XS: G chunks in all, chunk c of segment c / NCH at its K chunk c % NCH (SEGS 1: c itself); the loader takes the
    // segment's weight before each of its chunks
    constexpr uint32_t NCH = KT / KW, G = SEGS * NCH;
    [[maybe_unused]] auto seg_of = [](uint32_t c) -> uint32_t { return SEGS == 1 ? 0 : c / NCH; };
    [[maybe_unused]] auto kc_of = [](uint32_t c) -> uint32_t { return SEGS == 1 ? c : c % NCH; };
    [[maybe_unused]] auto use_seg = [&](uint32_t sg) {
        if constexpr (SEGS > 1) {
            ld.w_addr = seg_w[sg];
            ld.pitch = seg_pitch[sg];
            ld.w2_addr = seg_w2[sg];
        }
    };
    if constexpr (XS) {
        // a group: the reader's chunk c and, with SPLIT, the writer's chunk right after it when there is one
        // (split.h: the writer never has two in a row, and chunk 0 is the reader's). The alternation (SPLIT_ARG 1)
        // reserves x for the whole group (always two chunks but a row's odd last one: the x CB of XS_DEPTH x 2 chunks
        // takes them without straddling its end); a pattern mixes groups of 2 and 1, so its x goes in one chunk at a
        // time instead: a reservation that straddled the CB's end would write past it.
        for (uint32_t c = 0; c < G;) {
            const uint32_t n = (SPLIT && c + 1 < G && tk::split_writer<SPLIT_ARG>(c + 1)) ? 2 : 1;
            const uint32_t sg = seg_of(c);
            use_seg(sg);
            if constexpr (SPLIT_ARG > 1) {
                for (uint32_t q = 0; q < n; ++q) {
                    cb_reserve_back(cb_x, MT * KW);
                    const uint32_t lq = get_write_ptr(cb_x);
                    for (uint32_t m = 0; m < MT; ++m) {
                        for (uint32_t k = 0; k < KW; ++k) {
                            noc_async_read_page((mt0 + m) * KT + kc_of(c + q) * KW + k, x_acc, lq + (m * KW + k) * tx);
                        }
                    }
                    if (q == 0) {
                        // its barrier covers chunk c's x reads
                        ld.template chunk<KW, NC>(kc_of(c) * KW, seg_col0[sg], seg_nvalid[sg]);
                    } else {
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_x, MT * KW);
                }
                c += n;
                continue;
            }
            cb_reserve_back(cb_x, MT * KW * n);
            const uint32_t lx = get_write_ptr(cb_x);
            for (uint32_t q = 0; q < n; ++q) {
                for (uint32_t m = 0; m < MT; ++m) {
                    for (uint32_t k = 0; k < KW; ++k) {
                        noc_async_read_page(
                            (mt0 + m) * KT + kc_of(c + q) * KW + k, x_acc, lx + ((q * MT + m) * KW + k) * tx);
                    }
                }
            }
            ld.template chunk<KW, NC>(kc_of(c) * KW, seg_col0[sg], seg_nvalid[sg]);  // its barrier covers the x reads
            cb_push_back(cb_x, MT * KW * n);
            c += n;
        }
        return;
    }
    uint32_t g = 0;
    for (uint32_t b = 0; b < NKB; ++b) {
        cb_reserve_back(cb_x, MT * KB);
        const uint32_t lx = get_write_ptr(cb_x);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t k = 0; k < KB; ++k) {
                noc_async_read_page((mt0 + m) * KT + b * KB + k, x_acc, lx + (m * KB + k) * tx);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, MT * KB);
        if constexpr (ROWS) {
            g = tk::load_block_rows<KB, KW, NC, SPLIT_ARG>(ld, col0, nvalid, b, g, SPLIT, 0);
        } else {
            g = tk::load_block<KB, KW, SBW, NS, SPLIT_ARG>(ld, col0, nvalid, b, g, SPLIT, 0);
        }
    }
}
