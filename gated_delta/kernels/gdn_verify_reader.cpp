// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the multi-row decode kernel (gdn_verify_compute.cpp): R rows (tokens) of the decode
// step run in order on one state, as a speculative verify does; R = 1 is the decode step. Once per
// call, after the first group's gates, the constants: the ones tile, the shift tile L and (GNORM)
// the norm weight's tiles (row 0 of each, widened to fp32). The core's heads go in groups of up to 32 / R; an item is one head's row
// (item j of a group: head j / R, row j % R). Per group:
// - the gates of its items: item j's raw a and b (or g and beta) in column j of row 0 of one fp32
//   page each, with dt_bias and a_neg (RAW) in two more pages;
// - its raw q and k rows gathered into one tile set each (row j = item j; cb_qg, cb_kg), its
//   output gate rows likewise (GNORM; cb_gg), its v rows into a reader-only set: each head's tiles
//   staged once (one tile row holds all R tokens) and their rows copied by RISC-V;
// - then per item its state tiles at the head's first row, and row j of the group's normalized q
//   and k (cb_qn, cb_kn, from the compute) and of its v rows copied into row 0 of the compute's
//   input slots (the other rows zero). The copies are byte copies, so every value is the one the
//   norms (or the inputs) gave.
// Every input is read by tile column from a flat [B, R, W] tensor, as convolution and fused
// projection emit them. The
// tile helpers are the prefill reader's (gdn_reader.cpp, adapted from tenstorrent/tt-metal
// v0.81.1's chunk_gated_delta_rule, Apache-2.0).
//
// Zero pages with NOC reads of the core's zeros region. Pack each group's gates
// into one page, and gather q/k and output rows for grouped normalization.
// Push constants in two stages: the ones tile after the first group's gates,
// then L and the norm weight after gathering rows. This lets q/k normalization
// start without waiting for the shift tile or weight reads.
// SSM (compile-time arg 4) uses q and k as C and B, shared by each group of G
// heads, and v as x. Each head supplies dt_bias, a_neg, and D as runtime values.
// Pack D per item into row 0 of cb_dD, the compute kernel's otherwise unused cb_t1a.
// POISON (compile-time arg 3) is diagnostic; use 0 for normal calls. Nonzero values
// fill unused gate rows (1-31) and rows past the group's items in gathered q, k,
// and gate tiles with NaN. An unintended read then propagates to the output.
// POISON 2 also fills item-slot rows 1-31, which the outer product reads. This
// control must change outputs to show that NaNs reach the compute kernel.

#include <cstdint>
// DeviceZoneScopedN marks device profiler zones.
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_qg = 0, cb_kg = 1, cb_v = 2, cb_gab = 3, cb_gg = 5, cb_S0 = 6, cb_ones = 7;
constexpr uint32_t cb_t16 = 23;  // the state's FPU copy (gdn_verify.py's CB plan)
constexpr uint32_t cb_q = 8, cb_k = 9, cb_gd = 24, cb_gn = 25, cb_qn = 27, cb_kn = 30;
// Reader-only (never pushed): the gates' scratch, then a head's q, k, v tiles, its gate tiles and
// the group's v rows.
constexpr uint32_t cb_rbuf = 26;
constexpr uint32_t cb_dD = 21;  // SSM: D per item (row 0), the compute's cb_t1a
#ifdef GDN_FOLD
constexpr uint32_t cb_co = 28;  // the fused convolution: the conv's output, R bf16 tiles from the compute
#ifndef GDN_FOLD_SKIP
#define GDN_FOLD_SKIP 0  // Diagnostic skip bits: 2 omits packing, 8 omits gathering; results are invalid.
#endif
#endif
constexpr uint32_t GATES = 32;  // values per gate page (the columns of its row 0); rows per gathered tile
constexpr uint32_t SLOT = 128;  // bytes of scratch per element read (a face row, at the NOC's alignment)
constexpr uint32_t SCR = 2 * GATES * SLOT;  // the scratch: a and b of 32 values
constexpr uint32_t ONE_F32 = 0x3f800000u;

// Element (i, j) of a 32x32 tile stored as four 16x16 faces.
inline uint32_t tidx(uint32_t i, uint32_t j) { return ((i >> 4) << 9) + ((i & 15) << 4) + ((j >> 4) << 8) + (j & 15); }

// bytes (a multiple of 512) at addr zeroed by NOC reads of the zeros region; the caller waits
// (noc_async_read_barrier) before it writes there or hands the page on.
__attribute__((noinline)) void noc_zero(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(z, addr + off, MEM_ZEROS_SIZE);
    }
}

// Copy two face rows of N words each (s0 -> d0, s1 -> d1). Issue all loads before
// the first store so the RISC-V's L1 loads overlap. Each source word is copied bit for bit.
template <uint32_t N>
FORCE_INLINE void copy_faces(volatile tt_l1_ptr uint32_t* s0, volatile tt_l1_ptr uint32_t* s1,
                             volatile tt_l1_ptr uint32_t* d0, volatile tt_l1_ptr uint32_t* d1) {
    uint32_t a[N], b[N];
#pragma GCC unroll 16
    for (uint32_t w = 0; w < N; w++) {
        a[w] = s0[w];
    }
#pragma GCC unroll 16
    for (uint32_t w = 0; w < N; w++) {
        b[w] = s1[w];
    }
#pragma GCC unroll 16
    for (uint32_t w = 0; w < N; w++) {
        d0[w] = a[w];
    }
#pragma GCC unroll 16
    for (uint32_t w = 0; w < N; w++) {
        d1[w] = b[w];
    }
}

// Row rs of the tile at src into row rd of the tile at dst (both elem bytes per element): its two
// 16-element face rows; the rest of dst is left as it is.
// Inline this copy to avoid a function call for each row.
FORCE_INLINE void copy_row(uint32_t src, uint32_t rs, uint32_t dst, uint32_t rd, uint32_t elem) {
    const uint32_t words = 4 * elem;  // 16 elements
    volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + tidx(rs, 0) * elem);
    volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + tidx(rs, 16) * elem);
    volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + tidx(rd, 0) * elem);
    volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + tidx(rd, 16) * elem);
    if (words == 8) {
        copy_faces<8>(s0, s1, d0, d1);
    } else {
        copy_faces<8>(s0, s1, d0, d1);  // fp32: the face row's 16 words in two halves
        copy_faces<8>(s0 + 8, s1 + 8, d0 + 8, d1 + 8);
    }
}

#ifdef GDN_FOLD
// The fused convolution's row moves (bf16 only): gdn_conv_reader.cpp's move_row, inline with the offsets known, so the packing, the
// shift and the gather from the conv's output cost no call each. The same bytes as copy_row(src, r, dst, q, 2).
FORCE_INLINE void move_row(uint32_t src, uint32_t r, uint32_t dst, uint32_t q) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + tidx(r, 0) * 2);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + tidx(q, 0) * 2);
    copy_faces<8>(s, s + 128, d, d + 128);  // the second face: 256 elements = 128 words further
}
#endif
// The face row holding element (r, col) of tile `page` (src_elem bytes each) into the scratch slot
// at l1, at the source address's offset modulo 64 (the NOC moves data only between addresses alike
// modulo 64; the slot holds 63 + 64 bytes); returns where element (r, col & ~15) landed.
template <typename Acc>
FORCE_INLINE uint32_t read_elem(const Acc& acc, uint32_t page, uint32_t r, uint32_t col, uint32_t l1, uint32_t src_elem) {
    const uint64_t src = acc.get_noc_addr(page, tidx(r, col & ~15u) * src_elem);
    const uint32_t at = l1 + ((static_cast<uint32_t>(src) - l1) & 63);
    noc_async_read(src, at, 16 * src_elem);
    return at;
}

// After the read landed: element col of the face row at `at` as fp32 bits.
FORCE_INLINE uint32_t elem_bits(uint32_t at, uint32_t col, uint32_t src_elem) {
    volatile tt_l1_ptr uint16_t* s16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(at);
    volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at);
    return (src_elem == 2) ? (static_cast<uint32_t>(s16[col & 15]) << 16) : s32[col & 15];
}

// POISON diagnostic: fill bytes at addr with NaN in elements of esz bytes (2: bf16,
// 4: fp32). The page helper fills rows 1-31 of an fp32 page.
__attribute__((noinline)) void poison(uint32_t addr, uint32_t bytes, uint32_t esz) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t nan = (esz == 2) ? 0x7fc07fc0u : 0x7fc00000u;
    for (uint32_t w = 0; w < bytes / 4; w++) {
        p[w] = nan;
    }
    asm volatile("fence" ::: "memory");
}
__attribute__((noinline)) void poison_rows(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t r = 1; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            p[tidx(r, c)] = 0x7fc00000u;
        }
    }
    asm volatile("fence" ::: "memory");
    (void)bytes;
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);  // rows (tokens), 1..32
    constexpr uint32_t POISON = get_compile_time_arg_val(3);  // POISON diagnostic: unused rows NaN
    constexpr uint32_t GH = GATES / R;                   // heads per group
    constexpr uint32_t SSM = get_compile_time_arg_val(4);  // Mamba-2's step (see above)
    constexpr uint32_t F32S2 = get_compile_time_arg_val(5);  // the state also into cb_t16 (the FPU's copy)
    constexpr uint32_t PER_HEAD = SSM ? 3 : 2;              // runtime values per head
    constexpr auto x_a = TensorAccessorArgs<6>();        // the conv output: q, k, v (after the 6 args above)
    constexpr auto a_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();  // a (or g)
    constexpr auto b_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();  // b (or beta)
    constexpr auto gt_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();  // the gate
    constexpr auto s_a = TensorAccessorArgs<gt_a.next_compile_time_args_offset()>();  // the state
    constexpr auto c_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();  // ones
    constexpr auto w_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();  // norm weight
#ifdef GDN_FOLD
    // The fused convolution: x is the fused projection output (the conv's input); its conv state [B, 3, C], the taps packed per core
    // (gdn_verify.py: 5 pages a core, row c = the core's channel tile c), the window buffer [B, 3 + R, C] (WINDOW)
    constexpr auto cs_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr auto tp_a = TensorAccessorArgs<cs_a.next_compile_time_args_offset()>();
    constexpr auto wb_a = TensorAccessorArgs<tp_a.next_compile_time_args_offset()>();
#endif
#ifdef GDN_KDA
#ifdef GDN_FOLD
#error "gdn_verify: KDA with the conv fold is not written yet"
#endif
    // KDA dt_bias: one fp32 page per value head; rows 0..Kt-1 hold its K values.
    constexpr auto db_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();
    constexpr bool KDA = true;
#else
    constexpr bool KDA = false;
#endif

    uint32_t i = 0;
    const uint32_t h0 = get_arg_val<uint32_t>(i++);
    const uint32_t nh = get_arg_val<uint32_t>(i++);
    const uint32_t H = get_arg_val<uint32_t>(i++);
    const uint32_t HV = get_arg_val<uint32_t>(i++);
    const uint32_t x_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t q_col = get_arg_val<uint32_t>(i++);
    const uint32_t k_col = get_arg_val<uint32_t>(i++);
    const uint32_t v_col = get_arg_val<uint32_t>(i++);
    const uint32_t a_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t a_col = get_arg_val<uint32_t>(i++);  // element column of head 0's a (or g)
    const uint32_t b_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t b_col = get_arg_val<uint32_t>(i++);
    const uint32_t a_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t b_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t gnorm = get_arg_val<uint32_t>(i++);
    const uint32_t gt_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t gt_col = get_arg_val<uint32_t>(i++);
    const uint32_t w_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t a_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gt_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t c_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t raw = get_arg_val<uint32_t>(i++);
#ifdef GDN_FOLD
    const uint32_t f_x0 = get_arg_val<uint32_t>(i++);   // x's tile column of the conv's channel 0
    const uint32_t f_ct = get_arg_val<uint32_t>(i++);   // the conv's channel tiles (C / 32)
    const uint32_t cs_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tp_addr = get_arg_val<uint32_t>(i++);
    const uint32_t wb_addr = get_arg_val<uint32_t>(i++);
    const uint32_t tp_page = get_arg_val<uint32_t>(i++);  // this core's first packed taps page
#endif
#ifdef GDN_KDA
    const uint32_t f_col = get_arg_val<uint32_t>(i++);    // x's tile column of head 0's f (KDA)
    const uint32_t db_addr = get_arg_val<uint32_t>(i++);  // dt_bias's pages (KDA)
#endif
    const uint32_t rt_heads = i;  // then per head: dt_bias, a_neg (RAW; fp32 bits)

    constexpr uint32_t kv = Kt * Vt;
    const uint32_t G = HV / H;
    const uint32_t tx = get_tile_size(cb_qg);
    const uint32_t tg = get_tile_size(cb_gab);  // fp32 page
    const uint32_t tgt = get_tile_size(cb_gg);
    const uint32_t ts = get_tile_size(cb_S0);
    const uint32_t tf = get_tile_size(cb_ones);
    const uint32_t tq = get_tile_size(cb_q);  // fp32
    const auto x_acc = TensorAccessor(x_a, x_addr, tx);
    const auto a_acc = TensorAccessor(a_a, a_addr, a_bytes);
    const auto b_acc = TensorAccessor(b_a, b_addr, b_bytes);
    const auto gt_acc = TensorAccessor(gt_a, gt_addr, tgt);
    const auto s_acc = TensorAccessor(s_a, s_addr, ts);
    const auto c_acc = TensorAccessor(c_a, c_addr, tf);
    const auto w_acc = TensorAccessor(w_a, w_addr, w_bytes ? w_bytes : tf);
    const uint32_t scr = get_write_ptr(cb_rbuf);
    const uint32_t sq = scr + SCR, sk = sq + Kt * tx, sv = sk + Kt * tx, sg = sv + Vt * tx;
    const uint32_t vg = sg + Vt * tgt;  // the group's v rows (Vt tiles)
#ifdef GDN_KDA
    // KDA's staging after vg, by the parity of the head's index on this core (the next head's read ahead): its f tiles
    // (Kt pages of x: row r = row r's f) and its dt_bias page
    const uint32_t ksf = vg + Vt * tx, ksd = ksf + 2 * Kt * tx;
    const auto db_acc = TensorAccessor(db_a, db_addr, tf);
#endif
#ifdef GDN_FOLD
    // The fused convolution's staging, after vg: the core's NC x pages (rows 0..R-1 of the fused projection output), its NC conv-state pages
    // (rows 0..2), (WINDOW) its NC window pages; every conv tensor is bf16 TILE, as x (gdn_verify.py). The channel
    // tiles: per key head on this core (its value heads are all here: gdn_verify.py) q's Kt then k's Kt, then each
    // head's v Vt. The conv's output (cb_co) holds channel tile c in row c of tile r for token r.
    const uint32_t G0 = HV / H;
    const uint32_t f_nk = nh / G0;
    const uint32_t f_nc = f_nk * 2 * Kt + nh * Vt;
    const uint32_t fx = vg + Vt * tx, fs = fx + GDN_FOLD_NCMAX * tx;
    const uint32_t sga = fs + GDN_FOLD_NCMAX * tx;  // every head's gate tiles (GNORM), read with the conv's inputs
    const auto cs_acc = TensorAccessor(cs_a, cs_addr, tx);
    const auto tp_acc = TensorAccessor(tp_a, tp_addr, tx);
    (void)wb_addr;  // the window is the writer's (gdn_verify_writer.cpp), from this staging
    // channel tile c: its batch row and its tile column in x
    auto f_chan = [&](uint32_t c, uint32_t& b, uint32_t& col) {
        if (c < f_nk * 2 * Kt) {
            const uint32_t bh = h0 + (c / (2 * Kt)) * G0;  // the key head's first value head
            const uint32_t hk = (bh % HV) / G0, t = c % (2 * Kt);
            b = bh / HV;
            col = (t < Kt) ? (q_col + hk * Kt + t) : (k_col + hk * Kt + t - Kt);
        } else {
            const uint32_t e = c - f_nk * 2 * Kt;
            const uint32_t bh = h0 + e / Vt;
            b = bh / HV;
            col = v_col + (bh % HV) * Vt + e % Vt;
        }
    };
#endif

    // Constants: the ones tile, shift tile L, and optional norm weight. L[c + 1, c]
    // is 1, so E @ L moves column c + 1 to c with E in srcB. Each weight tile has
    // fp32 weights in row 0 and zeros elsewhere; normalization broadcasts that row.
    // Build constants after pushing the first group's gates. Batch weight reads
    // behind one barrier so compute can process gates while reads are in flight.
    const uint32_t wpage = w_bytes ? w_bytes : tf;
    uint32_t c = 0;  // the constants' tile 0 (the CB is never popped: its base)
    // Push the ones tile first, on its own.
    auto consts_ones = [&]() {
        DeviceZoneScopedN("R-const1");
        cb_reserve_back(cb_ones, 1);
        c = get_write_ptr(cb_ones);
        noc_async_read_page(0, c_acc, c);
        noc_async_read_barrier();
        cb_push_back(cb_ones, 1);
    };
    // then L and (GNORM) the weight's tiles, tiles 1 .. nconst - 1
    auto consts = [&]() {
        DeviceZoneScopedN("R-const");
        // KDA: after the weight, E_i (row i ones) then G_i (column i ones), i < Kt, read whole from the
        // dt_bias tensor's pages HV .. HV + 2 Kt - 1 (host-made) in this batch
        const uint32_t nzero = 2 + (gnorm ? Vt : 0);
        const uint32_t nconst = nzero + (KDA ? 2 * Kt : 0);
        cb_reserve_back(cb_ones, nconst - 1);
        for (uint32_t t = 1; t < nzero; t++) {
            noc_zero(c + t * tf, tf);
        }
        if (gnorm) {
            // The weight tiles land in the scratch (free again once the gates are pushed; the
            // staging after it is free until the gather), tile t at scr + t * wpage.
            for (uint32_t t = 0; t < Vt; t++) {
                noc_async_read_page(t, w_acc, scr + t * wpage);
            }
        }
#ifdef GDN_KDA
        for (uint32_t t = 0; t < 2 * Kt; t++) {
            noc_async_read_page(HV + t, db_acc, c + (nzero + t) * tf);
        }
#endif
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c + tf);
        for (uint32_t col = 0; col + 1 < 32; col++) {
            sl[tidx(col + 1, col)] = ONE_F32;
        }
        if (gnorm) {
            for (uint32_t t = 0; t < Vt; t++) {
                volatile tt_l1_ptr uint32_t* w32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c + (2 + t) * tf);
                volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + t * wpage);
                volatile tt_l1_ptr uint16_t* s16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + t * wpage);
                for (uint32_t j = 0; j < 32; j++) {
                    w32[tidx(0, j)] = (w_bytes < tf) ? (static_cast<uint32_t>(s16[tidx(0, j)]) << 16) : s32[tidx(0, j)];
                }
            }
        }
        asm volatile("fence" ::: "memory");
        cb_push_back(cb_ones, nconst - 1);
    };

    // The item slots hold one item each and the gathered sets one group each, always at their CBs'
    // bases: zero them once (rows past a group's items are never read back).
    noc_zero(get_write_ptr(cb_q), Kt * tq);
    noc_zero(get_write_ptr(cb_k), Kt * tq);
    noc_zero(get_write_ptr(cb_v), Vt * tx);
#ifdef GDN_FOLD
    // cb_qg first carries the packed window (GDN_FOLD_WP tiles; its rows past the core's channel tiles stay zero)
    noc_zero(get_write_ptr(cb_qg), GDN_FOLD_WP * tx);
#else
    noc_zero(get_write_ptr(cb_qg), Kt * tx);
#endif
    noc_zero(get_write_ptr(cb_kg), Kt * tx);
#ifdef GDN_KDA
    noc_zero(get_write_ptr(cb_gd), tf);  // f's page, rows 0..Kt-1 written per item (KDA)
    noc_zero(get_write_ptr(cb_gn), tf);  // dt_bias's page, likewise
#endif
    if (gnorm) {
        noc_zero(get_write_ptr(cb_gg), Vt * tgt);
    }
    noc_async_read_barrier();
    if constexpr (POISON == 2) {  // the control: the item slots too (row 0 is rewritten per item)
        poison(get_write_ptr(cb_q), Kt * tq, tq / 1024);
        poison(get_write_ptr(cb_k), Kt * tq, tq / 1024);
        poison(get_write_ptr(cb_v), Vt * tx, tx / 1024);
    }
    if constexpr (POISON) {
        // the gathered sets NaN (a group's rows overwrite theirs): 16-bit or 32-bit quiet NaNs
        poison(get_write_ptr(cb_qg), Kt * tx, tx / 1024);
        poison(get_write_ptr(cb_kg), Kt * tx, tx / 1024);
        if (gnorm) {
            poison(get_write_ptr(cb_gg), Vt * tgt, tgt / 1024);
        }
    }
    const uint32_t ex = tx / 1024, egt = tgt / 1024, ea = a_bytes / 1024, eb = b_bytes / 1024, eq = tq / 1024;
#ifdef GDN_FOLD
    // The conv's inputs, read with the first group's gates (whose barrier waits for them): the taps into cb_kg, each
    // channel tile's x page and conv-state page into the staging.
    cb_reserve_back(cb_kg, GDN_FOLD_WT);
    cb_reserve_back(cb_qg, GDN_FOLD_WP);
    {
        DeviceZoneScopedN("R-fread");
        const uint32_t wt = get_write_ptr(cb_kg);
        for (uint32_t k = 0; k < 5; k++) {
            noc_async_read_page(tp_page + k, tp_acc, wt + k * tx);
        }
        for (uint32_t c = 0; c < f_nc; c++) {
            uint32_t b, col;
            f_chan(c, b, col);
            noc_async_read_page(b * x_pitch + col, x_acc, fx + c * tx);
            noc_async_read_page(b * f_ct + col - f_x0, cs_acc, fs + c * tx);
        }
        for (uint32_t hl = 0; gnorm && hl < nh; hl++) {  // no DRAM read in the gather
            const uint32_t bh = h0 + hl;
            for (uint32_t t = 0; t < Vt; t++) {
                noc_async_read_page((bh / HV) * gt_pitch + gt_col + (bh % HV) * Vt + t, gt_acc, sga + (hl * Vt + t) * tgt);
            }
        }
    }
#endif

    for (uint32_t hh = 0; hh < nh; hh += GH) {
        const uint32_t n = (nh - hh < GH) ? (nh - hh) : GH;  // heads in this group
        const uint32_t items = n * R;
        {
            // The gates of items 0 .. items - 1: item j (head hh + j / R, row j % R) in column j.
            DeviceZoneScopedN("R-gates");
            cb_reserve_back(cb_gab, 2);
            const uint32_t wa = get_write_ptr(cb_gab), wb = wa + tg;
            noc_zero(wa, 2 * tg);
            uint32_t wd = 0, wn = 0, wD = 0;
            if (raw && !KDA) {  // KDA: dt_bias and the decay per item (below)
                cb_reserve_back(cb_gd, 1);
                cb_reserve_back(cb_gn, 1);
                wd = get_write_ptr(cb_gd);
                wn = get_write_ptr(cb_gn);
                noc_zero(wd, tg);
                noc_zero(wn, tg);
            }
            if constexpr (SSM) {
                cb_reserve_back(cb_dD, 1);
                wD = get_write_ptr(cb_dD);
                noc_zero(wD, tg);
            }
            if constexpr (POISON) {
                noc_async_read_barrier();
                poison_rows(wa, tg);
                poison_rows(wa + tg, tg);
                if (raw && !KDA) {  // KDA: dt_bias and the decay per item (below)
                    poison_rows(wd, tg);
                    poison_rows(wn, tg);
                }
            }
            uint32_t la[GATES], lb[GATES];  // where each value landed
            for (uint32_t j = 0; j < items; j++) {
                const uint32_t bh = h0 + hh + j / R;
                const uint32_t b = bh / HV;
                const uint32_t hv = bh % HV;
                const uint32_t r = j % R;
                la[j] = read_elem(a_acc, b * a_pitch + (a_col + hv) / 32, r, (a_col + hv) % 32, scr + j * SLOT, ea);
                lb[j] = read_elem(b_acc, b * b_pitch + (b_col + hv) / 32, r, (b_col + hv) % 32, scr + (GATES + j) * SLOT, eb);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* a32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wa);
            volatile tt_l1_ptr uint32_t* b32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb);
            for (uint32_t j = 0; j < items; j++) {
                const uint32_t hl = hh + j / R;  // the head's index on this core
                const uint32_t hv = (h0 + hl) % HV;
                a32[tidx(0, j)] = elem_bits(la[j], (a_col + hv) % 32, ea);
                b32[tidx(0, j)] = elem_bits(lb[j], (b_col + hv) % 32, eb);
                if (raw && !KDA) {  // KDA: dt_bias and the decay per item (below)
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wd)[tidx(0, j)] =
                        get_arg_val<uint32_t>(rt_heads + PER_HEAD * hl);
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wn)[tidx(0, j)] =
                        get_arg_val<uint32_t>(rt_heads + PER_HEAD * hl + 1);
                }
                if constexpr (SSM) {
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wD)[tidx(0, j)] =
                        get_arg_val<uint32_t>(rt_heads + PER_HEAD * hl + 2);
                }
            }
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_gab, 2);
            if (raw && !KDA) {  // KDA: dt_bias and the decay per item (below)
                cb_push_back(cb_gd, 1);
                cb_push_back(cb_gn, 1);
            }
            if constexpr (SSM) {
                cb_push_back(cb_dD, 1);
            }
        }
#ifdef GDN_FOLD
        if (hh == 0) {
            // The window of channel tile c into row c of the packed tiles (gdn_conv_reader.cpp's packing: rows 0..2
            // the conv state, then x's R rows); the taps went in whole. The gates' barrier covered the reads.
            DeviceZoneScopedN("R-fpack");
            invalidate_l1_cache();
            // the staging is complete (the gates' barrier): the writer shifts the conv state and writes the window from
            // it while this packs (semaphore 0; the writer sets it back to 0)
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
            const uint32_t wi = get_write_ptr(cb_qg);
            for (uint32_t c = 0; c < f_nc && !(GDN_FOLD_SKIP & 2); c++) {
                for (uint32_t r = 0; r < 3; r++) {
                    move_row(fs + c * tx, r, wi + r * tx, c);
                }
                for (uint32_t t = 0; t < R; t++) {
                    move_row(fx + c * tx, t, wi + (3 + t) * tx, c);
                }
            }
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_qg, GDN_FOLD_WP);
            cb_push_back(cb_kg, GDN_FOLD_WT);
        }
#endif
        if (hh == 0) {
            consts_ones();
        }
#ifdef GDN_FOLD
        if (hh == 0) {
            DeviceZoneScopedN("R-fwait");
            cb_wait_front(cb_co, R);
            invalidate_l1_cache();
        }
#endif
        {
            // The group's q, k, v and gate rows: each head's tiles staged, row r to row j = i R + r.
            DeviceZoneScopedN("R-gather");
            cb_reserve_back(cb_qg, Kt);
            cb_reserve_back(cb_kg, Kt);
            if (gnorm) {
                cb_reserve_back(cb_gg, Vt);
            }
            const uint32_t wqg = get_write_ptr(cb_qg), wkg = get_write_ptr(cb_kg);
            const uint32_t wgg = gnorm ? get_write_ptr(cb_gg) : 0;
#ifdef GDN_FOLD
            if constexpr (POISON) {  // the window and the taps went through these CBs: the gathered sets NaN again
                poison(wqg, Kt * tx, tx / 1024);
                poison(wkg, Kt * tx, tx / 1024);
            }
            const uint32_t rco = get_read_ptr(cb_co);
            for (uint32_t hi = 0; hi < n; hi++) {
                // q, k and v rows from the conv's output (row c of tile r: channel tile c at token r); the gate's
                // tiles from DRAM, as below
                const uint32_t bh = h0 + hh + hi;
                const uint32_t b = bh / HV;
                const uint32_t hv = bh % HV;
                const uint32_t hl = hh + hi;   // the head's index on this core
                const uint32_t ck = (hl / G0) * 2 * Kt;  // its key head's first channel tile (q; k after Kt)
                const uint32_t cv = f_nk * 2 * Kt + hl * Vt;
                const uint32_t sgh = sga + hl * Vt * tgt;  // this head's gate tiles, read at the start
                (void)b;
                (void)hv;
                for (uint32_t r = 0; r < R; r++) {
                    const uint32_t j = hi * R + r;
                    const uint32_t co = rco + r * tx;
                    for (uint32_t t = 0; t < Kt && !(GDN_FOLD_SKIP & 8); t++) {
                        move_row(co, ck + t, wqg + t * tx, j);
                        move_row(co, ck + Kt + t, wkg + t * tx, j);
                    }
                    for (uint32_t t = 0; t < Vt; t++) {
                        if (!(GDN_FOLD_SKIP & 8)) {
                            move_row(co, cv + t, vg + t * tx, j);
                        }
                        if (gnorm) {
                            copy_row(sgh + t * tgt, r, wgg + t * tgt, j, egt);
                        }
                    }
                }
            }
#else
            for (uint32_t hi = 0; hi < n; hi++) {
                const uint32_t bh = h0 + hh + hi;
                const uint32_t b = bh / HV;
                const uint32_t hv = bh % HV;
                const uint32_t hk = hv / G;
                for (uint32_t t = 0; t < Kt; t++) {
                    noc_async_read_page(b * x_pitch + q_col + hk * Kt + t, x_acc, sq + t * tx);
                    noc_async_read_page(b * x_pitch + k_col + hk * Kt + t, x_acc, sk + t * tx);
                }
                for (uint32_t t = 0; t < Vt; t++) {
                    noc_async_read_page(b * x_pitch + v_col + hv * Vt + t, x_acc, sv + t * tx);
                    if (gnorm) {
                        noc_async_read_page(b * gt_pitch + gt_col + hv * Vt + t, gt_acc, sg + t * tgt);
                    }
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t r = 0; r < R; r++) {
                    const uint32_t j = hi * R + r;
                    for (uint32_t t = 0; t < Kt; t++) {
                        copy_row(sq + t * tx, r, wqg + t * tx, j, ex);
                        copy_row(sk + t * tx, r, wkg + t * tx, j, ex);
                    }
                    for (uint32_t t = 0; t < Vt; t++) {
                        copy_row(sv + t * tx, r, vg + t * tx, j, ex);
                        if (gnorm) {
                            copy_row(sg + t * tgt, r, wgg + t * tgt, j, egt);
                        }
                    }
                }
            }
#endif
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_qg, Kt);
            cb_push_back(cb_kg, Kt);
            if (gnorm) {
                cb_push_back(cb_gg, Vt);
            }
        }
        if (hh == 0) {
            consts();  // L and the weight, after the gather
        }
        // Read each head's state ahead of its first item and delay the barrier until
        // use. Start the first head while compute normalizes q and k. Start each
        // later head after its predecessor's first item frees cb_S0 by making S_dec.
        auto state_issue = [&](uint32_t bh) {
            cb_reserve_back(cb_S0, kv);
            const uint32_t w = get_write_ptr(cb_S0);
            for (uint32_t t = 0; t < kv; t++) {
                noc_async_read_page(bh * kv + t, s_acc, w + t * ts);
            }
#ifdef GDN_KDA
            {  // the head's f tiles and dt_bias page, into its parity's staging
                const uint32_t par = (bh - h0) & 1, b = bh / HV, hv = bh % HV;
                for (uint32_t t = 0; t < Kt; t++) {
                    noc_async_read_page(b * x_pitch + f_col + hv * Kt + t, x_acc, ksf + (par * Kt + t) * tx);
                }
                noc_async_read_page(hv, db_acc, ksd + par * tf);
            }
#endif
            return w;
        };
        uint32_t ws_next = state_issue(h0 + hh);
        {
            DeviceZoneScopedN("R-wait-norm");
            cb_wait_front(cb_qn, Kt);
            cb_wait_front(cb_kn, Kt);
        }
        invalidate_l1_cache();
        const uint32_t rq = get_read_ptr(cb_qn), rk = get_read_ptr(cb_kn);
        DeviceZoneScopedN("R-items");
        for (uint32_t j = 0; j < items; j++) {
            if (j % R == 0) {
                const uint32_t ws = ws_next;  // issued ahead
                noc_async_read_barrier();
                cb_push_back(cb_S0, kv);
                if constexpr (F32S2 != 0) {  // the same tiles, a local L1 copy, for the compute's k S on the FPU
                    cb_reserve_back(cb_t16, kv);
                    noc_async_read(get_noc_addr(ws), get_write_ptr(cb_t16), kv * ts);
                    noc_async_read_barrier();
                    cb_push_back(cb_t16, kv);
                }
            }
#ifdef GDN_KDA
            {
                // KDA: row r's f (bf16, Kt tiles' row r) widened into rows 0..Kt-1 of an fp32 page (row t = channels
                // 32 t ..), and the head's dt_bias rows; the pages' other rows stay zero (zeroed once)
                DeviceZoneScopedN("R-kf");
                const uint32_t par = (hh + j / R) & 1, r = j % R;
                cb_reserve_back(cb_gd, 1);
                cb_reserve_back(cb_gn, 1);
                volatile tt_l1_ptr uint32_t* fd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_gd));
                for (uint32_t t = 0; t < Kt; t++) {
                    volatile tt_l1_ptr uint16_t* s16 =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ksf + (par * Kt + t) * tx);
                    for (uint32_t col = 0; col < 32; col++) {
                        fd[tidx(t, col)] = static_cast<uint32_t>(s16[tidx(r, col)]) << 16;
                    }
                }
                const uint32_t wn = get_write_ptr(cb_gn);
                for (uint32_t t = 0; t < Kt; t++) {
                    copy_row(ksd + par * tf, t, wn, t, 4);
                }
                asm volatile("fence" ::: "memory");
                cb_push_back(cb_gd, 1);
                cb_push_back(cb_gn, 1);
            }
#endif
            cb_reserve_back(cb_q, Kt);
            cb_reserve_back(cb_k, Kt);
            const uint32_t wq = get_write_ptr(cb_q), wk = get_write_ptr(cb_k);
            for (uint32_t t = 0; t < Kt; t++) {
                copy_row(rq + t * tq, j, wq + t * tq, 0, eq);
                copy_row(rk + t * tq, j, wk + t * tq, 0, eq);
            }
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_q, Kt);
            cb_push_back(cb_k, Kt);
            cb_reserve_back(cb_v, Vt);
            const uint32_t wv = get_write_ptr(cb_v);
            for (uint32_t t = 0; t < Vt; t++) {
                copy_row(vg + t * tx, j, wv + t * tx, 0, ex);
            }
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_v, Vt);
            if (j % R == 0 && j / R + 1 < n) {
                ws_next = state_issue(h0 + hh + j / R + 1);  // the next head's state, during this row
            }
        }
        cb_pop_front(cb_qn, Kt);
        cb_pop_front(cb_kn, Kt);
    }
#ifdef GDN_FOLD
    cb_pop_front(cb_co, R);
    noc_async_write_barrier();  // the conv state and the window written before the kernel ends
#endif
}
