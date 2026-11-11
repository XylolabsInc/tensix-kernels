// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the multi-row decode kernel (gdn_verify_compute.cpp): per item (one head's row) the
// state after it into its slot (rows 0..R-2: the verify's slot buffers, each laid out as the state)
// or, for the head's last row, into the state buffer it was read from (in place: each head's pages
// are read before they are written, and no two heads share pages); the outputs into row r of the
// head's output tiles, written once all R rows are in: o, flat [B, R, HV*V]. With the gated norm
// (GNORM) the compute norms a group of items at once: this kernel gathers each item's o (row 0 of
// cb_orow, fp32) into row j of cb_oasm for it, and takes the group's normed rows back from cb_out
// (row j = item j). Every row move is a byte copy. Adapted from the decode-step writer
// (tenstorrent/tt-metal v0.81.1's chunk_gated_delta_rule writer, Apache-2.0).
//
// Zero staging tiles once with NOC reads of the zeros region, keeping rows R..31
// as zero padding. Gather each group's output rows for gated normalization.

#include <cstdint>
// DeviceZoneScopedN marks device profiler zones.
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_oasm = 4, cb_Sout = 15, cb_out = 16, cb_orow = 31;
constexpr uint32_t cb_stage = 28;  // writer-only staging of a head's output tiles (never pushed)
constexpr uint32_t GATES = 32;     // rows per gathered tile

inline uint32_t tidx(uint32_t i, uint32_t j) { return ((i >> 4) << 9) + ((i & 15) << 4) + ((j >> 4) << 8) + (j & 15); }
#ifdef GDN_FOLD
constexpr uint32_t cb_rbuf = 26;  // the reader's scratch: its convolution staging, this kernel's staging after it
#endif

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

// Row rs of the tile at src into row rd of the tile at dst (elem bytes per element).
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
// Row r of the bf16 tile at src into row q of the bf16 tile at dst (gdn_conv_reader.cpp's move_row, inline).
FORCE_INLINE void move_row(uint32_t src, uint32_t r, uint32_t dst, uint32_t q) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + tidx(r, 0) * 2);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + tidx(q, 0) * 2);
    copy_faces<8>(s, s + 128, d, d + 128);
}
#endif
__attribute__((noinline)) void noc_zero(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(z, addr + off, MEM_ZEROS_SIZE);
    }
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t GH = GATES / R;  // heads per group, as the compute's
    constexpr uint32_t POISON = get_compile_time_arg_val(3);  // POISON diagnostic: cb_oasm's unused rows NaN
    // compile-time arg 4: SSM (as the reader's; the writer does the same in both modes)
    constexpr auto o_a = TensorAccessorArgs<5>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();  // state and slots
#ifdef GDN_FOLD
    constexpr auto cs_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();  // the conv state
    constexpr auto wb_a = TensorAccessorArgs<cs_a.next_compile_time_args_offset()>();  // the window buffer (WINDOW)
#endif

    uint32_t i = 0;
    const uint32_t h0 = get_arg_val<uint32_t>(i++);
    const uint32_t nh = get_arg_val<uint32_t>(i++);
    const uint32_t HV = get_arg_val<uint32_t>(i++);
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gnorm = get_arg_val<uint32_t>(i++);
    const uint32_t slot0 = i;  // R - 1 slot addresses follow

    constexpr uint32_t kv = Kt * Vt;
    const uint32_t to = get_tile_size(cb_out);
    const uint32_t ts = get_tile_size(cb_Sout);
    const uint32_t tf = get_tile_size(cb_orow);  // fp32
    const uint32_t eo = to / 1024;
    const auto o_acc = TensorAccessor(o_a, o_addr, to);
#ifdef GDN_FOLD
    // the fused convolution: CB 28 carries the conv's output to the reader; this staging sits in the reader's cb_rbuf, at
    // GDN_FOLD_STAGE bytes (after the reader's own parts: gdn_verify.py)
    const uint32_t stage = get_write_ptr(cb_rbuf) + GDN_FOLD_STAGE;
#else
    const uint32_t stage = get_write_ptr(cb_stage);
#endif
    noc_zero(stage, Vt * to);
    if (gnorm) {
        // cb_oasm holds one group at its base: zeroed once (rows past a group's items stay finite)
        noc_zero(get_write_ptr(cb_oasm), Vt * tf);
    }
    noc_async_read_barrier();
    if constexpr (POISON) {
        if (gnorm) {  // the rows past a group's items NaN (the items' rows are overwritten), fp32
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_oasm));
            for (uint32_t w = 0; w < Vt * tf / 4; w++) {
                p[w] = 0x7fc00000u;
            }
            asm volatile("fence" ::: "memory");
        }
    }

#ifdef GDN_FOLD
    {
        // The fused convolution's state shift and (WINDOW) window, off the reader's path: once the reader's staging holds each
        // channel tile's x page and conv-state page (semaphore 0, set by the reader after its read barrier), each
        // conv-state page copied whole, its rows 0..2 <- window rows R..R+2, and written back; the window
        // [s0, s1, s2, x0 .. x(R-1)] into its page (rows past 3 + R zero) and written (gdn_conv_reader.cpp's steps, out
        // of place: the reader packs from the same staging meanwhile). The writes finish with this kernel's barriers.
        uint32_t fi = slot0 + R - 1;
        const uint32_t H = get_arg_val<uint32_t>(fi++);
        const uint32_t q_col = get_arg_val<uint32_t>(fi++);
        const uint32_t k_col = get_arg_val<uint32_t>(fi++);
        const uint32_t v_col = get_arg_val<uint32_t>(fi++);
        const uint32_t f_x0 = get_arg_val<uint32_t>(fi++);
        const uint32_t f_ct = get_arg_val<uint32_t>(fi++);
        const uint32_t cs_addr = get_arg_val<uint32_t>(fi++);
        const uint32_t wb_addr = get_arg_val<uint32_t>(fi++);
        constexpr uint32_t tx = 2048;  // bf16 pages (gdn_verify.py: the conv's tensors are bf16)
        const auto cs_acc = TensorAccessor(cs_a, cs_addr, tx);
        const auto wb_acc = TensorAccessor(wb_a, wb_addr, tx);
        (void)wb_acc;
        const uint32_t G0 = HV / H, f_nk = nh / G0, f_nc = f_nk * 2 * Kt + nh * Vt;
        const uint32_t rb = get_write_ptr(cb_rbuf);
        const uint32_t fx = rb + GDN_FOLD_FX, fs = fx + GDN_FOLD_NCMAX * tx;  // the reader's staging
        const uint32_t ws = rb + GDN_FOLD_WS, ww = ws + GDN_FOLD_NCMAX * tx;  // this kernel's
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        {
            DeviceZoneScopedN("W-fwait");
            noc_semaphore_wait(sem, 1);
        }
        noc_semaphore_set(sem, 0);
        invalidate_l1_cache();
        DeviceZoneScopedN("W-fshift");
        for (uint32_t c = 0; c < f_nc; c++) {
            noc_async_read(get_noc_addr(fs + c * tx), ws + c * tx, tx);
        }
#if GDN_FOLD_WINDOW
        noc_zero(ww, f_nc * tx);
#endif
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t c = 0; c < f_nc; c++) {
            uint32_t b, col;
            if (c < f_nk * 2 * Kt) {
                const uint32_t bh = h0 + (c / (2 * Kt)) * G0;
                const uint32_t hk = (bh % HV) / G0, t = c % (2 * Kt);
                b = bh / HV;
                col = (t < Kt) ? (q_col + hk * Kt + t) : (k_col + hk * Kt + t - Kt);
            } else {
                const uint32_t e = c - f_nk * 2 * Kt;
                const uint32_t bh = h0 + e / Vt;
                b = bh / HV;
                col = v_col + (bh % HV) * Vt + e % Vt;
            }
            const uint32_t page = b * f_ct + col - f_x0;
#if GDN_FOLD_WINDOW
            for (uint32_t r = 0; r < 3; r++) {
                move_row(fs + c * tx, r, ww + c * tx, r);
            }
            for (uint32_t t = 0; t < R; t++) {
                move_row(fx + c * tx, t, ww + c * tx, 3 + t);
            }
#endif
            for (uint32_t r = 0; r < 3; r++) {
                const uint32_t wr = R + r;  // the window row
                if (wr < 3) {
                    move_row(fs + c * tx, wr, ws + c * tx, r);
                } else {
                    move_row(fx + c * tx, wr - 3, ws + c * tx, r);
                }
            }
            asm volatile("fence" ::: "memory");
            noc_async_write_page(page, cs_acc, ws + c * tx);
#if GDN_FOLD_WINDOW
            noc_async_write_page(page, wb_acc, ww + c * tx);
#endif
        }
    }
#endif
    for (uint32_t hh = 0; hh < nh; hh += GH) {
        const uint32_t n = (nh - hh < GH) ? (nh - hh) : GH;
        const uint32_t items = n * R;
        uint32_t wa = 0;
        if (gnorm) {
            cb_reserve_back(cb_oasm, Vt);
            wa = get_write_ptr(cb_oasm);
        }
        for (uint32_t j = 0; j < items; j++) {
            const uint32_t bh = h0 + hh + j / R;
            const uint32_t r = j % R;
            // A slot address of 0 (gated_delta_verify with slots=None): this row's state is not written.
            const uint32_t dst = (r + 1 < R) ? get_arg_val<uint32_t>(slot0 + r) : s_addr;
            const auto s_acc = TensorAccessor(s_a, dst, ts);
            {
                DeviceZoneScopedN("W-waitS");
                cb_wait_front(cb_Sout, kv);
            }
            const uint32_t rd = get_read_ptr(cb_Sout);
            if (dst != 0) {
                for (uint32_t t = 0; t < kv; t++) {
                    noc_async_write_page(bh * kv + t, s_acc, rd + t * ts);
                }
            }
            {
                DeviceZoneScopedN("W-waitO");
                cb_wait_front(gnorm ? cb_orow : cb_out, Vt);
            }
            invalidate_l1_cache();
            if (gnorm) {
                const uint32_t ro = get_read_ptr(cb_orow);
                for (uint32_t t = 0; t < Vt; t++) {
                    copy_row(ro + t * tf, 0, wa + t * tf, j, 4);
                }
                cb_pop_front(cb_orow, Vt);
            } else {
                const uint32_t ro = get_read_ptr(cb_out);
                for (uint32_t t = 0; t < Vt; t++) {
                    copy_row(ro + t * to, 0, stage + t * to, r, eo);
                }
                cb_pop_front(cb_out, Vt);
                if (r + 1 == R) {
                    const uint32_t b = bh / HV, hv = bh % HV;
                    asm volatile("fence" ::: "memory");
                    for (uint32_t t = 0; t < Vt; t++) {
                        noc_async_write_page(b * HV * Vt + hv * Vt + t, o_acc, stage + t * to);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_Sout, kv);
        }
        if (gnorm) {
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_oasm, Vt);
            {
                DeviceZoneScopedN("W-waitN");
                cb_wait_front(cb_out, Vt);
            }
            invalidate_l1_cache();
            const uint32_t ro = get_read_ptr(cb_out);
            for (uint32_t hi = 0; hi < n; hi++) {
                const uint32_t bh = h0 + hh + hi;
                const uint32_t b = bh / HV, hv = bh % HV;
                for (uint32_t r = 0; r < R; r++) {
                    for (uint32_t t = 0; t < Vt; t++) {
                        copy_row(ro + t * to, hi * R + r, stage + t * to, r, eo);
                    }
                }
                asm volatile("fence" ::: "memory");
                for (uint32_t t = 0; t < Vt; t++) {
                    noc_async_write_page(b * HV * Vt + hv * Vt + t, o_acc, stage + t * to);
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, Vt);
        }
    }
}
