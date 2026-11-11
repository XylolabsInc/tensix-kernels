// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the DeltaNet's conv kernel (gdn_conv_compute.cpp): the causal depthwise conv1d of size 4
// (taps w0..w3, the bias, SiLU) for T new tokens in order (T = 1: the decode step; T > 1: a
// speculative verify's rows), and the conv state's shift. Each core takes a run of items (batch b,
// channel tile ct: 32 channels), in groups of up to 32. Per group: the taps and the bias, packed by
// gdn_conv.py so that row j of each holds item j's 32 channels; the items' conv-state pages (rows
// 0..2: the last three tokens) and x pages (rows 0..T-1 of the fused projection output's qkv columns), read
// whole; the window [s0, s1, s2, x0 .. x(T-1)] of every item packed into row j of T + 3 tiles for the
// compute (window row r in tile r); optionally the window itself into its buffer ([B, 3 + T, C],
// for the verify's commit); then the state shifted in place in L1 (rows 0..2 <- window rows T..T+2)
// and written back to the pages it was read from (each item's pages are this core's alone). It
// replaces upstream's _causal_conv1d_decode_t1_inplace and, for T rows, the FIR conv
// _causal_conv1d_fir (tenstorrent/tt-metal v0.81.1, models/experimental/
// gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py, Apache-2.0).

#include <cstdint>
// DeviceZoneScopedN marks profiler zones and has no effect when profiling is disabled.
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in = 0;   // T + 3 tiles per group: the window, item j in row j
constexpr uint32_t cb_w = 1;    // 5 tiles per group: w0, w1, w2, w3, bias (packed)
constexpr uint32_t cb_rx = 20;  // reader-only: the group's x pages
constexpr uint32_t cb_rs = 21;  // reader-only: the group's state pages
constexpr uint32_t cb_rw = 23;  // reader-only: the group's window pages (WINDOW)
constexpr uint32_t cb_rsl = 24; // reader-only: the conv slots' staging, one tile a slot (NSLOT)

inline uint32_t tidx(uint32_t i, uint32_t j) { return ((i >> 4) << 9) + ((i & 15) << 4) + ((j >> 4) << 8) + (j & 15); }

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

// Row r of the bf16 tile at src into row q of the bf16 tile at dst (two 16-element face rows).
FORCE_INLINE void move_row(uint32_t src, uint32_t r, uint32_t dst, uint32_t q) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + tidx(r, 0) * 2);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + tidx(q, 0) * 2);
    copy_faces<8>(s, s + 128, d, d + 128);  // the second face: 256 elements = 128 words further
}

// bytes at addr zeroed by NOC reads of the core's zeros region (MEM_ZEROS_SIZE a read; bytes a
// multiple of it); the caller waits (noc_async_read_barrier) before it writes there.
__attribute__((noinline)) void noc_zero(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(z, addr + off, MEM_ZEROS_SIZE);
    }
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);       // tokens, 1..29 (T + 3 window rows)
    constexpr uint32_t WINDOW = get_compile_time_arg_val(1);  // 1: write the window buffer
// NSLOT is 0 or T - 1. Slot a ([B, 3, C], a < T - 1) holds the state after row a:
// window rows a + 1 .. a + 3. Commit copies the slot without relayout. Slot addresses
// follow the runtime args; slots share the state's accessor layout, with one base per slot.
    constexpr uint32_t NSLOT = get_compile_time_arg_val(2);
    static_assert(NSLOT == 0 || NSLOT + 1 == T, "gdn_conv: no slots or T - 1 slots");
    constexpr auto x_a = TensorAccessorArgs<3>();  // the fused projection output (x: its qkv columns)
    constexpr auto s_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();  // conv state
    constexpr auto w_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();  // packed taps
    constexpr auto v_a = TensorAccessorArgs<w_a.next_compile_time_args_offset()>();  // window buffer

    uint32_t i = 0;
    const uint32_t i0 = get_arg_val<uint32_t>(i++);      // this core's first item
    const uint32_t n = get_arg_val<uint32_t>(i++);       // its items
    const uint32_t CT = get_arg_val<uint32_t>(i++);      // channel tiles per batch row
    const uint32_t x_pitch = get_arg_val<uint32_t>(i++); // tile columns of the fused projection output
    const uint32_t x_col = get_arg_val<uint32_t>(i++);   // tile column of channel 0 in it
    const uint32_t w0 = get_arg_val<uint32_t>(i++);      // this core's first packed taps page
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t v_addr = get_arg_val<uint32_t>(i++);
    const uint32_t slot0 = i;  // NSLOT slot addresses follow

    constexpr uint32_t NW = T + 3;
    const uint32_t tx = get_tile_size(cb_rx);
    const uint32_t ts = get_tile_size(cb_rs);
    const uint32_t tw = get_tile_size(cb_w);
    const uint32_t tin = get_tile_size(cb_in);
    const auto x_acc = TensorAccessor(x_a, x_addr, tx);
    const auto s_acc = TensorAccessor(s_a, s_addr, ts);
    const auto w_acc = TensorAccessor(w_a, w_addr, tw);
    const auto v_acc = TensorAccessor(v_a, v_addr, ts);
    const uint32_t rx = get_write_ptr(cb_rx), rs = get_write_ptr(cb_rs);
    const uint32_t rw = WINDOW ? get_write_ptr(cb_rw) : 0;

// Zero unused rows of packed window tiles and rows past T + 3 in window pages with
// NOC reads of the core's zeros region. Clear only min(n, 32) window pages, the CB's
// capacity. The group loop's read barrier completes before the first write to either.
    {
        DeviceZoneScopedN("R-czero");
        noc_zero(get_write_ptr(cb_in), NW * tin);
        if constexpr (WINDOW) {
            noc_zero(rw, (n < 32 ? n : 32) * ts);
        }
        if constexpr (NSLOT > 0) {  // rows 3..31 of every slot page stay zero
            noc_zero(get_write_ptr(cb_rsl), NSLOT * ts);
        }
    }

    for (uint32_t g = 0, j0 = 0; j0 < n; g++, j0 += 32) {
        const uint32_t m = (n - j0 < 32) ? (n - j0) : 32;
        cb_reserve_back(cb_w, 5);
        const uint32_t ww = get_write_ptr(cb_w);
        {
            DeviceZoneScopedN("R-cread");
            for (uint32_t k = 0; k < 5; k++) {
                noc_async_read_page(w0 + g * 5 + k, w_acc, ww + k * tw);
            }
            for (uint32_t j = 0; j < m; j++) {
                const uint32_t it = i0 + j0 + j;
                const uint32_t b = it / CT, ct = it % CT;
                noc_async_read_page(b * x_pitch + x_col + ct, x_acc, rx + j * tx);
                noc_async_read_page(b * CT + ct, s_acc, rs + j * ts);
            }
            noc_async_read_barrier();
        }
        invalidate_l1_cache();
        cb_push_back(cb_w, 5);

        // The window of item j into row j of the T + 3 packed tiles (rows 0..2: the state; then x).
        cb_reserve_back(cb_in, NW);
        const uint32_t wi = get_write_ptr(cb_in);
        {
            DeviceZoneScopedN("R-cpack");
            for (uint32_t j = 0; j < m; j++) {
                for (uint32_t r = 0; r < 3; r++) {
                    move_row(rs + j * ts, r, wi + r * tin, j);
                }
                for (uint32_t t = 0; t < T; t++) {
                    move_row(rx + j * tx, t, wi + (3 + t) * tin, j);
                }
            }
            asm volatile("fence" ::: "memory");
        }
        cb_push_back(cb_in, NW);

        if constexpr (WINDOW) {
            DeviceZoneScopedN("R-cwin");
            for (uint32_t j = 0; j < m; j++) {
                const uint32_t it = i0 + j0 + j;
                for (uint32_t r = 0; r < 3; r++) {
                    move_row(rs + j * ts, r, rw + j * ts, r);
                }
                for (uint32_t t = 0; t < T; t++) {
                    move_row(rx + j * tx, t, rw + j * ts, 3 + t);
                }
                asm volatile("fence" ::: "memory");
                noc_async_write_page((it / CT) * CT + it % CT, v_acc, rw + j * ts);
            }
        }

        if constexpr (NSLOT > 0) {
            // The conv state after row a: window rows a + 1, a + 2, a + 3 (state rows while < 3, then x's rows) into
            // rows 0..2 of slot a's staging tile, written to slot a's page of the item; each staging tile is reused for
            // the next item once its write has left (writes flushed).
            DeviceZoneScopedN("R-cslot");
            const uint32_t sl = get_write_ptr(cb_rsl);
            for (uint32_t j = 0; j < m; j++) {
                const uint32_t it = i0 + j0 + j;
                noc_async_writes_flushed();
                for (uint32_t a = 0; a < NSLOT; a++) {
                    for (uint32_t r = 0; r < 3; r++) {
                        const uint32_t w = a + 1 + r;  // the window row
                        if (w < 3) {
                            move_row(rs + j * ts, w, sl + a * ts, r);
                        } else {
                            move_row(rx + j * tx, w - 3, sl + a * ts, r);
                        }
                    }
                }
                asm volatile("fence" ::: "memory");
                for (uint32_t a = 0; a < NSLOT; a++) {
                    const auto sl_acc = TensorAccessor(s_a, get_arg_val<uint32_t>(slot0 + a), ts);
                    noc_async_write_page((it / CT) * CT + it % CT, sl_acc, sl + a * ts);
                }
            }
        }

        // The shift, in place: rows 0..2 <- window rows T, T + 1, T + 2.
        DeviceZoneScopedN("R-cshift");
        for (uint32_t j = 0; j < m; j++) {
            for (uint32_t r = 0; r < 3; r++) {
                const uint32_t w = T + r;  // the window row
                if (w < 3) {
                    move_row(rs + j * ts, w, rs + j * ts, r);
                } else {
                    move_row(rx + j * tx, w - 3, rs + j * ts, r);
                }
            }
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t j = 0; j < m; j++) {
            const uint32_t it = i0 + j0 + j;
            noc_async_write_page((it / CT) * CT + it % CT, s_acc, rs + j * ts);
        }
        noc_async_write_barrier();
    }
}
