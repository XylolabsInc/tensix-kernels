// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the gated delta rule kernel (gdn_compute.cpp): the constants once, then per head its
// initial state and per chunk its q, k, v, g and beta tiles. Adapted from tenstorrent/tt-metal
// v0.81.1 (Apache-2.0) chunk_gated_delta_rule/device/kernels/dataflow/reader_chunk_gated_delta_rule.cpp
// and chunk_gdn_gate_read.hpp. Uses cb_* calls, TensorAccessor, and noc_async_read_page.
//
// Layouts (all TILE, interleaved; page = one tile, row-major over the tile grid):
//   q, k  flat [B, T, W] (flat = 1): head hk's K columns start at tile column col0 + hk*Kt of a
//         tile row of `pitch` tiles (W = 32*pitch; the fused [q | k | v] conv output works too), or
//         head-major [B*H, T, K] (flat = 0)
//   v     the same with HV heads of V columns
//   g, beta [B, T, HV], fp32 or bf16: tile (b, rt, hv/32) holds head hv in column hv % 32; or (raw
//   gates) the a and b columns of the fused projection output, head hv at column col0 + hv
//   consts [3, C, C] fp32: eye, tril, ones
//   state [B*HV, K, V] (any leading shape with the same tile order), or none (zeros)
//   with GNORM: the gate flat [B, T, W] (head hv's V columns from gate_col + hv*Vt, as v), and the
//   norm weight [1, 1, V] bf16 or fp32 (Vt tiles, row 0), widened to fp32 into cb_ones behind the
//   ones
// Value head bh = b*HV + hv reads key head hk = hv / G (G = HV / H; upstream's repeat_interleave).
// Rows past T are zero, except raw a and b use -1e4 so their computed gates become zero.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_qr = 0, cb_kr = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7, cb_S0 = 8;
constexpr uint32_t cb_gt = 10;  // the output gate (GNORM)

// Element (i, j) of a 32x32 tile stored as four 16x16 faces.
inline uint32_t tidx(uint32_t i, uint32_t j) { return ((i >> 4) << 9) + ((i & 15) << 4) + ((j >> 4) << 8) + (j & 15); }

// Zero rows r0..31 of the tile at addr (elem: 2 or 4 bytes).
__attribute__((noinline)) void zero_rows(uint32_t addr, uint32_t r0, uint32_t elem) {
    if (elem == 2) {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = r0; i < 32; i++) {
            for (uint32_t j = 0; j < 32; j++) {
                p[tidx(i, j)] = 0;
            }
        }
    } else {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = r0; i < 32; i++) {
            for (uint32_t j = 0; j < 32; j++) {
                p[tidx(i, j)] = 0;
            }
        }
    }
}

__attribute__((noinline)) void zero_bytes(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        p[i] = 0;
    }
}

// After the reads landed: column `col` of each of n gate pages -> column 0 as fp32, every other
// element zero, rows of tokens at or past T set to `pad` (fp32 bits: 0 for g and beta; -1e4 for raw
// a and b, which the compute's softplus and sigmoid then turn into 0). A bf16 source tile sits in
// the first half of the fp32 page.
__attribute__((noinline)) void settle_gate(uint32_t addr, uint32_t n, uint32_t col, uint32_t page_bytes, uint32_t src_bytes,
                                           uint32_t T, uint32_t tok0, uint32_t pad) {
    const bool bf16 = src_bytes < page_bytes;
    for (uint32_t r = 0; r < n; r++) {
        volatile tt_l1_ptr uint32_t* w32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr + r * page_bytes);
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr + r * page_bytes);
        uint32_t v[32];
        for (uint32_t i = 0; i < 32; i++) {
            const uint32_t tok = tok0 + r * 32 + i;
            const uint32_t idx = tidx(i, col);
            v[i] = (tok < T) ? (bf16 ? (static_cast<uint32_t>(w16[idx]) << 16) : w32[idx]) : pad;
        }
        for (uint32_t e = 0; e < 1024; e++) {
            w32[e] = 0;
        }
        for (uint32_t i = 0; i < 32; i++) {
            w32[tidx(i, 0)] = v[i];
        }
    }
}

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr auto q_a = TensorAccessorArgs<3>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    constexpr auto gt_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<gt_a.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t h0 = get_arg_val<uint32_t>(i++);
    const uint32_t nh = get_arg_val<uint32_t>(i++);
    const uint32_t NC = get_arg_val<uint32_t>(i++);
    const uint32_t T = get_arg_val<uint32_t>(i++);
    const uint32_t Tt = get_arg_val<uint32_t>(i++);  // tile rows of q/k/v/g/beta: (T + 31) / 32
    const uint32_t H = get_arg_val<uint32_t>(i++);
    const uint32_t HV = get_arg_val<uint32_t>(i++);
    const uint32_t flat = get_arg_val<uint32_t>(i++);
    const uint32_t q_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t k_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t v_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t q_col = get_arg_val<uint32_t>(i++);
    const uint32_t k_col = get_arg_val<uint32_t>(i++);
    const uint32_t v_col = get_arg_val<uint32_t>(i++);
    // g and beta: head hv's value in column (col0 + hv) of a tile row of `pitch` tiles: [B, T, HV]
    // (col0 0) or the raw a and b columns of the fused projection's output.
    const uint32_t g_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t g_col = get_arg_val<uint32_t>(i++);
    const uint32_t b_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t b_col = get_arg_val<uint32_t>(i++);
    const uint32_t pad = get_arg_val<uint32_t>(i++);  // fp32 bits for rows past T
    const uint32_t g_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t b_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t q_addr = get_arg_val<uint32_t>(i++);
    const uint32_t k_addr = get_arg_val<uint32_t>(i++);
    const uint32_t v_addr = get_arg_val<uint32_t>(i++);
    const uint32_t g_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_arg_val<uint32_t>(i++);
    const uint32_t c_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t has_state = get_arg_val<uint32_t>(i++);
    const uint32_t gnorm = get_arg_val<uint32_t>(i++);
    const uint32_t gt_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gt_pitch = get_arg_val<uint32_t>(i++);
    const uint32_t gt_col = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_bytes = get_arg_val<uint32_t>(i++);  // the norm weight's tile: 2048 or 4096

    constexpr uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt;
    const uint32_t G = HV / H;

    const uint32_t tq = get_tile_size(cb_qr);
    const uint32_t tk = get_tile_size(cb_kr);
    const uint32_t tv = get_tile_size(cb_v);
    const uint32_t tf = get_tile_size(cb_eye);  // fp32
    const uint32_t ts = get_tile_size(cb_S0);
    const uint32_t tg = get_tile_size(cb_g);  // fp32 page; the source may be bf16
    const auto q_acc = TensorAccessor(q_a, q_addr, tq);
    const auto k_acc = TensorAccessor(k_a, k_addr, tk);
    const auto v_acc = TensorAccessor(v_a, v_addr, tv);
    const auto g_acc = TensorAccessor(g_a, g_addr, g_bytes);
    const auto b_acc = TensorAccessor(b_a, b_addr, b_bytes);
    const auto c_acc = TensorAccessor(c_a, c_addr, tf);
    const auto s_acc = TensorAccessor(s_a, s_addr, ts);
    const uint32_t tgt = get_tile_size(cb_gt);
    const auto gt_acc = TensorAccessor(gt_a, gt_addr, tgt);
    const auto w_acc = TensorAccessor(w_a, w_addr, w_bytes ? w_bytes : tf);

    // Constants, once.
    const uint32_t const_cbs[3] = {cb_eye, cb_tril, cb_ones};
    for (uint32_t m = 0; m < 3; m++) {
        cb_reserve_back(const_cbs[m], cc);
        const uint32_t w = get_write_ptr(const_cbs[m]);
        for (uint32_t t = 0; t < cc; t++) {
            noc_async_read_page(m * cc + t, c_acc, w + t * tf);
        }
        noc_async_read_barrier();
        cb_push_back(const_cbs[m], cc);
    }
    if (gnorm) {
        // The norm weight's Vt tiles (bf16 or fp32, w_bytes each) into fp32 pages: row 0 widened
        // to fp32, everything else zero (the row broadcast reads row 0 only).
        cb_reserve_back(cb_ones, Vt);
        const uint32_t w = get_write_ptr(cb_ones);
        for (uint32_t t = 0; t < Vt; t++) {
            noc_async_read_page(t, w_acc, w + t * tf);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t t = 0; t < Vt; t++) {
            volatile tt_l1_ptr uint32_t* w32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + t * tf);
            volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w + t * tf);
            uint32_t row[32];
            for (uint32_t j = 0; j < 32; j++) {
                row[j] = (w_bytes < tf) ? (static_cast<uint32_t>(w16[tidx(0, j)]) << 16) : w32[tidx(0, j)];
            }
            for (uint32_t e = 0; e < 1024; e++) {
                w32[e] = 0;
            }
            for (uint32_t j = 0; j < 32; j++) {
                w32[tidx(0, j)] = row[j];
            }
        }
        asm volatile("fence" ::: "memory");
        cb_push_back(cb_ones, Vt);
    }

    const uint32_t tail = T % 32;  // valid rows of the last tile row (0: all 32)

    for (uint32_t hh = 0; hh < nh; hh++) {
        const uint32_t bh = h0 + hh;
        const uint32_t b = bh / HV;
        const uint32_t hv = bh % HV;
        const uint32_t hk = hv / G;
        // Page of tile (row rt, column t) of this head's block: base + rt * row + t.
        const uint32_t q_row = flat ? q_pitch : Kt;
        const uint32_t q_base = flat ? (b * Tt * q_pitch + q_col + hk * Kt) : ((b * H + hk) * Tt * Kt);
        const uint32_t k_row = flat ? k_pitch : Kt;
        const uint32_t k_base = flat ? (b * Tt * k_pitch + k_col + hk * Kt) : ((b * H + hk) * Tt * Kt);
        const uint32_t v_row = flat ? v_pitch : Vt;
        const uint32_t v_base = flat ? (b * Tt * v_pitch + v_col + hv * Vt) : (bh * Tt * Vt);
        const uint32_t gt_base = b * Tt * gt_pitch + gt_col + hv * Vt;

        // Initial state.
        cb_reserve_back(cb_S0, kv);
        {
            const uint32_t w = get_write_ptr(cb_S0);
            if (has_state) {
                for (uint32_t t = 0; t < kv; t++) {
                    noc_async_read_page(bh * kv + t, s_acc, w + t * ts);
                }
                noc_async_read_barrier();
            } else {
                zero_bytes(w, kv * ts);
            }
        }
        cb_push_back(cb_S0, kv);

        for (uint32_t c = 0; c < NC; c++) {
            cb_reserve_back(cb_qr, ck);
            cb_reserve_back(cb_kr, ck);
            cb_reserve_back(cb_v, cv);
            cb_reserve_back(cb_g, Ct);
            cb_reserve_back(cb_beta, Ct);
            if (gnorm) {
                cb_reserve_back(cb_gt, cv);
            }
            const uint32_t wgt = gnorm ? get_write_ptr(cb_gt) : 0;
            const uint32_t wq = get_write_ptr(cb_qr);
            const uint32_t wk = get_write_ptr(cb_kr);
            const uint32_t wv = get_write_ptr(cb_v);
            const uint32_t wg = get_write_ptr(cb_g);
            const uint32_t wb = get_write_ptr(cb_beta);
            for (uint32_t r = 0; r < Ct; r++) {
                const uint32_t rt = c * Ct + r;
                if (rt < Tt) {
                    for (uint32_t t = 0; t < Kt; t++) {
                        noc_async_read_page(q_base + rt * q_row + t, q_acc, wq + (r * Kt + t) * tq);
                        noc_async_read_page(k_base + rt * k_row + t, k_acc, wk + (r * Kt + t) * tk);
                    }
                    for (uint32_t t = 0; t < Vt; t++) {
                        noc_async_read_page(v_base + rt * v_row + t, v_acc, wv + (r * Vt + t) * tv);
                    }
                    if (gnorm) {
                        for (uint32_t t = 0; t < Vt; t++) {
                            noc_async_read_page(gt_base + rt * gt_pitch + t, gt_acc, wgt + (r * Vt + t) * tgt);
                        }
                    }
                    noc_async_read_page((b * Tt + rt) * g_pitch + (g_col + hv) / 32, g_acc, wg + r * tg);
                    noc_async_read_page((b * Tt + rt) * b_pitch + (b_col + hv) / 32, b_acc, wb + r * tg);
                }
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t r = 0; r < Ct; r++) {
                const uint32_t rt = c * Ct + r;
                if (rt >= Tt) {
                    zero_bytes(wq + r * Kt * tq, Kt * tq);
                    zero_bytes(wk + r * Kt * tk, Kt * tk);
                    zero_bytes(wv + r * Vt * tv, Vt * tv);
                    if (gnorm) {
                        zero_bytes(wgt + r * Vt * tgt, Vt * tgt);
                    }
                } else if (rt == Tt - 1 && tail != 0) {
                    for (uint32_t t = 0; gnorm && t < Vt; t++) {
                        zero_rows(wgt + (r * Vt + t) * tgt, tail, tgt / 1024);
                    }
                    for (uint32_t t = 0; t < Kt; t++) {
                        zero_rows(wq + (r * Kt + t) * tq, tail, tq / 1024);
                        zero_rows(wk + (r * Kt + t) * tk, tail, tk / 1024);
                    }
                    for (uint32_t t = 0; t < Vt; t++) {
                        zero_rows(wv + (r * Vt + t) * tv, tail, tv / 1024);
                    }
                }
            }
            settle_gate(wg, Ct, (g_col + hv) % 32, tg, g_bytes, T, c * Ct * 32, pad);
            settle_gate(wb, Ct, (b_col + hv) % 32, tg, b_bytes, T, c * Ct * 32, pad);
            asm volatile("fence" ::: "memory");
            cb_push_back(cb_qr, ck);
            cb_push_back(cb_kr, ck);
            cb_push_back(cb_v, cv);
            cb_push_back(cb_g, Ct);
            cb_push_back(cb_beta, Ct);
            if (gnorm) {
                cb_push_back(cb_gt, cv);
            }
        }
    }
}
