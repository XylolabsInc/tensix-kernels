// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// gdn_ops.h: the tile helpers of the gated delta rule prefill kernel (gdn_compute.cpp, adapted from
// tenstorrent/tt-metal v0.81.1's chunk_gated_delta_rule, Apache-2.0), shared by the
// decode and verify kernels. The including file defines the CB names they use
// (cb_s1, cb_s2, cb_s3: scratch CBs of SCR tiles; cb_ones: a ones tile, then the norm weight;
// cb_gt: the output gate; cb_out) and the compile-time args SCR (4) and DST (6). gdn_scan.py
// inlines this file.

#pragma once

constexpr int ADD = 0, SUB = 1, MUL = 2;

ALWI void WAIT(uint32_t cb, uint32_t n) { cb_wait_front(cb, n); }
ALWI void POP(uint32_t cb, uint32_t n) { gdn::pop(cb, n); }

// The scratch CBs s1-s3 (SCR tiles each) take blocks of 1 to SCR tiles. The packer and unpacker
// address tile i of a block at the CB pointer + i pages without wrapping, so a block that started
// near the end of the CB would spill into the next CB. Each scratch block therefore starts at the
// CB's base: once consumed, a dummy push and pop of the rest brings the pointers back to it.
constexpr uint32_t SCR = get_compile_time_arg_val(4);
ALWI void DONE(uint32_t cb, uint32_t n) {
    gdn::pop(cb, n);
    const uint32_t rest = (SCR - n % SCR) % SCR;
    if (rest) {
        cb_reserve_back(cb, rest);
        cb_push_back(cb, rest);
        cb_wait_front(cb, rest);
        gdn::pop(cb, rest);
    }
}

// Each helper below fills up to DST destination tiles per acquire (1, or 4: half of Dst at fp32)
// and packs them in order.
constexpr uint32_t DST = get_compile_time_arg_val(6);

ALWI uint32_t group(uint32_t n, uint32_t t0) { return (n - t0 < DST) ? (n - t0) : DST; }

ALWI void pack_group(uint32_t m, uint32_t o) {
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < m; d++) {
        pack_tile(d, o);
    }
    tile_regs_release();
}

// o[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
__attribute__((noinline)) void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    const uint32_t n = Mt * Nt;
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(b, a);  // matmul: in0 -> srcB, in1 -> srcA
    gdn::mm_init(a, b, tr ? 1 : 0);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            const uint32_t mi = (t0 + d) / Nt, ni = (t0 + d) % Nt;
            for (uint32_t ki = 0; ki < Kt; ki++) {
                const uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, d);
            }
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// o = A (op) B elementwise over n tiles.
__attribute__((noinline)) void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary: a -> srcA, b -> srcB
    if (op == ADD) {
        gdn::add_init(a, b);
    } else if (op == SUB) {
        gdn::sub_init(a, b);
    } else {
        gdn::mul_init(a, b);
    }
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            const uint32_t i = t0 + d;
            if (op == ADD) {
                add_tiles(a, b, i, i, d);
            } else if (op == SUB) {
                sub_tiles(a, b, i, i, d);
            } else {
                mul_tiles(a, b, i, i, d);
            }
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// o = exp(in) over n tiles.
__attribute__((noinline)) void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        gdn::copy_init(in);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            copy_tile(in, t0 + d, d);
        }
        exp_tile_init();
        for (uint32_t d = 0; d < m; d++) {
            exp_tile(d);
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// o[Mt,Nt] = A[Mt,Nt] * col[Mt] (column 0 of each col tile broadcast across the row).
__attribute__((noinline)) void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    const uint32_t n = Mt * Nt;
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);
    gdn::mul_bcast_cols_init(a, col);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            mul_tiles_bcast_cols(a, col, t0 + d, (t0 + d) / Nt, d);
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// o[Mt,Nt] = A[Mt,Nt] - row[Nt] (row 0 of each row tile broadcast down the column).
__attribute__((noinline)) void bcast_rows_sub(uint32_t a, uint32_t row, uint32_t o, uint32_t Mt, uint32_t Nt) {
    const uint32_t n = Mt * Nt;
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);
    gdn::sub_bcast_rows_init(a, row);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            gdn::sub_tiles_bcast_rows(a, row, t0 + d, (t0 + d) % Nt, d);
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// o = A * s[0,0] over n tiles.
__attribute__((noinline)) void bcast_scalar_mul(uint32_t a, uint32_t s, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, s);
    gdn::mul_bcast_scalar_init(a, s);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            mul_tiles_bcast_scalar(a, s, t0 + d, 0, d);
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// The [Mt,Nt] block of `in` written as [Nt,Mt]: tile (ni,mi) of o is transpose(in[mi,ni]).
__attribute__((noinline)) void transpose_block(uint32_t in, uint32_t o, uint32_t Mt, uint32_t Nt) {
    const uint32_t n = Mt * Nt;
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    gdn::transpose_init(in);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            const uint32_t ni = (t0 + d) / Mt, mi = (t0 + d) % Mt;
            gdn::transpose_tile(in, mi * Nt + ni, d);
        }
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// o = raw / sqrt(sum_k raw^2 + eps) (* scale), per row over Kt tiles; raw stays available.
__attribute__((noinline)) void l2norm(uint32_t raw, uint32_t o, uint32_t Ct, uint32_t Kt, uint32_t eps, uint32_t scale,
                                      bool scaled) {
    const uint32_t n = Ct * Kt;
    ew(raw, raw, cb_s1, n, MUL);  // squares, fp32
    WAIT(cb_s1, n);
    // Row sums: s1 @ ones puts each row's sum in every column.
    cb_reserve_back(cb_s2, Ct);
    pack_reconfig_data_format(cb_s2);
    reconfig_data_format(cb_ones, cb_s1);
    gdn::mm_init(cb_s1, cb_ones, 0);
    for (uint32_t ct = 0; ct < Ct; ct++) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            matmul_tiles(cb_s1, cb_ones, ct * Kt + kt, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s2);
        tile_regs_release();
    }
    cb_push_back(cb_s2, Ct);
    DONE(cb_s1, n);
    WAIT(cb_s2, Ct);
    // 1 / sqrt(sum + eps), times the scale for q.
    cb_reserve_back(cb_s3, Ct);
    pack_reconfig_data_format(cb_s3);
    reconfig_data_format_srca(cb_s2);
    gdn::copy_init(cb_s2);
    for (uint32_t ct = 0; ct < Ct; ct++) {
        tile_regs_acquire();
        copy_tile(cb_s2, ct, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        if (scaled) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s3);
        tile_regs_release();
    }
    cb_push_back(cb_s3, Ct);
    DONE(cb_s2, Ct);
    WAIT(cb_s3, Ct);
    bcast_cols_mul(raw, cb_s3, o, Ct, Kt);
    DONE(cb_s3, Ct);
}

// o = raw (* scale) over n tiles (NORM = 0: q and k come normalized).
__attribute__((noinline)) void scale_copy(uint32_t raw, uint32_t o, uint32_t n, uint32_t scale, bool scaled) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(raw);
    gdn::copy_init(raw);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(raw, i, 0);
        if (scaled) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// o[Ct,Vt] (in `o`, consumed) -> cb_out = rmsnorm(o) * w * silu(gate), per token over V: upstream's
// rms_norm_gated_ttnn (ttnn.rms_norm with the weight, then times silu(gate); its clip of silu(gate)
// to 1e4 is left out). w: Vt tiles behind the ones in cb_ones (row 0 holds the weights); gate: Ct x Vt
// tiles in cb_gt.
__attribute__((noinline)) void gnorm(uint32_t o, uint32_t Ct, uint32_t Vt, uint32_t eps, uint32_t inv_v) {
    const uint32_t n = Ct * Vt, cc = Ct * Ct;
    ew(o, o, cb_s1, n, MUL);
    WAIT(cb_s1, n);
    cb_reserve_back(cb_s2, Ct);  // row sums of the squares
    pack_reconfig_data_format(cb_s2);
    reconfig_data_format(cb_ones, cb_s1);
    gdn::mm_init(cb_s1, cb_ones, 0);
    for (uint32_t ct = 0; ct < Ct; ct++) {
        tile_regs_acquire();
        for (uint32_t vt = 0; vt < Vt; vt++) {
            matmul_tiles(cb_s1, cb_ones, ct * Vt + vt, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s2);
        tile_regs_release();
    }
    cb_push_back(cb_s2, Ct);
    DONE(cb_s1, n);
    WAIT(cb_s2, Ct);
    cb_reserve_back(cb_s1, Ct);  // 1 / sqrt(mean + eps)
    pack_reconfig_data_format(cb_s1);
    reconfig_data_format_srca(cb_s2);
    gdn::copy_init(cb_s2);
    for (uint32_t ct = 0; ct < Ct; ct++) {
        tile_regs_acquire();
        copy_tile(cb_s2, ct, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_v);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s1);
        tile_regs_release();
    }
    cb_push_back(cb_s1, Ct);
    DONE(cb_s2, Ct);
    WAIT(cb_s1, Ct);
    bcast_cols_mul(o, cb_s1, cb_s2, Ct, Vt);  // o / rms
    WAIT(cb_s2, n);
    DONE(cb_s1, Ct);
    DONE(o, n);
    cb_reserve_back(cb_s1, n);  // * w
    pack_reconfig_data_format(cb_s1);
    reconfig_data_format(cb_s2, cb_ones);
    gdn::mul_bcast_rows_init(cb_s2, cb_ones);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_s2, cb_ones, i, cc + i % Vt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s1);
        tile_regs_release();
    }
    cb_push_back(cb_s1, n);
    DONE(cb_s2, n);
    WAIT(cb_s1, n);
    WAIT(cb_gt, n);
    cb_reserve_back(cb_s2, n);  // silu(gate)
    pack_reconfig_data_format(cb_s2);
    reconfig_data_format_srca(cb_gt);
    for (uint32_t i = 0; i < n; i++) {
        gdn::copy_init(cb_gt);
        tile_regs_acquire();
        copy_tile(cb_gt, i, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s2);
        tile_regs_release();
    }
    cb_push_back(cb_s2, n);
    POP(cb_gt, n);
    WAIT(cb_s2, n);
    ew(cb_s1, cb_s2, cb_out, n, MUL);
    DONE(cb_s1, n);
    DONE(cb_s2, n);
}

