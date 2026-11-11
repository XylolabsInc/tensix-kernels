// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet prefill: the whole chunked gated delta rule for a list of heads on one Tensix
// core, sequential over chunks, the recurrent state S [K, V] held in L1 in fp32.
//
// Adapted from tenstorrent/tt-metal v0.81.1 (Apache-2.0)
// ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/
// chunk_gated_delta_rule.cpp, the "mono" program of upstream's chunk_gated_delta_rule op (derived
// there from flash-linear-attention's naive_chunk_gated_delta_rule). Changes here: several heads per
// core, q/k L2 normalization and q scaling in the kernel, and fp32 or bf16 state CBs.
// gdn_compat.h supplies compute helpers; this file's reader and writer share its per-head
// loop and CB plan.
//
// Per chunk (C tokens; Ct = C/32, Kt = K/32, Vt = V/32 tiles):
//   q = q / |q| * scale, k = k / |k|          (per token, over K; NORM = 1)
//   v_beta = v * beta, k_beta = k * beta
//   decay = cumsum(g) = tril @ g, decay_exp = exp(decay)
//   L_mask = tril(exp(decay_i - decay_j))
//   N = strictly_lower(k_beta @ k^T * L_mask), T_inv = (I + N)^-1 (Horner: R = I - N R, C-2 steps,
//       or per 32 x 32 diagonal block and block forward substitution: BLOCK_INV)
//   u = T_inv @ v_beta, w = T_inv @ (k_beta * decay_exp)
//   intra = (q @ k^T) * L_mask, q_decay = q * decay_exp, k_dec_t = (k * exp(g_sum - decay))^T
//   v_new = u - w @ S, o = q_decay @ S + intra @ v_new
//   S = S * exp(g_sum) + k_dec_t @ v_new
//
// Compile-time args: Ct, Kt, Vt, NORM, SCR (tiles of each scratch CB), BLOCK_INV (the WY
// inverse by 32 x 32 diagonal blocks instead of the whole chunk), DST (tiles per acquire: 1 or 4),
// GNORM (the output through upstream's gated RMSNorm, gnorm()), GATES (g and beta from the raw a
// and b columns, raw_gates()), F32S (below).
//
// F32S (compile-time arg 9) updates the state in Dst at fp32. With F32S 0, the FPU
// truncates fp32 state to tf32 for both decay and addition. It also truncates exp(g_sum),
// formed from decay_fac[0] * decay_exp[0]. Repeated truncation can accumulate in the state.
// With F32S 1, cb_sa and an fp32 cb_S0 unpack directly to Dst as stored. The matmuls
// w @ S and q_decay @ S read an FPU-readable copy packed to s3, at tf32; these feed
// v_new and o once. g_sum = ones @ g stays in cb_decay. Each pass updates DST_F32 - 1
// state tiles: form exp(g_sum) in Dst 0 from ones * g_sum[0, 0], apply exp on one SFPU
// row per face, unpack the state to Dst, multiply by decay on the SFPU, then accumulate
// k_dec_t @ v_new with the FPU matmul (Dst += srcA srcB). The host's INP_FLOAT32
// define selects fp32 softplus for raw gates.
// Runtime args: heads on this core, chunks per head, q scale, L2 eps, the gated norm's eps and 1 / V,
// then per head (GATES) -exp(A_log) and dt_bias (fp32 bits).

// O2 limits kernel text size. A program larger than the device's kernel config ring
// fails at enqueue; a smaller worker L1 gives a larger ring.
// Os can break the LLK's immediate-operand assembly.
#pragma GCC optimize("O2")

#include <cstdint>
#include "gdn_compat.h"
#include "api/compute/eltwise_binary_sfpu.h"

namespace {

// CB plan (must match gdn_scan.py). The reader produces 0-8, the writer consumes 26-27.
constexpr uint32_t cb_qr = 0, cb_kr = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7, cb_S0 = 8;
constexpr uint32_t cb_sa = 9;  // the state between chunks (compute-only; popped before the next push)
constexpr uint32_t cb_gt = 10;  // the output gate (GNORM), from the reader
constexpr uint32_t cb_q = 11, cb_k = 12;
constexpr uint32_t cb_decay = 13, cb_dexp = 14, cb_dfac = 15;
constexpr uint32_t cb_lmask = 16, cb_tinv = 17, cb_vbeta = 18, cb_kbeta = 19;
constexpr uint32_t cb_u = 20, cb_w = 21, cb_qdec = 22, cb_intra = 23;
constexpr uint32_t cb_vnew = 24, cb_kdect = 25, cb_out = 26, cb_final = 27;
constexpr uint32_t cb_s1 = 28, cb_s2 = 29, cb_s3 = 30;
constexpr uint32_t cb_h = 31;  // the Horner step's product, always Ct x Ct

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
constexpr uint32_t F32S = get_compile_time_arg_val(9);  // the state's update at fp32 (see the head of this file)
static_assert(F32S <= 1, "F32S: 0 or 1");
// F32S's pass: exp(g_sum) in Dst 0 and DST_F32 - 1 state tiles an acquire (half of Dst at fp32)
constexpr uint32_t DST_F32 = 4;

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

// Single-tile ops into a CB the caller reserved and pushes (out-of-line, so that the block
// inverse's mix of copies, adds and matmuls costs little kernel text).
__attribute__((noinline)) void pack_copy(uint32_t in, uint32_t itile, uint32_t o) {
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    gdn::copy_init(in);
    tile_regs_acquire();
    copy_tile(in, itile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
}

__attribute__((noinline)) void pack_add(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o) {
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    gdn::add_init(a, b);
    tile_regs_acquire();
    add_tiles(a, b, ia, ib, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
}

// F32S: copy n tiles from in to o with one pack_copy per tile, reusing the block-inverse
// helper to limit kernel text. Unpack-to-dest inputs arrive as stored; in stays available.
ALWI void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        cb_reserve_back(o, 1);
        pack_copy(in, i, o);
        cb_push_back(o, 1);
    }
}

// F32S decay multiply: every element of the decay tile has the same value. Apply exp
// to one SFPU row per face, then load that row once per face for multiplication.
#ifdef TRISC_MATH
template <int ITER>
inline void mul_by_dec_face(const uint dst_in0, const uint dst_in1, const uint dst_out) {
    constexpr uint dst_tile_size_sfpi = 32;  // SFPI address stride per Dst tile
    const sfpi::vFloat dv = sfpi::dst_reg[dst_in1 * dst_tile_size_sfpi];
    for (int d = 0; d < ITER; d++) {
        sfpi::dst_reg[dst_out * dst_tile_size_sfpi] = sfpi::dst_reg[dst_in0 * dst_tile_size_sfpi] * dv;
        sfpi::dst_reg++;
    }
}
#endif

// The math thread waits for the SFPU to finish (an FPU op that writes its Dst tiles comes next).
ALWI void wait_sfpu() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(ckernel::p_stall::STALL_CFG | ckernel::p_stall::STALL_MATH, ckernel::p_stall::WAIT_SFPU);
#endif
}

// F32S: S' = exp(g_sum) S + k_dec_t @ v_new in Dst at fp32, into nxt_S. cur_S (unpack-to-dest) is popped as it is
// read and nxt_S reserved after it, so that cur_S may be nxt_S (cb_sa between chunks). g_sum sits at [0, 0] of
// cb_decay's first tile (kept there by chunk()); k_dec_t [Kt, Ct] and v_new [Ct, Vt] are read, not popped.
__attribute__((noinline)) void snew_f32(uint32_t cur_S, uint32_t nxt_S, uint32_t Ct, uint32_t Vt, uint32_t kv) {
    constexpr uint32_t SD = DST_F32 - 1;
    pack_reconfig_data_format(nxt_S);  // fp32 between chunks; the state's format into cb_final
    for (uint32_t t0 = 0; t0 < kv; t0 += SD) {
        const uint32_t m = (kv - t0 < SD) ? (kv - t0) : SD;
        tile_regs_acquire();
        reconfig_data_format(cb_ones, cb_decay);
        gdn::mul_bcast_scalar_init(cb_ones, cb_decay);
        mul_tiles_bcast_scalar(cb_ones, cb_decay, 0, 0, 0);  // g_sum everywhere (its tf32 read is relative to g_sum)
        exp_tile_init();
        gdn::exp_first(0);
        reconfig_data_format_srca(cur_S);
        gdn::copy_init(cur_S);
        for (uint32_t d = 0; d < m; d++) {
            copy_tile(cur_S, d, 1 + d);  // the state as stored
        }
        POP(cur_S, m);
        mul_binary_tile_init();
        for (uint32_t d = 0; d < m; d++) {
            MATH((GDN_BPARAMS((mul_by_dec_face<8>), 1 + d, 0, 1 + d, RC)));  // S_dec, fp32
        }
        wait_sfpu();  // the matmul accumulates onto the SFPU's tiles
        reconfig_data_format(cb_vnew, cb_kdect);  // matmul: in0 -> srcB, in1 -> srcA
        gdn::mm_init(cb_kdect, cb_vnew, 0);
        for (uint32_t d = 0; d < m; d++) {
            const uint32_t ki = (t0 + d) / Vt, vi = (t0 + d) % Vt;
            for (uint32_t c = 0; c < Ct; c++) {
                matmul_tiles(cb_kdect, cb_vnew, ki * Ct + c, c * Vt + vi, 1 + d);  // += k_dec_t @ v_new
            }
        }
        cb_reserve_back(nxt_S, m);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; d++) {
            pack_tile(1 + d, nxt_S);
        }
        tile_regs_release();
        cb_push_back(nxt_S, m);
    }
}

// o = sum_{k < n} A[a0 + k * da] @ B[b0 + k * db].
__attribute__((noinline)) void pack_mm(
    uint32_t a, uint32_t a0, uint32_t da, uint32_t b, uint32_t b0, uint32_t db, uint32_t n, uint32_t o) {
    pack_reconfig_data_format(o);
    reconfig_data_format(b, a);
    gdn::mm_init(a, b, 0);
    tile_regs_acquire();
    for (uint32_t k = 0; k < n; k++) {
        matmul_tiles(a, b, a0 + k * da, b0 + k * db, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
}

// T_inv = (I - negN)^-1 for negN [Ct, Ct] in s3 (strictly lower), into cb_tinv row by row.
// Diagonal blocks: Horner on one tile, R = I + negN_bb R, 30 steps from R_1 (exact: a strictly
// lower 32 x 32 block is nilpotent of order 32), in cb_h used as a 2-deep FIFO (the newest R or
// product behind the one being read); the Ct inverses go to s1. Below the diagonal, by block
// forward substitution: T[i][j] = D_i sum_{k=j}^{i-1} negN[i][k] T[k][j].
__attribute__((noinline)) void block_inverse(uint32_t Ct) {
    for (uint32_t b = 0; b < Ct; b++) {
        const uint32_t nb = b * Ct + b;
        cb_reserve_back(cb_h, 1);
        pack_add(cb_eye, 0, cb_s3, nb, cb_h);  // R_1 = I + negN_bb
        cb_push_back(cb_h, 1);
        for (uint32_t m = 2; m < 32; m++) {
            WAIT(cb_h, 1);
            cb_reserve_back(cb_h, 1);
            pack_mm(cb_s3, nb, 0, cb_h, 0, 0, 1, cb_h);  // negN_bb @ R
            cb_push_back(cb_h, 1);
            POP(cb_h, 1);
            WAIT(cb_h, 1);
            cb_reserve_back(cb_h, 1);
            pack_add(cb_eye, 0, cb_h, 0, cb_h);  // R = I + negN_bb @ R
            cb_push_back(cb_h, 1);
            POP(cb_h, 1);
        }
        WAIT(cb_h, 1);
        cb_reserve_back(cb_s1, 1);
        pack_copy(cb_h, 0, cb_s1);
        cb_push_back(cb_s1, 1);
        POP(cb_h, 1);
    }
    WAIT(cb_s1, Ct);
    for (uint32_t i = 0; i < Ct; i++) {
        if (i > 0) {
            WAIT(cb_tinv, i * Ct);
        }
        cb_reserve_back(cb_tinv, Ct);
        for (uint32_t j = 0; j < Ct; j++) {
            if (j > i) {
                pack_copy(cb_eye, i * Ct + j, cb_tinv);  // an off-diagonal tile of eye: zeros
            } else if (j == i) {
                pack_copy(cb_s1, i, cb_tinv);
            } else {
                cb_reserve_back(cb_h, 1);
                pack_mm(cb_s3, i * Ct + j, 1, cb_tinv, j * Ct + j, Ct, i - j, cb_h);
                cb_push_back(cb_h, 1);
                WAIT(cb_h, 1);
                pack_mm(cb_s1, i, 0, cb_h, 0, 0, 1, cb_tinv);
                POP(cb_h, 1);
            }
        }
        cb_push_back(cb_tinv, Ct);
    }
    DONE(cb_s1, Ct);
}

// Raw gates (GATES): from the a and b columns of the fused projection (column 0 of the reader's Ct
// tiles; rows past T hold -1e4), g = A_neg * softplus(a + dt_bias) into cb_dfac and beta = sigmoid(b)
// into cb_dexp, upstream's beta and g. Both CBs are free here: dexp and dfac are made later in the
// chunk, after these are consumed, with the same block size.
__attribute__((noinline)) void raw_gates(uint32_t Ct, uint32_t a_neg, uint32_t dt) {
    constexpr uint32_t ONE = 0x3f800000u, THRESHOLD = 0x41a00000u;  // 1.0, 20.0 (torch's softplus)
    cb_reserve_back(cb_dfac, Ct);
    pack_reconfig_data_format(cb_dfac);
    reconfig_data_format_srca(cb_g);
    for (uint32_t i = 0; i < Ct; i++) {
        gdn::copy_init(cb_g);
        tile_regs_acquire();
        copy_tile(cb_g, i, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, dt);
        softplus_tile_init();
        softplus_tile(0, ONE, ONE, THRESHOLD);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, a_neg);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_dfac);
        tile_regs_release();
    }
    cb_push_back(cb_dfac, Ct);
    cb_reserve_back(cb_dexp, Ct);
    pack_reconfig_data_format(cb_dexp);
    reconfig_data_format_srca(cb_beta);
    for (uint32_t i = 0; i < Ct; i++) {
        gdn::copy_init(cb_beta);
        tile_regs_acquire();
        copy_tile(cb_beta, i, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_dexp);
        tile_regs_release();
    }
    cb_push_back(cb_dexp, Ct);
    POP(cb_g, Ct);
    POP(cb_beta, Ct);
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

// One chunk of one head: reads the state from cur_S, writes the next state to nxt_S.
template <bool norm, bool block, bool gated, bool raw>
__attribute__((noinline)) void chunk(
    uint32_t cur_S,
    uint32_t nxt_S,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t scale,
    uint32_t eps,
    uint32_t norm_eps,
    uint32_t inv_v,
    uint32_t a_neg,
    uint32_t dt) {
    // With raw gates, g and beta are made in cb_dfac / cb_dexp from the reader's raw columns.
    const uint32_t cb_gg = raw ? cb_dfac : cb_g;
    const uint32_t cb_bb = raw ? cb_dexp : cb_beta;
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    const uint32_t C = Ct * 32;

    WAIT(cb_qr, ck);
    WAIT(cb_kr, ck);
    WAIT(cb_v, cv);
    WAIT(cb_g, Ct);
    WAIT(cb_beta, Ct);
    if constexpr (raw) {
        raw_gates(Ct, a_neg, dt);
        WAIT(cb_gg, Ct);
        WAIT(cb_bb, Ct);
    }

    // ---- q, k: L2 norm (q also scaled), fp32 ----
    if constexpr (norm) {
        l2norm(cb_qr, cb_q, Ct, Kt, eps, scale, true);
        l2norm(cb_kr, cb_k, Ct, Kt, eps, scale, false);
    } else {
        scale_copy(cb_qr, cb_q, ck, scale, true);
        scale_copy(cb_kr, cb_k, ck, scale, false);
    }
    WAIT(cb_q, ck);
    WAIT(cb_k, ck);
    POP(cb_qr, ck);
    POP(cb_kr, ck);

    // ---- v_beta, k_beta ----
    bcast_cols_mul(cb_v, cb_bb, cb_vbeta, Ct, Vt);
    WAIT(cb_vbeta, cv);
    bcast_cols_mul(cb_k, cb_bb, cb_kbeta, Ct, Kt);
    WAIT(cb_kbeta, ck);
    POP(cb_bb, Ct);
    POP(cb_v, cv);

    // ---- decay = tril @ g, decay_exp, decay_row (in s1) ----
    mm(cb_tril, cb_gg, cb_decay, Ct, Ct, 1, false);
    WAIT(cb_decay, Ct);
    expc(cb_decay, cb_dexp, Ct);
    WAIT(cb_dexp, Ct);
    transpose_block(cb_decay, cb_s1, Ct, 1);
    WAIT(cb_s1, Ct);

    // ---- L_mask = tril(exp(decay_i - decay_j)) ----
    bcast_cols_mul(cb_ones, cb_decay, cb_s2, Ct, Ct);  // decay_i in every column
    WAIT(cb_s2, cc);
    bcast_rows_sub(cb_s2, cb_s1, cb_s3, Ct, Ct);  // decay_i - decay_j
    WAIT(cb_s3, cc);
    DONE(cb_s1, Ct);
    DONE(cb_s2, cc);
    ew(cb_s3, cb_tril, cb_s2, cc, MUL);  // zero above the diagonal before the exp
    WAIT(cb_s2, cc);
    DONE(cb_s3, cc);
    expc(cb_s2, cb_s3, cc);
    WAIT(cb_s3, cc);
    DONE(cb_s2, cc);
    ew(cb_s3, cb_tril, cb_lmask, cc, MUL);
    WAIT(cb_lmask, cc);
    DONE(cb_s3, cc);

    // ---- decay_fac = exp(g_sum - decay) ----
    mm(cb_ones, cb_gg, cb_s1, Ct, Ct, 1, false);  // g_sum in every row
    WAIT(cb_s1, Ct);
    POP(cb_gg, Ct);
    ew(cb_s1, cb_decay, cb_s2, Ct, SUB);
    WAIT(cb_s2, Ct);
    if constexpr (F32S) {
        POP(cb_decay, Ct);
        copy_tiles(cb_s1, cb_decay, Ct);  // g_sum kept for the update (cb_decay is free until the next chunk)
        DONE(cb_s1, Ct);
    } else {
        DONE(cb_s1, Ct);
        POP(cb_decay, Ct);
    }
    expc(cb_s2, cb_dfac, Ct);
    WAIT(cb_dfac, Ct);
    DONE(cb_s2, Ct);

    // ---- negN = -strictly_lower(k_beta @ k^T * L_mask); T_inv = sum_i negN^i (Horner) ----
    mm(cb_kbeta, cb_k, cb_s1, Ct, Kt, Ct, true);
    WAIT(cb_s1, cc);
    ew(cb_s1, cb_lmask, cb_s2, cc, MUL);
    WAIT(cb_s2, cc);
    DONE(cb_s1, cc);
    ew(cb_s2, cb_eye, cb_s1, cc, MUL);  // its diagonal
    WAIT(cb_s1, cc);
    ew(cb_s1, cb_s2, cb_s3, cc, SUB);  // negN, kept in s3
    WAIT(cb_s3, cc);
    DONE(cb_s1, cc);
    DONE(cb_s2, cc);
    if constexpr (block) {
        block_inverse(Ct);
    } else {
        // The whole C x C matrix at once (upstream's mono kernel): C - 2 steps of a Ct x Ct matmul.
        ew(cb_eye, cb_s3, cb_tinv, cc, ADD);  // R_1 = I + negN
        WAIT(cb_tinv, cc);
        for (uint32_t m = 2; m < C; m++) {
            mm(cb_s3, cb_tinv, cb_h, Ct, Ct, Ct, false);
            WAIT(cb_h, cc);
            POP(cb_tinv, cc);
            ew(cb_eye, cb_h, cb_tinv, cc, ADD);
            WAIT(cb_tinv, cc);
            POP(cb_h, cc);
        }
    }
    WAIT(cb_tinv, cc);
    DONE(cb_s3, cc);

    // ---- u = T_inv @ v_beta, w = T_inv @ (k_beta * decay_exp) ----
    mm(cb_tinv, cb_vbeta, cb_u, Ct, Ct, Vt, false);
    WAIT(cb_u, cv);
    POP(cb_vbeta, cv);
    bcast_cols_mul(cb_kbeta, cb_dexp, cb_s1, Ct, Kt);
    WAIT(cb_s1, ck);
    POP(cb_kbeta, ck);
    mm(cb_tinv, cb_s1, cb_w, Ct, Ct, Kt, false);
    WAIT(cb_w, ck);
    DONE(cb_s1, ck);
    POP(cb_tinv, cc);

    // ---- intra = (q @ k^T) * L_mask, q_decay, k_dec_t ----
    mm(cb_q, cb_k, cb_s1, Ct, Kt, Ct, true);
    WAIT(cb_s1, cc);
    ew(cb_s1, cb_lmask, cb_intra, cc, MUL);
    WAIT(cb_intra, cc);
    DONE(cb_s1, cc);
    POP(cb_lmask, cc);
    bcast_cols_mul(cb_q, cb_dexp, cb_qdec, Ct, Kt);
    WAIT(cb_qdec, ck);
    POP(cb_q, ck);
    bcast_cols_mul(cb_k, cb_dfac, cb_s1, Ct, Kt);
    WAIT(cb_s1, ck);
    POP(cb_k, ck);
    transpose_block(cb_s1, cb_kdect, Ct, Kt);
    WAIT(cb_kdect, kc);
    DONE(cb_s1, ck);

    // ---- the scan step ----
    WAIT(cur_S, kv);
    // F32S: cur_S is unpacked to Dst as stored, so the matmuls read a copy of it in s3
    const uint32_t S_mm = F32S ? cb_s3 : cur_S;
    if constexpr (F32S) {
        copy_tiles(cur_S, cb_s3, kv);
        WAIT(cb_s3, kv);
    }
    mm(cb_w, S_mm, cb_s1, Ct, Kt, Vt, false);  // w @ S
    WAIT(cb_s1, cv);
    POP(cb_w, ck);
    ew(cb_u, cb_s1, cb_vnew, cv, SUB);  // v_new = u - w @ S
    WAIT(cb_vnew, cv);
    POP(cb_u, cv);
    DONE(cb_s1, cv);
    mm(cb_qdec, S_mm, cb_s1, Ct, Kt, Vt, false);  // q_decay @ S
    WAIT(cb_s1, cv);
    POP(cb_qdec, ck);
    if constexpr (F32S) {
        DONE(cb_s3, kv);  // s3 takes o next
    }
    mm(cb_intra, cb_vnew, cb_s2, Ct, Ct, Vt, false);  // intra @ v_new
    WAIT(cb_s2, cv);
    POP(cb_intra, cc);
    if constexpr (gated) {
        ew(cb_s1, cb_s2, cb_s3, cv, ADD);  // o
        WAIT(cb_s3, cv);
        DONE(cb_s1, cv);
        DONE(cb_s2, cv);
        gnorm(cb_s3, Ct, Vt, norm_eps, inv_v);
    } else {
        ew(cb_s1, cb_s2, cb_out, cv, ADD);  // o
        DONE(cb_s1, cv);
        DONE(cb_s2, cv);
    }
    if constexpr (F32S) {
        POP(cb_dfac, Ct);
        POP(cb_dexp, Ct);
        WAIT(cb_decay, Ct);
        snew_f32(cur_S, nxt_S, Ct, Vt, kv);  // pops cur_S as it goes
        POP(cb_decay, Ct);
        POP(cb_kdect, kc);
        POP(cb_vnew, cv);
        return;
    }
    mm(cb_kdect, cb_vnew, cb_s1, Kt, Ct, Vt, false);  // k_dec_t @ v_new
    WAIT(cb_s1, kv);
    POP(cb_kdect, kc);
    POP(cb_vnew, cv);
    // exp(g_sum) = decay_fac[0] * decay_exp[0]; element [0,0] of the product is all the
    // scalar broadcast reads.
    ew(cb_dfac, cb_dexp, cb_s3, 1, MUL);
    WAIT(cb_s3, 1);
    POP(cb_dfac, Ct);
    POP(cb_dexp, Ct);
    bcast_scalar_mul(cur_S, cb_s3, cb_s2, kv);
    WAIT(cb_s2, kv);
    DONE(cb_s3, 1);
    POP(cur_S, kv);
    ew(cb_s2, cb_s1, nxt_S, kv, ADD);
    DONE(cb_s2, kv);
    DONE(cb_s1, kv);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr uint32_t NORM = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_INV = get_compile_time_arg_val(5);
    const uint32_t nh = get_arg_val<uint32_t>(0);
    const uint32_t NC = get_arg_val<uint32_t>(1);
    const uint32_t scale = get_arg_val<uint32_t>(2);
    const uint32_t eps = get_arg_val<uint32_t>(3);
    const uint32_t norm_eps = get_arg_val<uint32_t>(4);
    const uint32_t inv_v = get_arg_val<uint32_t>(5);
    constexpr uint32_t GNORM = get_compile_time_arg_val(7);
    constexpr uint32_t GATES = get_compile_time_arg_val(8);
    constexpr uint32_t cc = Ct * Ct;

    compute_kernel_hw_startup(cb_v, cb_g, cb_out);

    // Constants, pushed once by the reader and never popped.
    WAIT(cb_eye, cc);
    WAIT(cb_tril, cc);
    WAIT(cb_ones, cc + (GNORM ? Vt : 0));  // with GNORM the norm weight's Vt tiles behind the ones

    for (uint32_t h = 0; h < nh; h++) {
        // Per head (GATES): -exp(A_log) and dt_bias, fp32 bits.
        const uint32_t a_neg = GATES ? get_arg_val<uint32_t>(6 + 2 * h) : 0;
        const uint32_t dt = GATES ? get_arg_val<uint32_t>(7 + 2 * h) : 0;
        // The state passes through single-producer CBs: cb_S0 (the reader's, chunk 0 only) and
        // the compute-only cb_sa (each chunk pops the state it read before it pushes the next);
        // the last chunk writes cb_final.
        for (uint32_t c = 0; c < NC; c++) {
            const uint32_t cur_S = (c == 0) ? cb_S0 : cb_sa;
            const uint32_t nxt_S = (c == NC - 1) ? cb_final : cb_sa;
            chunk<NORM != 0, BLOCK_INV != 0, GNORM != 0, GATES != 0>(
                cur_S, nxt_S, Ct, Kt, Vt, scale, eps, norm_eps, inv_v, a_neg, dt);
        }
    }
}
