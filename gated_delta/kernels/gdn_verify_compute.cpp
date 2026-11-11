// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet multi-row decode: R decode steps of the gated delta rule in order (one token each)
// for a list of heads on one Tensix core, as a speculative verify runs them. The state after each
// row goes to the writer (rows 0..R-2: the verify's slots; row R-1: the layer's state buffer, in
// place) and, packed in the same format, to cb_Sc for the next row. This matches
// single-row decode through gdn_decode.py, which reads the state it leaves in DRAM.
// It replaces R calls of upstream's recurrent_gated_delta_rule_decode_ttnn
// (tenstorrent/tt-metal v0.81.1, models/experimental/gated_attention_gated_deltanet/tt/
// ttnn_delta_rule_ops.py, Apache-2.0) with the state copies between them, and the forward's head
// reshapes, repeat_interleave, gate math and gated RMSNorm around them. The tile helpers are the
// prefill kernel's (gdn_ops.h).
//
// Per head and row (K = V = 128 as Kt = Vt = 4 tiles whose row 0 holds the token):
//   q = q / |q| * scale, k = k / |k|
//   g = a_neg * softplus(a + dt_bias), beta = sigmoid(b)   (RAW; else g and beta read as given)
//   S_dec = exp(g) * S
//   v_new = beta * (v - k @ S_dec)
//   S' = S_dec + k^T v_new           -> the writer, and the next row's S (cb_Sc)
//   o = q @ S_dec + (q . k) v_new
//   out = rmsnorm(o) * w * silu(gate) (GNORM), else o
//
// Each acquire fills up to DST tiles. Every intermediate fills a CB exactly, and
// initialization sits outside tile loops where possible. A group holds all R rows
// of up to 32 / R heads. Gates occupy row 0 of one tile: row r of head i is column
// i R + r. Compute gates once per group, then move each item to [0, 0] with E @ L.
//
// Normalize q and k once per group. The reader gathers raw rows into cb_qg and cb_kg,
// with item j in row j. Compute writes cb_qn and cb_kn; the reader copies each item's
// normalized rows back to row 0 of cb_q and cb_k. The writer gathers each item's o
// from row 0 of cb_orow into row j of cb_oasm for grouped gated normalization. The
// weight broadcasts down rows, and SiLU uses gathered gate rows. Norms are row-wise,
// so moving rows does not change the arithmetic. Data-movement copies preserve bytes.
//
// CLIP clamps final tiles in fp32 Dst before packing. Output-format rounding is
// monotonic, so this matches ttnn.clip after packing when c rounds identically in
// both paths. gdn_verify.py accepts only bounds whose fp32 values round to the same
// bf16 value by truncation and round-to-nearest. For example, 1e4 satisfies this;
// 0.3 does not.
//
// Narrow SFPU chains process only rows used by items through gdn_compat.h's *_rows
// and rsqrt_col. Gates use row 0 (two faces, two iterations, covering rows 0-3).
// Norm reciprocals use column 0 through the last item; gated SiLU uses item rows.
// ITEM_ITER is 2 ceil(items / 4) for the largest group. Above 16 items it is 0,
// selecting full-tile ops. Used elements receive the full op's math. Gate shifts
// and broadcasts read row 0 and [0, 0]; norms are row-wise; the reader and writer
// copy only item rows. Unused rows do not affect the result.
//
// SSM (compile-time arg 10) runs Mamba-2's recurrence with the same dataflow. q and k
// are C and B, v is x (K = state size N, V = head dimension P; K and V may differ);
// no L2 norms (gathered rows are copied as they are). Gates give exp(A dt) as
// exp(g) and dt = softplus(a + dt_bias) in place of beta; v_new = dt x (no k S); and the output gets
// D x added: o = C S' + D x (= C S_dec + (C . B) v_new + D x), with D per item from the reader's
// cb_t1a page, moved to [0, 0] item by item as the gates are (cb_t1b, 2 tiles). No gated norm
// (GNORM 0). DT_FLOOR (compile-time arg 11, fp32 bits; 0: none): dt = max(softplus(a + dt_bias),
// floor), the modeling code's torch_forward (time_step_min), before both the decay and v_new.
//
// SSM updates state in fp32 Dst. FPU reads truncate fp32 to tf32; repeatedly using
// those reads for decay and addition can accumulate bias when decay is near one.
// row_ssm keeps g, not exp(g), in cb_dec. Only g passes through the FPU: ones * g[0, 0]
// fills a tile, so truncation error is relative to g. SFPU exp produces fp32 decay
// in cb_dfull. The state unpacks directly to Dst as stored, SFPU multiplication
// applies decay, and the FPU matmul accumulates B^T v_new onto those Dst tiles.
// S' goes to the writer, cb_Sc, and cb_sd; the output is C S' + D x.
//
// F32S (compile-time arg 12) also keeps DeltaNet decay and addition in fp32 Dst.
// The k and q matmuls read S_dec at tf32; their truncation affects a single update
// or output rather than directly truncating the carried state. Gates leave g in
// cb_dec. Pass 1 handles DST - 1 state tiles per acquire: form decay in Dst 0 with
// ones * g[0, 0] and exp on one SFPU row per face, unpack state to Dst, then multiply
// by a decay value held in a register per face. Pack S_dec to cb_sd for matmuls and
// cb_t16 for unpack-to-dest. After computing v_new, pass 2 reloads S_dec from cb_t16
// without truncation and accumulates k^T v_new with the FPU matmul. Pack S' to the
// writer and cb_Sc. F32S 0 uses the FPU decay and addition path.
//
// F32S 2 uses one update pass; F32S 1 uses the two-pass form above. Compute
// v_new = beta (v - exp(g) k S), where S comes in a second, FPU-readable copy (at a head's first row the reader's copy
// of the state in cb_t16; after it the previous row's S', packed to cb_sd); then one pass over the state at fp32 as
// the SSM mode's row_ssm (the decay made in Dst, the state by unpack-to-dest, the SFPU's multiply, k^T v_new
// accumulated onto it by the matmul): S' to the writer, cb_Sc and cb_sd, and o = q S' (= q S_dec + (q . k) v_new).
// This avoids a second unpack and pack of the state tiles. The reader provides an
// additional L1 copy for the initial state's FPU reads.
//
// Compile-time args: Kt, Vt, RAW, R (rows), SCR (unused here; gdn_ops.h reads it),
// GNORM, DST (tiles per acquire), NH (maximum heads per core; unread), CLIP (fp32
// bits, 0 disables clipping), ITEM_ITER (0 selects full ops). SCR and DST occupy
// indices 4 and 6 for gdn_ops.h.
// Runtime args: heads on this core, q scale, L2 eps, the gated norm's eps, 1 / V (fp32 bits).

// O2 limits kernel text size. A program larger than the device's kernel config ring
// fails at enqueue; a smaller worker L1 gives a larger ring.
#pragma GCC optimize("O2")

#include <cstdint>
// DeviceZoneScopedN marks device profiler zones.
#include "api/debug/kernel_profiler.hpp"
#include "gdn_compat.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"

namespace {

// CB plan (must match gdn_verify.py). t4a-t4c hold Vt tiles (Kt for DeltaNet), t1a/t1b one tile, t16 the
// Kt x Vt outer product: each block fills its CB exactly.
constexpr uint32_t cb_qg = 0, cb_kg = 1, cb_v = 2, cb_gab = 3, cb_oasm = 4, cb_gg = 5, cb_S0 = 6;
constexpr uint32_t cb_ones = 7;  // the ones tile, L, the norm weight's Vt tiles (row 0 each)
constexpr uint32_t cb_q = 8, cb_k = 9, cb_sd = 10, cb_dec = 11, cb_bt = 12, cb_dl = 13, cb_kt = 14;
constexpr uint32_t cb_Sout = 15, cb_out = 16;
constexpr uint32_t cb_t4a = 17, cb_t4b = 18, cb_t4c = 19, cb_qk = 20, cb_t1a = 21, cb_t1b = 22, cb_t16 = 23;
constexpr uint32_t cb_gd = 24, cb_gn = 25;  // dt_bias and a_neg per value (RAW); cb_dec and cb_bt hold 2 tiles
constexpr uint32_t cb_qn = 27, cb_kn = 30;  // the group's normalized q and k rows, for the reader
constexpr uint32_t cb_Sc = 29;              // the state after the previous row, in the state's format
constexpr uint32_t cb_orow = 31;            // an item's o (row 0), for the writer
constexpr uint32_t cb_dfull = cb_qk;        // SSM: exp(g) over a whole tile (fp32, unpack-to-dest); qk unused
constexpr uint32_t GATES = 32;              // values per gate tile (the columns of its row 0)
// cb_ones: the ones tile, L (the gates' shift), the norm weight's tiles.
constexpr uint32_t UPI = 1, WI = 2;
// gdn_ops.h's l2norm and gnorm name these; this kernel has its own versions and never calls them.
constexpr uint32_t cb_s1 = cb_t16, cb_s2 = cb_t4a, cb_s3 = cb_t4b, cb_gt = cb_gg;

#include "gdn_ops.h"

constexpr uint32_t CLIP = get_compile_time_arg_val(8);  // c as fp32 bits; 0: no clip
constexpr uint32_t ITEM_ITER = get_compile_time_arg_val(9);  // SFPU iterations for the items' rows; 0: full
constexpr uint32_t SSM = get_compile_time_arg_val(10);       // Mamba-2's step (see above)
constexpr uint32_t DT_FLOOR = get_compile_time_arg_val(11);  // SSM: dt's floor as fp32 bits; 0: none
static_assert(SSM != 0 || DT_FLOOR == 0, "DT_FLOOR: SSM only");
constexpr uint32_t F32S = get_compile_time_arg_val(12);  // the DeltaNet's state update at fp32 (see above)
static_assert(SSM == 0 || F32S == 0, "F32S: the DeltaNet's (the SSM mode has its own)");
static_assert(F32S == 0 || DST >= 2, "F32S: DST >= 2");
static_assert(F32S <= 2, "F32S: 0, 1 or 2");
#ifdef GDN_KDA
// KDA (Kimi Delta Attention, as used by Ling-3.0-tiny) uses per-key-channel decay,
// fp32 state updates on the F32S 2 path, and sigmoid gating. gdn_verify.py's kda
// argument enables it at compile time through GDN_KDA.
constexpr bool KDA = true;
static_assert(F32S == 2 && SSM == 0, "KDA: F32S 2 (an fp32 state), not the SSM mode");
#else
constexpr bool KDA = false;
#endif

// SSM: dt = max(dt, floor) in Dst (min(x, +inf) leaves it), over row 0 as the other gate ops (the items'
// columns; the rest is read by nothing): the same SFPU function and LLK entry as clamp_tile, as gdn_compat.h's
// *_rows; the whole tile when ITEM_ITER is 0, as they do.
template <int ITER>
ALWI void clamp_rows(uint32_t idst, uint32_t lo, uint32_t hi) {
    MATH((GDN_UPARAMS((ckernel::sfpu::calculate_clamp<APPROX, ITER>), idst, R, lo, hi)));
}
ALWI void dt_floor(uint32_t idst) {
    if constexpr (DT_FLOOR != 0) {
        clamp_tile_init();
        if constexpr (ITEM_ITER == 0) {
            clamp_tile(idst, DT_FLOOR, 0x7f800000u);
        } else {
            clamp_rows<2>(idst, DT_FLOOR, 0x7f800000u);
        }
    }
}
static_assert(ITEM_ITER == 0 || ITEM_ITER == 2 || ITEM_ITER == 4 || ITEM_ITER == 6 || ITEM_ITER == 8, "ITEM_ITER");

// 1 / sqrt on a norm's row sums (column 0, the items' rows) or the whole tile
ALWI void norm_rsqrt(uint32_t idst) {
    if constexpr (ITEM_ITER == 0) {
        rsqrt_tile(idst);
    } else {
        gdn::rsqrt_col<(int)ITEM_ITER - 1>(idst);
    }
}

// The kernel's output: o = A (op) B over n tiles (ew), clamped to [-c, c] in Dst before the pack
// when CLIP is set. The binary op's init again for each group, after the previous group's clamp.
__attribute__((noinline)) void ew_out(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    if constexpr (CLIP == 0) {
        ew(a, b, o, n, op);
    } else {
        cb_reserve_back(o, n);
        pack_reconfig_data_format(o);
        reconfig_data_format(a, b);  // binary: a -> srcA, b -> srcB
        for (uint32_t t0 = 0; t0 < n; t0 += DST) {
            const uint32_t m = group(n, t0);
            if (op == ADD) {
                gdn::add_init(a, b);
            } else if (op == SUB) {
                gdn::sub_init(a, b);
            } else {
                gdn::mul_init(a, b);
            }
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
            clamp_tile_init();
            for (uint32_t d = 0; d < m; d++) {
                clamp_tile(d, CLIP | 0x80000000u, CLIP);  // [-c, c]
            }
            pack_group(m, o);
        }
        cb_push_back(o, n);
    }
}

// One tile: sum over each row of the Nt tiles of `sq` (matmul with the ones tile) -> o.
__attribute__((noinline)) void rowsum(uint32_t sq, uint32_t o, uint32_t Nt) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, sq);
    gdn::mm_init(sq, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t t = 0; t < Nt; t++) {
        matmul_tiles(sq, cb_ones, t, 0, 0);
    }
    pack_group(1, o);
    cb_push_back(o, 1);
}

// o1 = o2 = A + B over n tiles (o1 and o2 in one data format: each Dst group packed twice).
__attribute__((noinline)) void add_twice(uint32_t a, uint32_t b, uint32_t o1, uint32_t o2, uint32_t n) {
    cb_reserve_back(o1, n);
    cb_reserve_back(o2, n);
    pack_reconfig_data_format(o1);
    reconfig_data_format(a, b);
    gdn::add_init(a, b);
    for (uint32_t t0 = 0; t0 < n; t0 += DST) {
        const uint32_t m = group(n, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            add_tiles(a, b, t0 + d, t0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; d++) {
            pack_tile(d, o1);
        }
        for (uint32_t d = 0; d < m; d++) {
            pack_tile(d, o2);
        }
        tile_regs_release();
    }
    cb_push_back(o1, n);
    cb_push_back(o2, n);
}

// o = raw / sqrt(sum_k raw^2 + eps) (* scale), one row over Kt tiles (Ct = 1).
__attribute__((noinline)) void l2norm1(uint32_t raw, uint32_t o, uint32_t Kt, uint32_t eps, uint32_t scale,
                                       bool scaled) {
    ew(raw, raw, cb_t4a, Kt, MUL);  // squares, fp32
    WAIT(cb_t4a, Kt);
    rowsum(cb_t4a, cb_t1a, Kt);
    POP(cb_t4a, Kt);
    WAIT(cb_t1a, 1);
    cb_reserve_back(cb_t1b, 1);  // 1 / sqrt(sum + eps), times the scale for q
    pack_reconfig_data_format(cb_t1b);
    reconfig_data_format_srca(cb_t1a);
    gdn::copy_init(cb_t1a);
    tile_regs_acquire();
    copy_tile(cb_t1a, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    norm_rsqrt(0);
    if (scaled) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale);
    }
    pack_group(1, cb_t1b);
    cb_push_back(cb_t1b, 1);
    POP(cb_t1a, 1);
    WAIT(cb_t1b, 1);
    bcast_cols_mul(raw, cb_t1b, o, 1, Kt);
    POP(cb_t1b, 1);
}

// cb_oasm (the group's o rows, Vt tiles, consumed) and cb_gg (its gate rows, consumed) -> cb_out =
// rmsnorm(o) * w * silu(gate), row by row on the gathered rows (Ct = 1).
__attribute__((noinline)) void gnorm_group(uint32_t Vt, uint32_t eps, uint32_t inv_v) {
    constexpr uint32_t o = cb_oasm;
    WAIT(o, Vt);
    ew(o, o, cb_t4a, Vt, MUL);
    WAIT(cb_t4a, Vt);
    rowsum(cb_t4a, cb_t1a, Vt);  // row sums of the squares
    POP(cb_t4a, Vt);
    WAIT(cb_t1a, 1);
    cb_reserve_back(cb_t1b, 1);  // 1 / sqrt(mean + eps)
    pack_reconfig_data_format(cb_t1b);
    reconfig_data_format_srca(cb_t1a);
    gdn::copy_init(cb_t1a);
    tile_regs_acquire();
    copy_tile(cb_t1a, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_v);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    norm_rsqrt(0);
    pack_group(1, cb_t1b);
    cb_push_back(cb_t1b, 1);
    POP(cb_t1a, 1);
    WAIT(cb_t1b, 1);
    bcast_cols_mul(o, cb_t1b, cb_t4a, 1, Vt);  // o / rms
    WAIT(cb_t4a, Vt);
    POP(cb_t1b, 1);
    POP(o, Vt);
    cb_reserve_back(cb_t4b, Vt);  // * w: row 0 of weight tile t down every row
    pack_reconfig_data_format(cb_t4b);
    reconfig_data_format(cb_t4a, cb_ones);
    gdn::mul_bcast_rows_init(cb_t4a, cb_ones);
    for (uint32_t t0 = 0; t0 < Vt; t0 += DST) {
        const uint32_t m = group(Vt, t0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            mul_tiles_bcast_rows(cb_t4a, cb_ones, t0 + d, WI + t0 + d, d);
        }
        pack_group(m, cb_t4b);
    }
    cb_push_back(cb_t4b, Vt);
    POP(cb_t4a, Vt);
    WAIT(cb_gg, Vt);
    cb_reserve_back(cb_t4c, Vt);  // silu(gate)
    pack_reconfig_data_format(cb_t4c);
    reconfig_data_format_srca(cb_gg);
    for (uint32_t t0 = 0; t0 < Vt; t0 += DST) {
        const uint32_t m = group(Vt, t0);
        gdn::copy_init(cb_gg);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            copy_tile(cb_gg, t0 + d, d);
        }
        if constexpr (KDA) {  // KDA: sigmoid(gate)
            sigmoid_tile_init();
            for (uint32_t d = 0; d < m; d++) {
                if constexpr (ITEM_ITER == 0) {
                    sigmoid_tile(d);
                } else {
                    gdn::sigmoid_rows<(int)ITEM_ITER>(d);
                }
            }
        } else {
            silu_tile_init();
            for (uint32_t d = 0; d < m; d++) {
                if constexpr (ITEM_ITER == 0) {
                    silu_tile(d);
                } else {
                    gdn::silu_rows<(int)ITEM_ITER>(d);
                }
            }
        }
        pack_group(m, cb_t4c);
    }
    cb_push_back(cb_t4c, Vt);
    POP(cb_gg, Vt);
    WAIT(cb_t4b, Vt);
    WAIT(cb_t4c, Vt);
    ew_out(cb_t4b, cb_t4c, cb_out, Vt, MUL);  // the output, clipped (CLIP)
    POP(cb_t4b, Vt);
    POP(cb_t4c, Vt);
}

// exp(g) (SSM and F32S: g itself) into cb_dec and beta (SSM: dt) into cb_bt for a group of heads' rows, one
// column of row 0 each: from the raw a and b with dt_bias and a_neg (RAW) or from g and beta.
__attribute__((noinline)) void gates(bool raw) {
    constexpr uint32_t ONE = 0x3f800000u, THRESHOLD = 0x41a00000u;  // 1.0, 20.0 (torch's softplus)
    WAIT(cb_gab, 2);  // the a (or g) page, then the b (or beta) page
    if constexpr (!KDA) {  // KDA: no scalar g (the decay is per channel, made per item: kda_decay)
    cb_reserve_back(cb_dec, 1);
    pack_reconfig_data_format(cb_dec);
    reconfig_data_format_srca(cb_gab);
    gdn::copy_init(cb_gab);
    tile_regs_acquire();
    copy_tile(cb_gab, 0, 0);
    if (raw) {
        WAIT(cb_gd, 1);
        WAIT(cb_gn, 1);
        reconfig_data_format_srca(cb_gd);
        gdn::copy_init(cb_gd);
        copy_tile(cb_gd, 0, 1);
        add_binary_tile_init();
        if constexpr (ITEM_ITER == 0) {
            add_binary_tile(0, 1, 0);  // a + dt_bias
        } else {
            gdn::add_binary_rows<2>(0, 1, 0);  // row 0 (the items' columns)
        }
        softplus_tile_init();
        if constexpr (ITEM_ITER == 0) {
            softplus_tile(0, ONE, ONE, THRESHOLD);
        } else {
            gdn::softplus_rows<2>(0, ONE, ONE, THRESHOLD);
        }
        dt_floor(0);
        reconfig_data_format_srca(cb_gn);
        gdn::copy_init(cb_gn);
        copy_tile(cb_gn, 0, 1);
        mul_binary_tile_init();
        if constexpr (ITEM_ITER == 0) {
            mul_binary_tile(0, 1, 0);  // g = a_neg * softplus(...)
        } else {
            gdn::mul_binary_rows<2>(0, 1, 0);
        }
    }
    if constexpr (!SSM && !F32S) {  // SSM, F32S: g; the row takes its exp per item over a whole tile
        exp_tile_init();
        if constexpr (ITEM_ITER == 0) {
            exp_tile(0);
        } else {
            gdn::exp_rows<2>(0);
        }
    }
    pack_group(1, cb_dec);
    cb_push_back(cb_dec, 1);
    if (raw && !SSM) {
        POP(cb_gd, 1);
        POP(cb_gn, 1);
    }
    }  // !KDA

    cb_reserve_back(cb_bt, 1);
    pack_reconfig_data_format(cb_bt);
    reconfig_data_format_srca(cb_gab);
    gdn::copy_init(cb_gab);
    tile_regs_acquire();
    if constexpr (SSM) {
        // dt = softplus(a + dt_bias): the decay's own first steps again (the same ops, the same values)
        copy_tile(cb_gab, 0, 0);
        reconfig_data_format_srca(cb_gd);
        gdn::copy_init(cb_gd);
        copy_tile(cb_gd, 0, 1);
        add_binary_tile_init();
        if constexpr (ITEM_ITER == 0) {
            add_binary_tile(0, 1, 0);
        } else {
            gdn::add_binary_rows<2>(0, 1, 0);
        }
        softplus_tile_init();
        if constexpr (ITEM_ITER == 0) {
            softplus_tile(0, ONE, ONE, THRESHOLD);
        } else {
            gdn::softplus_rows<2>(0, ONE, ONE, THRESHOLD);
        }
        dt_floor(0);
    } else {
        copy_tile(cb_gab, 1, 0);
        if (raw) {
            sigmoid_tile_init();
            if constexpr (ITEM_ITER == 0) {
                sigmoid_tile(0);
            } else {
                gdn::sigmoid_rows<2>(0);
            }
        }
    }
    pack_group(1, cb_bt);
    cb_push_back(cb_bt, 1);
    POP(cb_gab, 2);
    if constexpr (SSM) {
        POP(cb_gd, 1);
        POP(cb_gn, 1);
        // D per item, as the reader wrote it (cb_t1a), into the shifted CB (cb_t1b)
        WAIT(cb_t1a, 1);
        cb_reserve_back(cb_t1b, 1);
        pack_reconfig_data_format(cb_t1b);
        reconfig_data_format_srca(cb_t1a);
        gdn::copy_init(cb_t1a);
        tile_regs_acquire();
        copy_tile(cb_t1a, 0, 0);
        pack_group(1, cb_t1b);
        cb_push_back(cb_t1b, 1);
        POP(cb_t1a, 1);
    }
}

// SSM: n tiles of `in` copied to `o` as they are (in place of the L2 norm)
__attribute__((noinline)) void copy_block(uint32_t in, uint32_t o, uint32_t n) {
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
        pack_group(m, o);
    }
    cb_push_back(o, n);
}

// The next row's gate to [0, 0]: the tile at the front of cb (2 tiles) @ L, the old one popped.
__attribute__((noinline)) void shift_up(uint32_t cb, uint32_t li) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    reconfig_data_format(cb_ones, cb);
    gdn::mm_init(cb, cb_ones, 0);
    tile_regs_acquire();
    matmul_tiles(cb, cb_ones, 0, li, 0);
    pack_group(1, cb);
    cb_push_back(cb, 1);
    POP(cb, 1);
}

// F32S's decay multiply, using one value per face (default; GDN_FULL_EXP: exp over the whole tile and mul_binary_tile): ones * g
// makes every element of the decay's tile the same value, so exp runs on one SFPU row a face (exp_first) and the
// multiply loads that row once a face into a register for the face's 8 rows of each state tile (mul_dec).
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
ALWI void exp_dec(uint32_t idst) {
    exp_tile_init();
#ifdef GDN_FULL_EXP
    exp_tile(idst);
#else
    gdn::exp_first(idst);
#endif
}
ALWI void mul_dec_init() { mul_binary_tile_init(); }
ALWI void mul_dec(uint32_t i0, uint32_t i1, uint32_t o) {
#ifdef GDN_FULL_EXP
    mul_binary_tile(i0, i1, o);
#else
    MATH((GDN_BPARAMS((mul_by_dec_face<8>), i0, i1, o, RC)));
#endif
}

// F32S pass 1: S_dec = exp(g) * S in Dst at fp32, DST - 1 tiles an acquire after the decay in Dst 0, which each
// acquire makes itself (ones * g[0, 0] on the FPU, exp on one row a face): no copy of it. s_in consumed chunk by
// chunk; S_dec packed to cb_sd (for the matmuls) and to cb_t16 (for pass 2). Then the next item's g to [0, 0].
__attribute__((noinline)) void sdec_f32(uint32_t s_in, uint32_t kv, bool next, uint32_t upi) {
    constexpr uint32_t SD = DST - 1;
    cb_reserve_back(cb_sd, kv);
    pack_reconfig_data_format(cb_sd);  // cb_sd and cb_t16: fp32
    for (uint32_t t0 = 0; t0 < kv; t0 += SD) {
        const uint32_t m = (kv - t0 < SD) ? (kv - t0) : SD;
        tile_regs_acquire();
        reconfig_data_format(cb_ones, cb_dec);
        gdn::mul_bcast_scalar_init(cb_ones, cb_dec);
        mul_tiles_bcast_scalar(cb_ones, cb_dec, 0, 0, 0);  // g everywhere (its tf32 read is relative to g)
        exp_dec(0);
        reconfig_data_format_srca(s_in);
        gdn::copy_init(s_in);
        for (uint32_t d = 0; d < m; d++) {
            copy_tile(s_in, d, 1 + d);
        }
        POP(s_in, m);
        mul_dec_init();
        for (uint32_t d = 0; d < m; d++) {
            mul_dec(1 + d, 0, 1 + d);
        }
        cb_reserve_back(cb_t16, m);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; d++) {
            pack_tile(1 + d, cb_sd);
        }
        for (uint32_t d = 0; d < m; d++) {
            pack_tile(1 + d, cb_t16);
        }
        tile_regs_release();
        cb_push_back(cb_t16, m);
    }
    cb_push_back(cb_sd, kv);
    if (next) {
        shift_up(cb_dec, upi);
    } else {
        POP(cb_dec, 1);
    }
}

// F32S pass 2: S' = S_dec (cb_t16, unpack-to-dest) + k^T v_new (the FPU's matmul accumulating onto it), DST tiles an
// acquire -> cb_Sout and, unless it is the head's last row, cb_Sc.
__attribute__((noinline)) void snew_f32(uint32_t Vt, uint32_t kv, bool carry) {
    WAIT(cb_t16, kv);
    cb_reserve_back(cb_Sout, kv);
    if (carry) {
        cb_reserve_back(cb_Sc, kv);
    }
    pack_reconfig_data_format(cb_Sout);  // cb_Sout and cb_Sc: the state's format
    for (uint32_t t0 = 0; t0 < kv; t0 += DST) {
        const uint32_t m = group(kv, t0);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_t16);
        gdn::copy_init(cb_t16);
        for (uint32_t d = 0; d < m; d++) {
            copy_tile(cb_t16, d, d);
        }
        POP(cb_t16, m);
        reconfig_data_format(cb_dl, cb_kt);  // matmul: in0 -> srcB, in1 -> srcA
        gdn::mm_init(cb_kt, cb_dl, 0);
        for (uint32_t d = 0; d < m; d++) {
            const uint32_t t = t0 + d;
            matmul_tiles(cb_kt, cb_dl, t / Vt, t % Vt, d);  // += k^T v_new, tile (t / Vt, t % Vt)
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; d++) {
            pack_tile(d, cb_Sout);
        }
        if (carry) {
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(d, cb_Sc);
            }
        }
        tile_regs_release();
    }
    cb_push_back(cb_Sout, kv);
    if (carry) {
        cb_push_back(cb_Sc, kv);
    }
}

// The math thread waits for the SFPU to finish (an FPU op that reads its Dst tiles comes next).
ALWI void wait_sfpu() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(ckernel::p_stall::STALL_CFG | ckernel::p_stall::STALL_MATH, ckernel::p_stall::WAIT_SFPU);
#endif
}

// F32S 2: the output o = q S' (in t4a) to cb_out, clamped in Dst to [-c, c] before the pack when CLIP is set.
__attribute__((noinline)) void out_f32(uint32_t Vt) {
    cb_reserve_back(cb_out, Vt);
    pack_reconfig_data_format(cb_out);
    reconfig_data_format_srca(cb_t4a);
    for (uint32_t t0 = 0; t0 < Vt; t0 += DST) {
        const uint32_t m = group(Vt, t0);
        gdn::copy_init(cb_t4a);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; d++) {
            copy_tile(cb_t4a, t0 + d, d);
        }
        if constexpr (CLIP != 0) {
            clamp_tile_init();
            for (uint32_t d = 0; d < m; d++) {
                clamp_tile(d, CLIP | 0x80000000u, CLIP);  // [-c, c]
            }
        }
        pack_group(m, cb_out);
    }
    cb_push_back(cb_out, Vt);
}

// F32S 2, one item (one row of one head, its q and k normalized already): see the head of this file. s_in: the
// state for the update (unpack-to-dest); s_mm: the same state for k S (the FPU's operand).
template <bool gated>
__attribute__((noinline)) void row_f32(uint32_t Kt, uint32_t Vt, uint32_t s_in, uint32_t s_mm, bool carry, bool next,
                                       uint32_t upi) {
    const uint32_t kv = Kt * Vt;
    constexpr uint32_t SD = DST - 1;  // state tiles an acquire, after the decay in Dst 0
    {
        DeviceZoneScopedN("V-wait-in");
        WAIT(cb_q, Kt);
        WAIT(cb_k, Kt);
        WAIT(cb_v, Vt);
        WAIT(s_in, kv);
        WAIT(s_mm, kv);
    }
    WAIT(cb_dec, 1);
    WAIT(cb_bt, 1);
    {
        DeviceZoneScopedN("V-kS");
        mm(cb_k, s_mm, cb_t4a, 1, Kt, Vt, false);  // k @ S (read at tf32 once, not carried)
        WAIT(cb_t4a, Vt);
        POP(s_mm, kv);
    }
    {
        DeviceZoneScopedN("V-dec");  // exp(g) as the FPU's scalar for exp(g) k S (its tf32 read touches v_new once)
        cb_reserve_back(cb_qk, 1);
        pack_reconfig_data_format(cb_qk);
        reconfig_data_format(cb_ones, cb_dec);
        gdn::mul_bcast_scalar_init(cb_ones, cb_dec);
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_ones, cb_dec, 0, 0, 0);
        exp_dec(0);
        pack_group(1, cb_qk);
        cb_push_back(cb_qk, 1);
        WAIT(cb_qk, 1);
    }
    {
        DeviceZoneScopedN("V-vnew");
        bcast_scalar_mul(cb_t4a, cb_qk, cb_t4c, Vt);  // k S_dec = exp(g) k S
        WAIT(cb_t4c, Vt);
        POP(cb_t4a, Vt);
        POP(cb_qk, 1);
        ew(cb_v, cb_t4c, cb_t4b, Vt, SUB);  // v - k S_dec
        WAIT(cb_t4b, Vt);
        POP(cb_t4c, Vt);
        POP(cb_v, Vt);
        bcast_scalar_mul(cb_t4b, cb_bt, cb_dl, Vt);  // v_new = beta * (...)
        WAIT(cb_dl, Vt);
        POP(cb_t4b, Vt);
        if (next) {
            shift_up(cb_bt, upi);
        } else {
            POP(cb_bt, 1);
        }
    }
    {
        DeviceZoneScopedN("V-kT");
        transpose_block(cb_k, cb_kt, 1, Kt);  // k^T: Kt tiles of one column
        WAIT(cb_kt, Kt);
    }
    {
        DeviceZoneScopedN("V-Snew");  // S' = exp(g) S + k^T v_new in Dst at fp32, SD tiles an acquire
        cb_reserve_back(cb_Sout, kv);
        cb_reserve_back(cb_sd, kv);
        pack_reconfig_data_format(cb_Sout);  // cb_Sout, cb_Sc and cb_sd: the state's format (gdn_verify.py)
        for (uint32_t t0 = 0; t0 < kv; t0 += SD) {
            const uint32_t m = (kv - t0 < SD) ? (kv - t0) : SD;
            tile_regs_acquire();
            reconfig_data_format(cb_ones, cb_dec);
            gdn::mul_bcast_scalar_init(cb_ones, cb_dec);
            mul_tiles_bcast_scalar(cb_ones, cb_dec, 0, 0, 0);  // g everywhere
            exp_dec(0);
            reconfig_data_format_srca(s_in);
            gdn::copy_init(s_in);
            for (uint32_t d = 0; d < m; d++) {
                copy_tile(s_in, d, 1 + d);  // the chunk at the front of s_in
            }
            POP(s_in, m);  // s_in may be cb_Sc itself: its place is refilled below
            if (carry) {
                cb_reserve_back(cb_Sc, m);
            }
            mul_dec_init();
            for (uint32_t d = 0; d < m; d++) {
                mul_dec(1 + d, 0, 1 + d);  // S_dec, fp32
            }
            wait_sfpu();  // the matmul accumulates onto the SFPU's tiles
            reconfig_data_format(cb_dl, cb_kt);  // matmul: in0 -> srcB, in1 -> srcA
            gdn::mm_init(cb_kt, cb_dl, 0);
            for (uint32_t d = 0; d < m; d++) {
                const uint32_t t = t0 + d;
                matmul_tiles(cb_kt, cb_dl, t / Vt, t % Vt, 1 + d);  // += k^T v_new, tile (t / Vt, t % Vt)
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(1 + d, cb_Sout);
            }
            if (carry) {
                for (uint32_t d = 0; d < m; d++) {
                    pack_tile(1 + d, cb_Sc);
                }
            }
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(1 + d, cb_sd);
            }
            tile_regs_release();
            if (carry) {
                cb_push_back(cb_Sc, m);
            }
        }
        cb_push_back(cb_Sout, kv);
        cb_push_back(cb_sd, kv);
        if (next) {
            shift_up(cb_dec, upi);
        } else {
            POP(cb_dec, 1);
        }
        POP(cb_kt, Kt);
    }
    {
        DeviceZoneScopedN("V-qS");
        WAIT(cb_sd, kv);
        if constexpr (gated) {
            mm(cb_q, cb_sd, cb_orow, 1, Kt, Vt, false);  // o = q S' -> the writer, which gathers the group's rows
        } else if constexpr (CLIP == 0) {
            mm(cb_q, cb_sd, cb_out, 1, Kt, Vt, false);  // o = q S', the output
        } else {
            mm(cb_q, cb_sd, cb_t4a, 1, Kt, Vt, false);  // o = q S'
            WAIT(cb_t4a, Vt);
            out_f32(Vt);  // clipped (CLIP)
            POP(cb_t4a, Vt);
        }
        if (!carry) {
            POP(cb_sd, kv);  // else: the next row's k S
        }
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        POP(cb_dl, Vt);
    }
}

#ifdef GDN_KDA
constexpr uint32_t KDA_LB = 0xc0a00000u;  // -5.0f, Ling's kda_lower_bound
// cb_ones after the norm weight: E_i (row i ones, i < Kt), then G_i (column i ones, i < Kt), fp32
constexpr uint32_t KDA_EI = WI + (get_compile_time_arg_val(5) ? get_compile_time_arg_val(1) : 0);
constexpr uint32_t KDA_GI = KDA_EI + get_compile_time_arg_val(0);
constexpr uint32_t KDA_VT = get_compile_time_arg_val(1);  // t4a and t4b hold Vt tiles
static_assert(KDA_VT >= 2, "KDA: t4a takes T_hi and T_lo");
static_assert(DST <= 4, "KDA: kda_mul_rows takes at most DST - 1 = 3 state tiles an acquire");

// A block that does not fill a Vt-tile scratch CB leaves its pointers off its base, and the next Vt-tile block
// would spill past its end (the packer and unpacker do not wrap): a dummy push and pop of the rest brings them back.
ALWI void kda_rebase(uint32_t cb, uint32_t n) {
    const uint32_t rest = KDA_VT - n;
    cb_reserve_back(cb, rest);
    cb_push_back(cb, rest);
    cb_wait_front(cb, rest);
    gdn::pop(cb, rest);
}

// KDA, one item's decay: from f (cb_gd: rows 0-3 of an fp32 page, row i = f[32 i .. 32 i + 31]) and
// the head's dt_bias (cb_gn, the same layout), on the SFPU over rows 0-3: d = exp(-5 sigmoid(a (f + dt_bias))). Then
//   cb_dec: Kt tiles D_i, row r of tile i = d[32 i + r] in every column, at fp32: T_hi @ E_i + T_lo @ E_i accumulated
//           in Dst (T = the transpose of hi = bf16(d) and of lo = d - hi; hi passes the FPU exactly, lo's tf32
//           truncation is 2^-19 of d);
//   cb_qk:  Kt tiles R_i = G_i @ d, every row of tile i = d[32 i .. 32 i + 31] (tf32), for k S_dec = (k * d) S.
// E_i and G_i matmuls are independent. Process up to Kt tiles per acquire, bounded
// by DST, without serial column shifts or broadcasts.
__attribute__((noinline)) void kda_decay(uint32_t Kt, uint32_t a_bits) {
    {
        DeviceZoneScopedN("V-kgate");
        WAIT(cb_gd, 1);
        WAIT(cb_gn, 1);
        cb_reserve_back(cb_t1a, 1);
        cb_reserve_back(cb_qg, 1);
        cb_reserve_back(cb_t4b, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_gd);
        gdn::copy_init(cb_gd);
        copy_tile(cb_gd, 0, 0);
        reconfig_data_format_srca(cb_gn);
        gdn::copy_init(cb_gn);
        copy_tile(cb_gn, 0, 1);
        add_binary_tile_init();
        gdn::add_binary_rows<2>(0, 1, 0);  // f + dt_bias, rows 0-3
        binop_with_scalar_tile_init();
        mul_unary_tile(0, a_bits);  // * exp(A_log[h])
        sigmoid_tile_init();
        gdn::sigmoid_rows<2>(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, KDA_LB);  // g
        exp_tile_init();
        gdn::exp_rows<2>(0);  // d = exp(g), fp32
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_t1a);
        pack_tile(0, cb_t1a);
        pack_reconfig_data_format(cb_qg);
        pack_tile(0, cb_qg);  // hi = bf16(d)
        pack_reconfig_data_format(cb_t4b);
        pack_tile(0, cb_t4b);  // d for the FPU (cb_t1a unpacks to Dst only)
        tile_regs_release();
        cb_push_back(cb_t1a, 1);
        cb_push_back(cb_qg, 1);
        cb_push_back(cb_t4b, 1);
        POP(cb_gd, 1);
        POP(cb_gn, 1);
    }
    {
        DeviceZoneScopedN("V-ksplit");  // lo = d - hi, both at fp32 in Dst (d by unpack-to-dest, hi exact as bf16)
        WAIT(cb_t1a, 1);
        WAIT(cb_qg, 1);
        cb_reserve_back(cb_t1b, 1);
        pack_reconfig_data_format(cb_t1b);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_t1a);
        gdn::copy_init(cb_t1a);
        copy_tile(cb_t1a, 0, 0);
        reconfig_data_format_srca(cb_qg);
        gdn::copy_init(cb_qg);
        copy_tile(cb_qg, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        pack_group(1, cb_t1b);
        cb_push_back(cb_t1b, 1);
        POP(cb_t1a, 1);
        WAIT(cb_t1b, 1);
    }
    {
        DeviceZoneScopedN("V-ktr");  // T_hi, T_lo -> cb_t4a tiles 0 and 1 (one acquire)
        cb_reserve_back(cb_t4a, 2);
        pack_reconfig_data_format(cb_t4a);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_qg);
        gdn::transpose_init(cb_qg);
        gdn::transpose_tile(cb_qg, 0, 0);
        reconfig_data_format_srca(cb_t1b);
        gdn::transpose_init(cb_t1b);
        gdn::transpose_tile(cb_t1b, 0, 1);
        pack_group(2, cb_t4a);
        cb_push_back(cb_t4a, 2);
        POP(cb_qg, 1);
        POP(cb_t1b, 1);
        WAIT(cb_t4a, 2);
    }
    {
        DeviceZoneScopedN("V-kdec");  // D_i = T_hi @ E_i + T_lo @ E_i, accumulated in Dst at fp32
        cb_reserve_back(cb_dec, Kt);
        pack_reconfig_data_format(cb_dec);
        reconfig_data_format(cb_ones, cb_t4a);  // matmul: in0 -> srcB, in1 -> srcA
        gdn::mm_init(cb_t4a, cb_ones, 0);
        for (uint32_t i0 = 0; i0 < Kt; i0 += DST) {
            const uint32_t m = (Kt - i0 < DST) ? (Kt - i0) : DST;
            tile_regs_acquire();
            for (uint32_t d = 0; d < m; d++) {
                matmul_tiles(cb_t4a, cb_ones, 0, KDA_EI + i0 + d, d);
                matmul_tiles(cb_t4a, cb_ones, 1, KDA_EI + i0 + d, d);
            }
            pack_group(m, cb_dec);
        }
        cb_push_back(cb_dec, Kt);
        POP(cb_t4a, 2);
        kda_rebase(cb_t4a, 2);
    }
    {
        DeviceZoneScopedN("V-krow");  // R_i = G_i @ d (d fp32 in cb_t4b, read at tf32)
        WAIT(cb_t4b, 1);
        cb_reserve_back(cb_qk, Kt);
        pack_reconfig_data_format(cb_qk);
        reconfig_data_format(cb_t4b, cb_ones);  // matmul: in0 (G_i) -> srcB, in1 (d) -> srcA
        gdn::mm_init(cb_ones, cb_t4b, 0);
        for (uint32_t i0 = 0; i0 < Kt; i0 += DST) {
            const uint32_t m = (Kt - i0 < DST) ? (Kt - i0) : DST;
            tile_regs_acquire();
            for (uint32_t d = 0; d < m; d++) {
                matmul_tiles(cb_ones, cb_t4b, KDA_GI + i0 + d, 0, d);
            }
            pack_group(m, cb_qk);
        }
        cb_push_back(cb_qk, Kt);
        POP(cb_t4b, 1);
        kda_rebase(cb_t4b, 1);
        WAIT(cb_qk, Kt);
    }
}

// KDA: S_dec = D_i * S on the SFPU for M state tiles in Dst 1 .. M, with D_i in
// Dst 0. Load D_i's vector once per face for each pair of vectors and reuse it
// across all M tiles. SFPU vectors 2j and 2j + 1 cover rows 4j .. 4j + 3 in
// alternating columns; D_i is constant along each row. This uses 4 + 24 M
// instructions per face instead of mul_binary_tile's 32 M, which loads both operands.
#ifdef TRISC_MATH
template <int M>
inline void kda_rows_face(const uint dst_d, const uint dst_s, const uint) {
    constexpr uint T = 32;  // SFPI address stride per Dst tile
#pragma GCC unroll 4
    for (int j = 0; j < 4; j++) {
        const sfpi::vFloat dv = sfpi::dst_reg[dst_d * T];
#pragma GCC unroll 3
        for (int m = 0; m < M; m++) {
            sfpi::dst_reg[(dst_s + m) * T] = sfpi::dst_reg[(dst_s + m) * T] * dv;
        }
        sfpi::dst_reg++;
#pragma GCC unroll 3
        for (int m = 0; m < M; m++) {
            sfpi::dst_reg[(dst_s + m) * T] = sfpi::dst_reg[(dst_s + m) * T] * dv;
        }
        sfpi::dst_reg++;
    }
}
#endif
ALWI void kda_mul_rows(uint32_t m) {
    if (m >= 3) {
        MATH((GDN_BPARAMS((kda_rows_face<3>), 0, 1, 1, RC)));
    } else if (m == 2) {
        MATH((GDN_BPARAMS((kda_rows_face<2>), 0, 1, 1, RC)));
    } else {
        MATH((GDN_BPARAMS((kda_rows_face<1>), 0, 1, 1, RC)));
    }
}

// KDA, one item: row_f32 (F32S 2) with the decay per key channel. k S_dec = (k * d) S (d in row form, cb_qk; the FPU
// at tf32 as F32S's k S); S' = D_i * S (the SFPU per state tile, D_i from cb_dec, groups within a row tile) + k^T v_new
// accumulated onto it by the FPU; o = q S'.
template <bool gated>
__attribute__((noinline)) void row_kda(uint32_t Kt, uint32_t Vt, uint32_t s_in, uint32_t s_mm, bool carry, bool next,
                                       uint32_t upi) {
    const uint32_t kv = Kt * Vt;
    constexpr uint32_t SD = DST - 1;
    {
        DeviceZoneScopedN("V-wait-in");
        WAIT(cb_q, Kt);
        WAIT(cb_k, Kt);
        WAIT(cb_v, Vt);
        WAIT(s_in, kv);
        WAIT(s_mm, kv);
    }
    WAIT(cb_bt, 1);
    {
        DeviceZoneScopedN("V-kS");
        ew(cb_k, cb_qk, cb_t4c, Kt, MUL);  // k * d (row 0)
        WAIT(cb_t4c, Kt);
        POP(cb_qk, Kt);
        mm(cb_t4c, s_mm, cb_t4a, 1, Kt, Vt, false);  // (k * d) @ S = k S_dec
        WAIT(cb_t4a, Vt);
        POP(cb_t4c, Kt);
        POP(s_mm, kv);
    }
    {
        DeviceZoneScopedN("V-vnew");
        ew(cb_v, cb_t4a, cb_t4b, Vt, SUB);  // v - k S_dec
        WAIT(cb_t4b, Vt);
        POP(cb_t4a, Vt);
        POP(cb_v, Vt);
        bcast_scalar_mul(cb_t4b, cb_bt, cb_dl, Vt);  // v_new = beta * (...)
        WAIT(cb_dl, Vt);
        POP(cb_t4b, Vt);
        if (next) {
            shift_up(cb_bt, upi);
        } else {
            POP(cb_bt, 1);
        }
    }
    {
        DeviceZoneScopedN("V-kT");
        transpose_block(cb_k, cb_kt, 1, Kt);
        WAIT(cb_kt, Kt);
    }
    {
        DeviceZoneScopedN("V-Snew");  // S' = D_i S + k^T v_new in Dst at fp32
        WAIT(cb_dec, Kt);
        cb_reserve_back(cb_Sout, kv);
        cb_reserve_back(cb_sd, kv);
        pack_reconfig_data_format(cb_Sout);
        for (uint32_t t0 = 0; t0 < kv; ) {
            const uint32_t i = t0 / Vt;  // the row tile
            const uint32_t left = (i + 1) * Vt - t0;
            const uint32_t m = (left < SD) ? left : SD;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_dec);
            gdn::copy_init(cb_dec);
            copy_tile(cb_dec, i, 0);  // D_i at fp32 (unpack-to-dest)
            reconfig_data_format_srca(s_in);
            gdn::copy_init(s_in);
            for (uint32_t d = 0; d < m; d++) {
                copy_tile(s_in, d, 1 + d);
            }
            POP(s_in, m);
            if (carry) {
                cb_reserve_back(cb_Sc, m);
            }
            mul_dec_init();
            kda_mul_rows(m);  // S_dec, fp32 (m <= SD = DST - 1 = 3)
            wait_sfpu();
            reconfig_data_format(cb_dl, cb_kt);
            gdn::mm_init(cb_kt, cb_dl, 0);
            for (uint32_t d = 0; d < m; d++) {
                const uint32_t t = t0 + d;
                matmul_tiles(cb_kt, cb_dl, t / Vt, t % Vt, 1 + d);  // += k^T v_new
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(1 + d, cb_Sout);
            }
            if (carry) {
                for (uint32_t d = 0; d < m; d++) {
                    pack_tile(1 + d, cb_Sc);
                }
            }
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(1 + d, cb_sd);
            }
            tile_regs_release();
            if (carry) {
                cb_push_back(cb_Sc, m);
            }
            t0 += m;
        }
        cb_push_back(cb_Sout, kv);
        cb_push_back(cb_sd, kv);
        POP(cb_dec, Kt);
        POP(cb_kt, Kt);
    }
    {
        DeviceZoneScopedN("V-qS");
        WAIT(cb_sd, kv);
        if constexpr (gated) {
            mm(cb_q, cb_sd, cb_orow, 1, Kt, Vt, false);
        } else if constexpr (CLIP == 0) {
            mm(cb_q, cb_sd, cb_out, 1, Kt, Vt, false);
        } else {
            mm(cb_q, cb_sd, cb_t4a, 1, Kt, Vt, false);
            WAIT(cb_t4a, Vt);
            out_f32(Vt);
            POP(cb_t4a, Vt);
        }
        if (!carry) {
            POP(cb_sd, kv);
        }
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        POP(cb_dl, Vt);
    }
}
#endif

// One item (one row of one head), its q and k normalized already (cb_q, cb_k from the reader): the
// state from s_in (cb_S0 for the head's first row, cb_Sc after) and, unless it is the head's last
// row (carry), the new state also into cb_Sc; o into cb_orow (GNORM, for the writer) or cb_out.
template <bool raw, bool gated>
__attribute__((noinline)) void row(uint32_t Kt, uint32_t Vt, uint32_t s_in, bool carry, bool next, uint32_t upi) {
    const uint32_t kv = Kt * Vt;
    {
        DeviceZoneScopedN("V-wait-in");
        WAIT(cb_q, Kt);
        WAIT(cb_k, Kt);
        WAIT(cb_v, Vt);
        WAIT(s_in, kv);
    }
    WAIT(cb_dec, 1);
    WAIT(cb_bt, 1);
    if constexpr (F32S == 1) {
        DeviceZoneScopedN("V-Sdec");
        sdec_f32(s_in, kv, next, upi);  // S_dec at fp32 -> cb_sd and cb_t16
        WAIT(cb_sd, kv);
    } else {
        DeviceZoneScopedN("V-Sdec");
        bcast_scalar_mul(s_in, cb_dec, cb_sd, kv);  // S_dec = exp(g) * S
        WAIT(cb_sd, kv);
        POP(s_in, kv);
        if (next) {
            shift_up(cb_dec, upi);
        } else {
            POP(cb_dec, 1);
        }
    }
    {
        DeviceZoneScopedN("V-kS");
        mm(cb_k, cb_sd, cb_t4a, 1, Kt, Vt, false);  // k @ S_dec
        WAIT(cb_t4a, Vt);
    }
    {
        DeviceZoneScopedN("V-vnew");
        ew(cb_v, cb_t4a, cb_t4b, Vt, SUB);  // v - k @ S_dec
        WAIT(cb_t4b, Vt);
        POP(cb_t4a, Vt);
        POP(cb_v, Vt);
        bcast_scalar_mul(cb_t4b, cb_bt, cb_dl, Vt);  // v_new = beta * (...)
        WAIT(cb_dl, Vt);
        POP(cb_t4b, Vt);
        if (next) {
            shift_up(cb_bt, upi);
        } else {
            POP(cb_bt, 1);
        }
    }
    {
        DeviceZoneScopedN("V-kT");
        transpose_block(cb_k, cb_kt, 1, Kt);  // k^T: Kt tiles of one column
        WAIT(cb_kt, Kt);
    }
    if constexpr (F32S == 1) {
        DeviceZoneScopedN("V-Snew");
        snew_f32(Vt, kv, carry);  // S' = S_dec + k^T v_new at fp32 -> the writer (and the next row)
    } else {
        {
            DeviceZoneScopedN("V-outer");
            mm(cb_kt, cb_dl, cb_t16, Kt, 1, Vt, false);  // k^T v_new (outer product)
            WAIT(cb_t16, kv);
        }
        {
            DeviceZoneScopedN("V-Sadd");
            if (carry) {
                add_twice(cb_sd, cb_t16, cb_Sout, cb_Sc, kv);  // S' -> the writer and the next row
            } else {
                ew(cb_sd, cb_t16, cb_Sout, kv, ADD);  // S' -> the writer
            }
            POP(cb_t16, kv);
        }
    }
    {
        DeviceZoneScopedN("V-qS");
        mm(cb_q, cb_sd, cb_t4a, 1, Kt, Vt, false);  // q @ S_dec
        WAIT(cb_t4a, Vt);
        POP(cb_sd, kv);
    }
    {
        DeviceZoneScopedN("V-qk");
        mm(cb_q, cb_kt, cb_qk, 1, Kt, 1, false);  // q . k in [0, 0]
        WAIT(cb_qk, 1);
        POP(cb_kt, Kt);
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        bcast_scalar_mul(cb_dl, cb_qk, cb_t4b, Vt);  // (q . k) v_new
        WAIT(cb_t4b, Vt);
        POP(cb_qk, 1);
        POP(cb_dl, Vt);
    }
    if constexpr (gated) {
        DeviceZoneScopedN("V-oadd");
        ew(cb_t4a, cb_t4b, cb_orow, Vt, ADD);  // o -> the writer, which gathers the group's rows
        POP(cb_t4a, Vt);
        POP(cb_t4b, Vt);
    } else {
        ew_out(cb_t4a, cb_t4b, cb_out, Vt, ADD);  // the output, clipped (CLIP)
        POP(cb_t4a, Vt);
        POP(cb_t4b, Vt);
    }
}


// SSM, one item (one row of one head; C and B in cb_q and cb_k, x in cb_v): the state update in Dst at
// fp32 (see the head of this file), then o = C S' + D x. s_in, cb_dfull: unpack-to-dest (gdn_verify.py).
__attribute__((noinline)) void row_ssm(uint32_t Kt, uint32_t Vt, uint32_t s_in, bool carry, bool next, uint32_t upi) {
    const uint32_t kv = Kt * Vt;
    constexpr uint32_t SD = DST - 1;  // state tiles per acquire, after the decay's tile in Dst 0
    static_assert(SSM == 0 || DST >= 2, "SSM: DST >= 2");
    {
        DeviceZoneScopedN("V-wait-in");
        WAIT(cb_q, Kt);
        WAIT(cb_k, Kt);
        WAIT(cb_v, Vt);
        WAIT(s_in, kv);
    }
    WAIT(cb_dec, 1);
    WAIT(cb_bt, 1);
    {
        DeviceZoneScopedN("V-dec");
        cb_reserve_back(cb_dfull, 1);  // exp(g) over a whole tile: ones * g[0, 0] (FPU), exp (SFPU)
        pack_reconfig_data_format(cb_dfull);
        reconfig_data_format(cb_ones, cb_dec);
        gdn::mul_bcast_scalar_init(cb_ones, cb_dec);
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_ones, cb_dec, 0, 0, 0);
        exp_tile_init();
        exp_tile(0);
        pack_group(1, cb_dfull);
        cb_push_back(cb_dfull, 1);
        if (next) {
            shift_up(cb_dec, upi);
        } else {
            POP(cb_dec, 1);
        }
    }
    {
        DeviceZoneScopedN("V-vnew");
        WAIT(cb_t1b, 1);
        bcast_scalar_mul(cb_v, cb_bt, cb_dl, Vt);  // v_new = dt x
        bcast_scalar_mul(cb_v, cb_t1b, cb_t4c, Vt);  // D x, for the output
        WAIT(cb_dl, Vt);
        WAIT(cb_t4c, Vt);
        POP(cb_v, Vt);
        if (next) {
            shift_up(cb_bt, upi);
            shift_up(cb_t1b, upi);
        } else {
            POP(cb_bt, 1);
            POP(cb_t1b, 1);
        }
    }
    {
        DeviceZoneScopedN("V-kT");
        transpose_block(cb_k, cb_kt, 1, Kt);  // B^T: Kt tiles of one column
        WAIT(cb_kt, Kt);
    }
    {
        DeviceZoneScopedN("V-Snew");  // S' = exp(g) S + B^T v_new, SD tiles a Dst acquire
        WAIT(cb_dfull, 1);
        cb_reserve_back(cb_Sout, kv);
        cb_reserve_back(cb_sd, kv);
        pack_reconfig_data_format(cb_Sout);  // Sout, Sc and sd: the state's format (gdn_verify.py)
        // s_in is consumed SD tiles at a time: after the head's first row it is cb_Sc itself, whose kv
        // tiles this row's S' refills, so each chunk's tiles are popped before their place is reserved.
        for (uint32_t t0 = 0; t0 < kv; t0 += SD) {
            const uint32_t m = (kv - t0 < SD) ? (kv - t0) : SD;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_dfull);
            gdn::copy_init(cb_dfull);
            copy_tile(cb_dfull, 0, 0);
            reconfig_data_format_srca(s_in);
            gdn::copy_init(s_in);
            for (uint32_t d = 0; d < m; d++) {
                copy_tile(s_in, d, 1 + d);  // the chunk at the front of s_in
            }
            POP(s_in, m);
            if (carry) {
                cb_reserve_back(cb_Sc, m);
            }
            mul_binary_tile_init();
            for (uint32_t d = 0; d < m; d++) {
                mul_binary_tile(1 + d, 0, 1 + d);  // S_dec, fp32
            }
// Wait for the SFPU before the FPU matmul accumulates into the same Dst tiles.
// The LLK does not insert this dependency between the two operations.
            wait_sfpu();
            reconfig_data_format(cb_dl, cb_kt);  // matmul: in0 -> srcB, in1 -> srcA
            gdn::mm_init(cb_kt, cb_dl, 0);
            for (uint32_t d = 0; d < m; d++) {
                const uint32_t t = t0 + d;
                matmul_tiles(cb_kt, cb_dl, t / Vt, t % Vt, 1 + d);  // += B^T v_new, tile (t / Vt, t % Vt)
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(1 + d, cb_Sout);
            }
            if (carry) {
                for (uint32_t d = 0; d < m; d++) {
                    pack_tile(1 + d, cb_Sc);
                }
            }
            for (uint32_t d = 0; d < m; d++) {
                pack_tile(1 + d, cb_sd);
            }
            tile_regs_release();
            if (carry) {
                cb_push_back(cb_Sc, m);
            }
        }
        cb_push_back(cb_Sout, kv);
        cb_push_back(cb_sd, kv);
        POP(cb_dfull, 1);
        POP(cb_kt, Kt);
    }
    {
        DeviceZoneScopedN("V-qS");
        WAIT(cb_sd, kv);
        mm(cb_q, cb_sd, cb_t4a, 1, Kt, Vt, false);  // C S'
        WAIT(cb_t4a, Vt);
        POP(cb_sd, kv);
        POP(cb_q, Kt);
        POP(cb_k, Kt);
        POP(cb_dl, Vt);
    }
    {
        DeviceZoneScopedN("V-oadd");
        ew_out(cb_t4a, cb_t4c, cb_out, Vt, ADD);  // + D x: the output, clipped (CLIP)
        POP(cb_t4a, Vt);
        POP(cb_t4c, Vt);
    }
}

#ifdef GDN_FOLD
// Fused convolution (GDN_FOLD, enabled by gdn_verify.py with conv=...): causal conv1d for
// R tokens, on the window the reader packed into cb_qg (GDN_FOLD_WP tiles, window row w in tile w, row c = this core's
// channel tile c) with the packed taps in cb_kg (w0, w1, w2, w3, bias; GDN_FOLD_WT tiles), one bf16 tile a token into
// cb_co for the reader, which gathers the q, k and v rows from it. gdn_conv_compute.cpp's token() op for op: the same
// operands in the same order, the intermediates in fp32 one-tile CBs (here cb_t1a, cb_t1b and cb_qk, which nothing
// holds before the first group's norms), the window, the taps and the output in bf16, so every element is gdn_conv's
// bit for bit:
//   out_t = silu(((w3 * w(t+3) + w0 * w(t)) + w1 * w(t+1)) + w2 * w(t+2) + bias)
constexpr uint32_t cb_co = 28;  // the conv's output (bf16, R tiles; the writer's staging moved into cb_rbuf)
constexpr uint32_t cb_fa = cb_t1a, cb_fb = cb_t1b, cb_fc = cb_qk;
constexpr int F_ADD = 0, F_MUL = 2;

// o = a[ia] (op) b[ib], one tile (the inputs available; o pushed): gdn_conv_compute.cpp's binop.
__attribute__((noinline)) void fold_binop(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o, int op) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == F_MUL) {
        gdn::mul_init(a, b);
    } else {
        gdn::add_init(a, b);
    }
    tile_regs_acquire();
    if (op == F_MUL) {
        mul_tiles(a, b, ia, ib, 0);
    } else {
        add_tiles(a, b, ia, ib, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
    cb_push_back(o, 1);
}

// silu(in[0]) -> cb_co: gdn_conv_compute.cpp's silu_out.
__attribute__((noinline)) void fold_silu(uint32_t in) {
    cb_reserve_back(cb_co, 1);
    pack_reconfig_data_format(cb_co);
    reconfig_data_format_srca(in);
    gdn::copy_init(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_co);
    tile_regs_release();
    cb_push_back(cb_co, 1);
}

#ifndef GDN_FOLD_SKIP
#define GDN_FOLD_SKIP 0  // Diagnostic skip bit 4 pushes output tiles without computing them; results are invalid.
#endif
template <uint32_t R>
__attribute__((noinline)) void fold_conv() {
    constexpr uint32_t cb_in = cb_qg, cb_w = cb_kg;
    cb_wait_front(cb_in, GDN_FOLD_WP);
    cb_wait_front(cb_w, GDN_FOLD_WT);
    if constexpr ((GDN_FOLD_SKIP & 4) != 0) {
        cb_reserve_back(cb_co, R);
        cb_push_back(cb_co, R);
    }
    for (uint32_t t = 0; t < R && !(GDN_FOLD_SKIP & 4); t++) {
        DeviceZoneScopedN("V-ftok");
        fold_binop(cb_w, 3, cb_in, t + 3, cb_fa, F_MUL);  // w3 * x_t
        fold_binop(cb_w, 0, cb_in, t, cb_fb, F_MUL);      // w0 * w(t)
        cb_wait_front(cb_fa, 1);
        cb_wait_front(cb_fb, 1);
        fold_binop(cb_fa, 0, cb_fb, 0, cb_fc, F_ADD);
        gdn::pop(cb_fa, 1);
        gdn::pop(cb_fb, 1);
        fold_binop(cb_w, 1, cb_in, t + 1, cb_fa, F_MUL);  // w1 * w(t+1)
        cb_wait_front(cb_fc, 1);
        cb_wait_front(cb_fa, 1);
        fold_binop(cb_fc, 0, cb_fa, 0, cb_fb, F_ADD);
        gdn::pop(cb_fc, 1);
        gdn::pop(cb_fa, 1);
        fold_binop(cb_w, 2, cb_in, t + 2, cb_fa, F_MUL);  // w2 * w(t+2)
        cb_wait_front(cb_fb, 1);
        cb_wait_front(cb_fa, 1);
        fold_binop(cb_fb, 0, cb_fa, 0, cb_fc, F_ADD);
        gdn::pop(cb_fb, 1);
        gdn::pop(cb_fa, 1);
        cb_wait_front(cb_fc, 1);
        uint32_t acc = cb_fc;
#if GDN_FOLD_BIAS
        fold_binop(cb_fc, 0, cb_w, 4, cb_fa, F_ADD);
        gdn::pop(cb_fc, 1);
        cb_wait_front(cb_fa, 1);
        acc = cb_fa;
#endif
        fold_silu(acc);
        gdn::pop(acc, 1);
    }
    gdn::pop(cb_in, GDN_FOLD_WP);
    gdn::pop(cb_w, GDN_FOLD_WT);
}
#endif
}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t RAW = get_compile_time_arg_val(2);
    constexpr uint32_t R = get_compile_time_arg_val(3);
    constexpr uint32_t GNORM = get_compile_time_arg_val(5);
    static_assert(SSM == 0 || GNORM == 0, "SSM: no gated norm in the kernel");
    constexpr uint32_t GH = GATES / R;  // heads per group: its R GH items fill one tile's rows
    const uint32_t nh = get_arg_val<uint32_t>(0);
    const uint32_t scale = get_arg_val<uint32_t>(1);
    const uint32_t eps = get_arg_val<uint32_t>(2);
    const uint32_t norm_eps = get_arg_val<uint32_t>(3);
    const uint32_t inv_v = get_arg_val<uint32_t>(4);

    compute_kernel_hw_startup(cb_v, cb_gab, cb_out);

    const uint32_t upi = UPI;
    for (uint32_t h0 = 0; h0 < nh; h0 += GH) {
        const uint32_t items = ((nh - h0 < GH) ? (nh - h0) : GH) * R;
        {
            DeviceZoneScopedN("V-gates");
            gates(RAW != 0);
        }
#ifdef GDN_FOLD
        if (h0 == 0) {  // the conv, once a call: after the first group's gates, before its norms
            DeviceZoneScopedN("V-fold");
            fold_conv<R>();
        }
#endif
    // Constants are pushed once and never popped. Norms wait only for the ones tile
    // used in row sums. Wait for L and the weight afterward; the reader sends them
    // after gathering rows.
        WAIT(cb_ones, 1);
        {
            DeviceZoneScopedN("V-l2norm");
            WAIT(cb_qg, Kt);
            WAIT(cb_kg, Kt);
            if constexpr (SSM) {  // C and B as they are
                copy_block(cb_qg, cb_qn, Kt);
                copy_block(cb_kg, cb_kn, Kt);
            } else {
                l2norm1(cb_qg, cb_qn, Kt, eps, scale, true);  // -> the reader
                l2norm1(cb_kg, cb_kn, Kt, eps, scale, false);
            }
            POP(cb_qg, Kt);
            POP(cb_kg, Kt);
        }
        WAIT(cb_ones, 2 + (GNORM ? Vt : 0));
        for (uint32_t j = 0; j < items; j++) {
            const uint32_t r = j % R;
            DeviceZoneScopedN("V-row");
            if constexpr (SSM) {
                row_ssm(Kt, Vt, r == 0 ? cb_S0 : cb_Sc, r + 1 < R, j + 1 < items, upi);
            } else if constexpr (KDA) {
#ifdef GDN_KDA
                kda_decay(Kt, get_arg_val<uint32_t>(5 + h0 + j / R));  // exp(A_log) of the core's head h0 + j / R
                row_kda<GNORM != 0>(Kt, Vt, r == 0 ? cb_S0 : cb_Sc, r == 0 ? cb_t16 : cb_sd, r + 1 < R, j + 1 < items, upi);
#endif
            } else if constexpr (F32S == 2) {
                row_f32<GNORM != 0>(Kt, Vt, r == 0 ? cb_S0 : cb_Sc, r == 0 ? cb_t16 : cb_sd, r + 1 < R, j + 1 < items, upi);
            } else {
                row<RAW != 0, GNORM != 0>(Kt, Vt, r == 0 ? cb_S0 : cb_Sc, r + 1 < R, j + 1 < items, upi);
            }
        }
        if constexpr (GNORM != 0) {
            DeviceZoneScopedN("V-gnorm");
            gnorm_group(Vt, norm_eps, inv_v);
        }
    }
}
