// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// moe8.route (moe8.py): the router's top k and its weights for R rows on one core, from the
// router's float32 logits [1, 1, R, E] (TILE) into moe8's ids [1, 1, R, k] (uint16, or uint32 with
// IDS32; TILE) and weights [1, 1, R, k] (bf16 TILE), each written as one whole page.
// - The pick compares the logits' fp32 bits (exact: no bf16 ties); among equal logits the lower id.
//   Slots in decreasing order of logit.
// - KIND 0 (the softmax over all experts, top k, renormalized, which is the softmax over the k):
//   the pick by logit; the weights exp(l - lmax) for each pick, their sum, the quotients.
// - KIND 1 (sigmoid scores, the pick by score + bias, the scores renormalized): every
//   expert's score bounded from a table (sigmoid at steps of 1/16), the candidates (an upper bound at
//   or above the k-th largest lower bound) scored exactly, the pick among them; the weights the
//   picks' scores over their sum.
// - KIND 2 (the sigmoid kind without a bias): the pick by logit as KIND 0
//   (sigmoid is increasing), the weights the picks' sigmoids over their sum.
// SHARED: the logits' column E (the next one after the experts: a shared expert's gate folded
// into the router's matmul) as sigmoid(logit) in bf16 at row r, column 0 of a [1, 1, R, 1] tile.
// FLAGS:
// - NORM (bit 0): the weights over their sum (the kinds above). Without it (the sigmoid kinds
//   only), each weight is the pick's own sigmoid.
// - SCALE: every weight times a constant (Q2.30, 2^30 for none: a routed scaling factor),
//   folded into the one division a row.
// - GROUPS (bit 2, KIND 1 only; DeepSeek-V3's group-limited routing): the experts in NG groups
//   of E / NG in order; a group's score the sum of its two largest keys (score + bias); the TG best
//   groups kept (the lower group among equal scores), the rest masked; the top k of the kept by key.
//   Every expert scored exactly (no table bounds).
// - SLOT (bit 1): slot K of each row gets id SLOT_ID and weight 1.0 (a shared expert without a
//   gate as moe8's slot k + 1).
// All in fixed point with integers only (this RISC-V has no FPU); each weight rounded to bf16 at the
// end (round to nearest even).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_sigmoid_q30.h"

constexpr uint32_t cb_l = 0, cb_i = 1, cb_w = 2, cb_g = 3;

// fp32 bits -> an unsigned key in the floats' order (NaNs aside)
inline uint32_t order_key(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }

// fp32 bits -> Q32.32 (|x| < 2^31; denormals and |x| < 2^-56 -> 0)
inline int64_t to_q32(uint32_t b) {
    const int32_t e = static_cast<int32_t>((b >> 23) & 0xFF);
    if (e == 0) {
        return 0;
    }
    const int64_t m = static_cast<int64_t>((b & 0x7FFFFFu) | 0x800000u);
    const int32_t sh = e - 118;  // x = m * 2^(e - 150); Q32.32: m * 2^(e - 118)
    int64_t v;
    if (sh >= 39) {
        v = INT64_MAX;
    } else if (sh >= 0) {
        v = m << sh;
    } else if (sh > -64) {
        v = m >> (-sh);
    } else {
        v = 0;
    }
    return (b & 0x80000000u) ? -v : v;
}

// exp(d) for d <= 0 given in Q32.32, as Q2.30 (0 for d < -64): 2^(d log2 e), the fraction by a
// degree-9 polynomial (truncation under 1e-8 relative)
inline uint64_t exp_q30(int64_t d) {
    if (d < -(static_cast<int64_t>(64) << 32)) {
        return 0;
    }
    const int32_t d24 = static_cast<int32_t>(d >> 8);  // Q8.24, in [-2^30, 0]
    constexpr int64_t LOG2E = 1549082005;               // log2(e) in Q2.30
    const int32_t t = static_cast<int32_t>((static_cast<int64_t>(d24) * LOG2E) >> 30);  // Q8.24
    const int32_t n = t >> 24;                          // floor
    const int64_t f = static_cast<int64_t>(t - (n << 24)) << 6;  // Q0.30 in [0, 1)
    // 2^f = sum ln2^j / j! f^j, coefficients in Q2.30
    constexpr int64_t C[10] = {1073741824, 744261118, 257941248, 59597083, 10327387,
                               1431680,    165394,    16377,     1419,     109};
    int64_t p = C[9];
    for (int j = 8; j >= 0; --j) {
        p = C[j] + ((p * f) >> 30);
    }
    const int32_t sh = -n;
    return sh >= 32 ? 0 : static_cast<uint64_t>(p) >> sh;
}

// n / d for 64-bit unsigned integers by shift and subtract (no libgcc division here)
inline uint64_t udiv64(uint64_t n, uint64_t d) {
    uint64_t q = 0, r = 0;
    for (int b = 63; b >= 0; --b) {
        r = (r << 1) | ((n >> b) & 1);
        if (r >= d) {
            r -= d;
            q |= static_cast<uint64_t>(1) << b;
        }
    }
    return q;
}

// 1 / y for y in Q2.30 within [1, 2], as Q2.30: Newton from the linear guess 24/17 - 8/17 y (the
// classic 48/17 - 32/17 d for d = y / 2 in [1/2, 1], halved; relative error under 1/17), three
// steps (each squares the error: under 2^-30)
inline uint64_t recip_q30(uint64_t y) {
    constexpr int64_t A = 1515870810;  // 24/17 in Q2.30
    constexpr int64_t B = 505290270;   // 8/17 in Q2.30
    int64_t r = A - ((B * static_cast<int64_t>(y)) >> 30);
    for (int i = 0; i < 3; ++i) {
        const int64_t e = (static_cast<int64_t>(1) << 31) - ((static_cast<int64_t>(y) * r) >> 30);
        r = (r * e) >> 30;  // r (2 - y r)
    }
    return static_cast<uint64_t>(r);
}

// sigmoid(x) for x in Q32.32, as Q2.30: 1 / (1 + exp(-|x|)), mirrored for x < 0
inline uint64_t sigmoid_q30(int64_t x) {
    const uint64_t u = exp_q30(x >= 0 ? -x : x);  // exp(-|x|) in (0, 1]
    const uint64_t r = recip_q30((static_cast<uint64_t>(1) << 30) + u);
    return x >= 0 ? r : (u * r) >> 30;  // 1 / (1 + u), or u / (1 + u)
}

// the table's bounds of sigmoid(x), x in Q32.32: (lower, upper) in Q2.30
inline void sigmoid_bounds(int64_t x, uint32_t& lo, uint32_t& hi) {
    const int64_t i = (x + (static_cast<int64_t>(16) << 32)) >> 28;  // floor((x + 16) * 16)
    if (i < 0) {
        lo = 0;
        hi = SIG_Q30[0];
    } else if (i >= 512) {
        lo = SIG_Q30[512];
        hi = 1u << 30;
    } else {
        lo = SIG_Q30[i];
        hi = SIG_Q30[i + 1];
    }
    // the table is rounded and sigmoid_q30 carries its own small error: widen by 64 units each way
    lo = lo > 64 ? lo - 64 : 0;
    hi = hi + 64;
}

// w in Q0.30 (0 .. 2^30) -> bf16 bits, round to nearest even
inline uint16_t q30_to_bf16(uint64_t w) {
    if (w == 0) {
        return 0;
    }
    int32_t p = 63;
    while (!((w >> p) & 1)) {
        --p;
    }
    uint32_t ex = static_cast<uint32_t>(p + 97);  // (p - 30) + 127
    uint32_t mant;
    if (p >= 7) {
        const int32_t s = p - 7;
        mant = static_cast<uint32_t>(w >> s) & 0x7F;
        const uint64_t rem = w & ((static_cast<uint64_t>(1) << s) - 1);
        const uint64_t half = s > 0 ? static_cast<uint64_t>(1) << (s - 1) : 0;
        if (s > 0 && (rem > half || (rem == half && (mant & 1)))) {
            if (++mant == 0x80) {
                mant = 0;
                ++ex;
            }
        }
    } else {
        mant = static_cast<uint32_t>(w << (7 - p)) & 0x7F;
    }
    return static_cast<uint16_t>((ex << 7) | mant);
}

// the byte offset of element (r, c) (c < 16) in a 32 x 32 tile of es-byte elements
inline uint32_t tile_off(uint32_t r, uint32_t c, uint32_t es) {
    return ((r >> 4) * 2) * 256 * es + (r & 15) * 16 * es + c * es;
}

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr bool IDS32 = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t KIND = get_compile_time_arg_val(4);  // 0 softmax, 1 sigmoid + bias, 2 sigmoid
    constexpr bool SHARED = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t FLAGS = get_compile_time_arg_val(6);
    constexpr bool NORM = (FLAGS & 1) != 0;
    constexpr bool SLOT = (FLAGS & 2) != 0;
    constexpr bool GROUPS = (FLAGS & 4) != 0;
    constexpr uint32_t SCALE = get_compile_time_arg_val(7);  // the weights' factor, Q2.30 (2^30: none)
    constexpr uint32_t SLOT_ID = get_compile_time_arg_val(8);
    constexpr uint32_t NG = get_compile_time_arg_val(9);  // GROUPS: groups, and those kept
    constexpr uint32_t TG = get_compile_time_arg_val(10);
    constexpr auto l_args = TensorAccessorArgs<11>();
    constexpr auto i_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    static_assert(E % 32 == 0 && K >= 1 && K <= 16 && R >= 1 && R <= 32, "moe8.route: shapes");
    static_assert(KIND <= 2, "moe8.route: kind");
    static_assert(NORM || KIND != 0, "moe8.route: the softmax kind is normalized");
    static_assert(K + (SLOT ? 1 : 0) <= 16, "moe8.route: k + the fixed slot");
    static_assert(!GROUPS || (KIND == 1 && NG >= 1 && NG <= 32 && E % NG == 0 && E / NG >= 2 && TG >= 1 &&
                              TG <= NG && TG * (E / NG) >= K),
                  "moe8.route: groups");
    constexpr uint32_t ET = E / 32;
    constexpr uint32_t ES = IDS32 ? 4 : 2;
    constexpr uint32_t IPAGE = 1024 * ES;

    const uint32_t l_addr = get_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const auto l_acc = TensorAccessor(l_args, l_addr, 4096);
    const auto i_acc = TensorAccessor(i_args, i_addr, IPAGE);
    const auto w_acc = TensorAccessor(w_args, w_addr, 2048);
    const auto b_acc = TensorAccessor(b_args, b_addr, 4096);
    const auto g_acc = TensorAccessor(g_args, g_addr, 2048);

    // the logits' rows: row r, columns 0-15 and 16-31 of each tile, 64 bytes each
    cb_reserve_back(cb_l, 1);
    const uint32_t lv = get_write_ptr(cb_l);  // [R][E] fp32 bits
    for (uint32_t t = 0; t < ET; ++t) {
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t src = ((r >> 4) * 2) * 1024 + (r & 15) * 64;
            const uint32_t dst = lv + (r * E + t * 32) * 4;
            noc_async_read(l_acc.get_noc_addr(t, src), dst, 64);
            noc_async_read(l_acc.get_noc_addr(t, src + 1024), dst + 64, 64);
        }
    }
    // KIND 1: the bias's row 0, after the logits
    const uint32_t lb = lv + R * E * 4;
    // SHARED: each row's columns E .. E + 15 (64 bytes), after the sigmoid kind's scratch
    const uint32_t lsg = lb + E * 4 + (8 + 8 + 4) * E;
    if constexpr (SHARED) {
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t src = ((r >> 4) * 2) * 1024 + (r & 15) * 64;
            noc_async_read(l_acc.get_noc_addr(ET, src), lsg + r * 64, 64);
        }
    }
    if constexpr (KIND == 1) {
        for (uint32_t t = 0; t < ET; ++t) {
            noc_async_read(b_acc.get_noc_addr(t, 0), lb + t * 128, 64);
            noc_async_read(b_acc.get_noc_addr(t, 1024), lb + t * 128 + 64, 64);
        }
    }
    // the output pages, zero but for the picks
    cb_reserve_back(cb_i, 1);
    cb_reserve_back(cb_w, 1);
    const uint32_t li = get_write_ptr(cb_i), lw = get_write_ptr(cb_w);
    uint32_t lg = 0;
    if constexpr (SHARED) {
        cb_reserve_back(cb_g, 1);
        lg = get_write_ptr(cb_g);
    }
    {
        const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
        if constexpr (SHARED) {
            for (uint32_t off = 0; off < 2048; off += MEM_ZEROS_SIZE) {
                noc_async_read(z, lg + off, 2048 - off < MEM_ZEROS_SIZE ? 2048 - off : MEM_ZEROS_SIZE);
            }
        }
        for (uint32_t off = 0; off < IPAGE; off += MEM_ZEROS_SIZE) {
            noc_async_read(z, li + off, IPAGE - off < MEM_ZEROS_SIZE ? IPAGE - off : MEM_ZEROS_SIZE);
        }
        for (uint32_t off = 0; off < 2048; off += MEM_ZEROS_SIZE) {
            noc_async_read(z, lw + off, 2048 - off < MEM_ZEROS_SIZE ? 2048 - off : MEM_ZEROS_SIZE);
        }
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lv);
    volatile tt_l1_ptr uint32_t* bias = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lb);
    // KIND 1's per-expert scratch after the bias: the bias and the row's x in Q32.32, the upper
    // bounds (Q2.30)
    int64_t* bq = reinterpret_cast<int64_t*>(lb + E * 4);
    int64_t* xq = bq + E;
    uint32_t* hq = reinterpret_cast<uint32_t*>(xq + E);
    if constexpr (KIND == 1) {
        for (uint32_t e = 0; e < E; ++e) {
            bq[e] = to_q32(bias[e]);
        }
    }
    for (uint32_t r = 0; r < R; ++r) {
        uint32_t ti[K];
        uint64_t sc[K];  // the picks' weights before the renormalization, Q2.30
        if constexpr (KIND != 1) {
            uint32_t tk[K];
            for (uint32_t j = 0; j < K; ++j) {
                tk[j] = 0;
                ti[j] = E;
            }
            for (uint32_t e = 0; e < E; ++e) {
                const uint32_t kk = order_key(vals[r * E + e]);
                if (kk <= tk[K - 1]) {
                    continue;  // equal keys: the lower id (seen first) stays
                }
                uint32_t j = K - 1;
                while (j > 0 && kk > tk[j - 1]) {
                    tk[j] = tk[j - 1];
                    ti[j] = ti[j - 1];
                    --j;
                }
                tk[j] = kk;
                ti[j] = e;
            }
            if constexpr (KIND == 0) {
                // exp(l - lmax) over the k
                const int64_t lmax = to_q32(vals[r * E + ti[0]]);
                for (uint32_t j = 0; j < K; ++j) {
                    sc[j] = exp_q30(to_q32(vals[r * E + ti[j]]) - lmax);
                }
            } else {
                // the picks' sigmoids
                for (uint32_t j = 0; j < K; ++j) {
                    sc[j] = sigmoid_q30(to_q32(vals[r * E + ti[j]]));
                }
            }
        } else if constexpr (GROUPS) {
            // every expert's key exactly: (sigmoid << 2) + bias in Q32.32 (kept in xq), its score in hq
            constexpr uint32_t GS = E / NG;
            for (uint32_t e = 0; e < E; ++e) {
                const uint64_t s = sigmoid_q30(to_q32(vals[r * E + e]));
                hq[e] = static_cast<uint32_t>(s);
                xq[e] = (static_cast<int64_t>(s) << 2) + bq[e];
            }
            // each group's score: the sum of its two largest keys
            int64_t gsc[NG];
            for (uint32_t g = 0; g < NG; ++g) {
                int64_t a = INT64_MIN, b = INT64_MIN;
                for (uint32_t e = g * GS; e < (g + 1) * GS; ++e) {
                    const int64_t v = xq[e];
                    if (v > a) {
                        b = a;
                        a = v;
                    } else if (v > b) {
                        b = v;
                    }
                }
                gsc[g] = a + b;
            }
            // the TG best groups (the lower group among equal scores)
            bool keep[NG];
            for (uint32_t g = 0; g < NG; ++g) {
                keep[g] = false;
            }
            for (uint32_t t = 0; t < TG; ++t) {
                uint32_t best = NG;
                for (uint32_t g = 0; g < NG; ++g) {
                    if (!keep[g] && (best == NG || gsc[g] > gsc[best])) {
                        best = g;
                    }
                }
                keep[best] = true;
            }
            // the top k of the kept groups' experts by key (the lower id among equal keys)
            int64_t tk[K];
            for (uint32_t j = 0; j < K; ++j) {
                tk[j] = INT64_MIN;
                ti[j] = E;
                sc[j] = 0;
            }
            for (uint32_t e = 0; e < E; ++e) {
                if (!keep[e / GS]) {
                    continue;
                }
                const int64_t v = xq[e];
                if (ti[K - 1] != E && v <= tk[K - 1]) {
                    continue;
                }
                uint32_t j = K - 1;
                while (j > 0 && (ti[j - 1] == E || v > tk[j - 1])) {
                    tk[j] = tk[j - 1];
                    ti[j] = ti[j - 1];
                    sc[j] = sc[j - 1];
                    --j;
                }
                tk[j] = v;
                ti[j] = e;
                sc[j] = hq[e];
            }
        } else {
            // one pass: each expert's x and bias in Q32.32 (kept), its lower bound of score + bias
            // from the table, the k-th largest of them
            int64_t lk[K];
            for (uint32_t j = 0; j < K; ++j) {
                lk[j] = INT64_MIN;
            }
            for (uint32_t e = 0; e < E; ++e) {
                const int64_t x = to_q32(vals[r * E + e]);
                xq[e] = x;
                uint32_t lo, hi;
                sigmoid_bounds(x, lo, hi);
                hq[e] = hi;
                const int64_t v = (static_cast<int64_t>(lo) << 2) + bq[e];
                if (v <= lk[K - 1]) {
                    continue;
                }
                uint32_t j = K - 1;
                while (j > 0 && v > lk[j - 1]) {
                    lk[j] = lk[j - 1];
                    --j;
                }
                lk[j] = v;
            }
            // the candidates (an upper bound at or above it) scored exactly; the top k of them
            int64_t tk[K];
            for (uint32_t j = 0; j < K; ++j) {
                tk[j] = INT64_MIN;
                ti[j] = E;
                sc[j] = 0;
            }
            for (uint32_t e = 0; e < E; ++e) {
                if ((static_cast<int64_t>(hq[e]) << 2) + bq[e] < lk[K - 1]) {
                    continue;
                }
                const uint64_t s = sigmoid_q30(xq[e]);
                const int64_t v = (static_cast<int64_t>(s) << 2) + bq[e];
                if (v <= tk[K - 1]) {
                    continue;  // equal keys: the lower id stays
                }
                uint32_t j = K - 1;
                while (j > 0 && v > tk[j - 1]) {
                    tk[j] = tk[j - 1];
                    ti[j] = ti[j - 1];
                    sc[j] = sc[j - 1];
                    --j;
                }
                tk[j] = v;
                ti[j] = e;
                sc[j] = s;
            }
        }
        uint64_t sum = 0;
        for (uint32_t j = 0; j < K; ++j) {
            sum += sc[j];
        }
        // NORM: one division a row, SCALE * 2^60 / sum, then products (sum >= the largest weight's
        // scale); 2^60 / sum without a scale
        const uint64_t inv = NORM && sum ? udiv64(static_cast<uint64_t>(SCALE) << 30, sum) : 0;
        for (uint32_t j = 0; j < K; ++j) {
            uint64_t q;
            if constexpr (NORM) {
                q = (sc[j] * inv) >> 30;
            } else if constexpr (SCALE == (1u << 30)) {
                q = sc[j];
            } else {
                q = (sc[j] * SCALE) >> 30;
            }
            const uint32_t o = tile_off(r, j, 2);
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lw + o) = q30_to_bf16(q);
            const uint32_t oi = tile_off(r, j, ES);
            if constexpr (IDS32) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(li + oi) = ti[j];
            } else {
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(li + oi) = static_cast<uint16_t>(ti[j]);
            }
        }
        if constexpr (SLOT) {
            // the fixed slot: SLOT_ID at weight 1.0
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lw + tile_off(r, K, 2)) = 0x3F80;
            if constexpr (IDS32) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(li + tile_off(r, K, ES)) = SLOT_ID;
            } else {
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(li + tile_off(r, K, ES)) =
                    static_cast<uint16_t>(SLOT_ID);
            }
        }
    }
    if constexpr (SHARED) {
        // the shared expert's gate: sigmoid of each row's column E, bf16, at (r, 0)
        volatile tt_l1_ptr uint32_t* sv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lsg);
        for (uint32_t r = 0; r < R; ++r) {
            const uint64_t s = sigmoid_q30(to_q32(sv[r * 16]));
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lg + tile_off(r, 0, 2)) = q30_to_bf16(s);
        }
        noc_async_write(lg, g_acc.get_noc_addr(0), 2048);
    }
    noc_async_write(li, i_acc.get_noc_addr(0), IPAGE);
    noc_async_write(lw, w_acc.get_noc_addr(0), 2048);
    noc_async_write_barrier();
}
