// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// Copyright 2026 Xylolabs
// Modified by Xylolabs, 2026: SFPU functions (gelu_tanh, sigmoid, bf16 round, clamped SwiGLU) and row helpers
// written in the form of upstream's calculate_silu and silu_rows / mul_rows. Adapted from tenstorrent/tt-metal
// v0.81.1, tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h and
// models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_compute.cpp.
// SPDX-License-Identifier: Apache-2.0
//
// dense_ops.h: compute-kernel helpers shared by the kernels of this repository, for tt-metal's compute API (v0.81).
//
// - Init wrappers for the matmul, the eltwise ops and the row-broadcast multiply.
// - SFPU functions in the form of tt-metal's calculate_silu (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/
//   ckernel_sfpu_silu.h, Apache-2.0; see NOTICE): gelu_tanh through the sigmoid primitive silu uses, the sigmoid
//   itself, a round of fp32 dest to bf16, and the two halves of the clamped SwiGLU.
// - "rows" forms: an SFPU op over the top half of a tile only (faces 0 and 1, VectorMode::R) and over its first ITER
//   SFPU iterations, for decode tiles whose valid rows are few (ITER 8 covers all 16 rows of a face).
//
// The Python side inlines this file into a kernel's source (SOURCE_CODE kernels have no include path of their own).

#pragma once

#include <cstdint>

#if !__has_include("api/compute/src_order.h")
#error "dense_ops.h needs tt-metal's compute API of v0.81 or later (api/compute/src_order.h)"
#endif

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"

// (math thread only: the unpack and pack threads do not have sfpi; silu's headers, which compute_kernel_api.h
// includes there, bring in the sigmoid)
#if defined(TRISC_MATH)
namespace ckernel::sfpu {
// gelu_tanh(x) = 0.5 x (1 + tanh(u)), u = sqrt(2 / pi) (x + 0.044715 x^3), = x * sigmoid(2 u): the sigmoid primitive
// silu uses (silu_tile_init initializes it), applied to 2 u, times x
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void tk_gelu_tanh() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = x * (1.5957691216f + 0.0713548162f * (x * x));
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(u);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
// sigmoid over ITERATIONS SFPU iterations (the primitive silu and gelu_tanh use), rounded to bf16 as they are
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void tk_sigmoid() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
// the fp32 dest rounded to bf16 precision, to nearest, so the result does not rest on the packer's own fp32-to-bf16
// rounding: with the value already rounded the pack is exact
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void tk_round_bf16() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        x = sfpi::convert<sfpi::vFloat16b>(x, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}
// The clamped SwiGLU of gpt-oss (alpha 1.702, limit 7): out = min(g, 7) * sigmoid(1.702 min(g, 7)) * (clamp(u, -7, 7)
// + 1). The gate half, with the sigmoid primitive silu uses (silu_tile_init initializes it); a NaN passes the clamp as
// in torch's clamp.
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void tk_oai_gate() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        v_if(x > 7.0f) { x = 7.0f; }
        v_endif;
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x * 1.702f);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
// the up half: clamp(u, -7, 7) + 1
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void tk_oai_up() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        v_if(x > 7.0f) { x = 7.0f; }
        v_elseif(x < -7.0f) { x = -7.0f; }
        v_endif;
        x = x + 1.0f;
        if constexpr (!is_fp32_dest_acc_en) {
            x = sfpi::convert<sfpi::vFloat16b>(x, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace tk {

// Full init of the matmul (hardware formats, unpacker, math, packer): in0 -> srcB, in1 -> srcA, subblock ct x rt, in0
// row stride kt; tr 1: each in1 tile transposed within itself.
ALWI void mm_init(uint32_t in0, uint32_t in1, uint32_t out, uint32_t ct, uint32_t rt, uint32_t kt, uint32_t tr = 0) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_block_init(in0, in1, tr, ct, rt, kt);
}

// Re-init of the matmul after another op (formats already set); tr as mm_init's.
ALWI void mm_init_short(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t rt, uint32_t kt, uint32_t tr = 0) {
    matmul_block_init(in0, in1, tr, ct, rt, kt);
}

ALWI void mul_bcast_rows_init(uint32_t a, uint32_t row) { ::mul_bcast_rows_init(a, row); }

// Full init for an eltwise binary op (a -> srcA, b -> srcB, out packed).
ALWI void unary_like_init(uint32_t a, uint32_t b, uint32_t out) { compute_kernel_hw_startup(a, b, out); }

ALWI void mul_bcast_scalar_init(uint32_t a, uint32_t b) { ::mul_bcast_scalar_init(a, b); }

ALWI void mul_init(uint32_t a, uint32_t b) { ::mul_init(a, b); }

ALWI void copy_init(uint32_t cb) { ::copy_init(cb); }

// silu over the top half of a tile (faces 0 and 1), its first ITER SFPU iterations of each (ITER 8: all 16 rows)
template <int ITER>
ALWI void silu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, ITER), idst, VectorMode::R));
}

// gelu_tanh over the top half's first ITER iterations (ITER 8: all 16 rows), or over all four faces
template <int ITER>
ALWI void gelu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_gelu_tanh, (DST_ACCUM_MODE, ITER), idst, VectorMode::R));
}
ALWI void gelu_full(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_gelu_tanh, (DST_ACCUM_MODE, 8), idst, VectorMode::RC));
}

// dest tile idst rounded to bf16 precision (all four faces)
ALWI void round_full(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_round_bf16, (DST_ACCUM_MODE, 8), idst, VectorMode::RC));
}

// sigmoid over the top half's first ITER iterations (ITER 8: all 16 rows)
template <int ITER>
ALWI void sigmoid_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_sigmoid, (DST_ACCUM_MODE, ITER), idst, VectorMode::R));
}

// the clamped SwiGLU's halves (tk_oai_gate, tk_oai_up) over the top half's first ITER iterations, or all four faces
template <int ITER>
ALWI void oai_gate_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_oai_gate, (DST_ACCUM_MODE, ITER), idst, VectorMode::R));
}
ALWI void oai_gate_full(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_oai_gate, (DST_ACCUM_MODE, 8), idst, VectorMode::RC));
}
template <int ITER>
ALWI void oai_up_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_oai_up, (DST_ACCUM_MODE, ITER), idst, VectorMode::R));
}
ALWI void oai_up_full(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, tk_oai_up, (DST_ACCUM_MODE, 8), idst, VectorMode::RC));
}

// the SFPU multiply dest[out] = dest[i0] * dest[i1] over the top half of the tiles (VectorMode::R)
ALWI void mul_rows(uint32_t i0, uint32_t i1, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        i0,
        i1,
        out,
        VectorMode::R)));
}

ALWI void pop(uint32_t cb, uint32_t n) { cb_pop_front(cb, n); }

}  // namespace tk
