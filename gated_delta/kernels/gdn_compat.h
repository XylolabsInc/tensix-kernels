// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Shared compute helpers for the gated delta-rule kernels. These wrappers call the
// upstream tt-metal compute API. gdn_scan.py inlines this header into kernel source.

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_binary_sfpu.h"  // binary SFPU ops over selected rows
#include "api/compute/bcast.h"
#include "api/compute/reconfig_data_format.h"

#include "api/compute/transpose.h"

namespace gdn {

// Pop consumed tiles from a compute circular buffer through the standard dataflow API.
ALWI void pop(uint32_t cb, uint32_t n) {
    cb_pop_front(cb, n);
}

// matmul: in0 -> srcB, in1 -> srcA; tr = 1 transposes each in1 tile.
ALWI void mm_init(uint32_t in0, uint32_t in1, uint32_t tr) {
    matmul_init(in0, in1, tr);
}

ALWI void add_init(uint32_t a, uint32_t b) {
    ::add_init(a, b);
}

ALWI void sub_init(uint32_t a, uint32_t b) {
    ::sub_init(a, b);
}

ALWI void mul_init(uint32_t a, uint32_t b) {
    ::mul_init(a, b);
}

ALWI void copy_init(uint32_t in) {
    ::copy_init(in);
}

ALWI void transpose_init(uint32_t in) {
    ::transpose_init(in);
}

ALWI void transpose_tile(uint32_t in, uint32_t itile, uint32_t idst) {
    ::transpose_tile(in, itile, idst);
}

ALWI void mul_bcast_cols_init(uint32_t a, uint32_t col) {
    ::mul_bcast_cols_init(a, col);
}

ALWI void mul_bcast_rows_init(uint32_t a, uint32_t row) {
    ::mul_bcast_rows_init(a, row);
}

// Initialize subtraction with row broadcast.
ALWI void sub_bcast_rows_init(uint32_t a, uint32_t row) {
    ::sub_bcast_rows_init(a, row);
}

ALWI void sub_tiles_bcast_rows(uint32_t a, uint32_t row, uint32_t ia, uint32_t ib, uint32_t idst) {
    ::sub_tiles_bcast_rows(a, row, ia, ib, idst);
}

ALWI void mul_bcast_scalar_init(uint32_t a, uint32_t s) {
    ::mul_bcast_scalar_init(a, s);
}

}  // namespace gdn

// Run SFPU ops over part of a tile through _llk_math_eltwise_*_sfpu_params_,
// using the SFPU function's ITERATIONS and VectorMode. R covers faces 0 and 1
// (rows 0-15); C covers faces 0 and 2 (columns 0-15). Within a face, one iteration
// covers rows 0-3 in alternating columns; two cover those rows in full. ITER
// iterations cover rows 0 .. 2 ITER - 1 in full. Selected elements receive the
// full op's math; other elements retain their Dst values. Use the full op's init.
#define GDN_UPARAMS(FN, IDST, M, ...) _llk_math_eltwise_unary_sfpu_params_(FN, IDST, VectorMode::M, ##__VA_ARGS__)
#define GDN_BPARAMS(FN, I0, I1, O, M) _llk_math_eltwise_binary_sfpu_params_(FN, I0, I1, O, VectorMode::M)

namespace gdn {

// rows 0 .. 2 ITER - 1 of faces 0 and 1
template <int ITER>
ALWI void softplus_rows(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold) {
    MATH((GDN_UPARAMS(
        (ckernel::sfpu::calculate_softplus<APPROX, DST_ACCUM_MODE, ITER>), idst, R, beta, beta_reciprocal, threshold)));
}
// Apply exp to the first SFPU row of each face; the decay tile holds one repeated value.
ALWI void exp_first(uint32_t idst) {
    exp_tile<false, false, InputClamping::ClampToNegative, 1>(idst, VectorMode::RC);
}
template <int ITER>
ALWI void exp_rows(uint32_t idst) {
    exp_tile<false, false, InputClamping::ClampToNegative, ITER>(idst, VectorMode::R);
}
template <int ITER>
ALWI void sigmoid_rows(uint32_t idst) {
    MATH((GDN_UPARAMS((ckernel::sfpu::calculate_sigmoid<APPROX, DST_ACCUM_MODE, ITER>), idst, R)));
}
template <int ITER>
ALWI void silu_rows(uint32_t idst) {
    MATH((GDN_UPARAMS((ckernel::sfpu::calculate_silu<DST_ACCUM_MODE, ITER>), idst, R)));
}
template <int ITER>
ALWI void add_binary_rows(uint32_t i0, uint32_t i1, uint32_t o) {
    MATH((GDN_BPARAMS(
        (ckernel::sfpu::calculate_sfpu_binary<APPROX, ckernel::BinaryOp::ADD, ITER, DST_ACCUM_MODE>), i0, i1, o, R)));
}
template <int ITER>
ALWI void mul_binary_rows(uint32_t i0, uint32_t i1, uint32_t o) {
    MATH((GDN_BPARAMS(
        (ckernel::sfpu::calculate_sfpu_binary_mul<APPROX, ckernel::BinaryOp::MUL, ITER, DST_ACCUM_MODE>), i0, i1, o, R)));
}
// rows 0 .. 2 ITER - 1 of column 0 (faces 0 and 2): ITER = 2 n - 1 covers rows 0 .. 4 n - 1 there
template <int ITER>
ALWI void rsqrt_col(uint32_t idst) {
    MATH((GDN_UPARAMS((ckernel::sfpu::calculate_rsqrt<APPROX, ITER, DST_ACCUM_MODE, false>), idst, C)));
}

}  // namespace gdn
