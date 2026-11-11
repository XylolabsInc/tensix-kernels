// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Compute of the block Hadamard of one token (t2.py hadamard): Y_b = H_32 X_b H_32 for the core's
// block (Z = X_b H in fp32, then Y = H Z), packed in bf16. Y_b is 32 times the normalized
// transform of the block (H_1024 = H_32 (x) H_32); the ternary scales carry the 1/32.
//
// CBs: c_8 X_b (bf16), c_10 Z (fp32), c_11 Y (bf16), c_12 H_32 (bf16).

#include "dense_ops.h"

constexpr uint32_t cb_xb = 8, cb_z = 10, cb_y = 11, cb_h = 12;

void kernel_main() {
    // Z = X_b H (X in srcB, H in srcA)
    tk::mm_init(cb_xb, cb_h, cb_z, 1, 1, 1);
    cb_wait_front(cb_h, 1);
    cb_wait_front(cb_xb, 1);
    cb_reserve_back(cb_z, 1);
    tile_regs_acquire();
    matmul_block(cb_xb, cb_h, 0, 0, 0, 0, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_z);
    tile_regs_release();
    cb_push_back(cb_z, 1);
    tk::pop(cb_xb, 1);

    // Y_b = H Z (H in srcB, Z in srcA)
    reconfig_data_format(cb_z, cb_h);
    pack_reconfig_data_format(cb_y);
    tk::mm_init_short(cb_h, cb_z, 1, 1, 1);
    cb_wait_front(cb_z, 1);
    cb_reserve_back(cb_y, 1);
    tile_regs_acquire();
    matmul_block(cb_h, cb_z, 0, 0, 0, 0, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_y);
    tile_regs_release();
    cb_push_back(cb_y, 1);
    tk::pop(cb_z, 1);
    tk::pop(cb_h, 1);
}
