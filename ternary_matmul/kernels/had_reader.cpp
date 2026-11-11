// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of the block Hadamard of one token (t2.py hadamard; had_compute.cpp, had_writer.cpp): one
// block of 1024 inputs per core, block b = the core's runtime argument.
//   - H_32 (+-1, Sylvester order, face order) once, made here;
//   - X_b: the 32 x 32 tile whose row a is row 0 of x's K tile 32b + a (each read with the tile's
//     first 576 bytes, which hold both halves of row 0), the signs applied as an XOR of the bf16
//     sign bits (the mask tensor of hadamard_signs: 16 words per K tile, one page).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_xb = 8, cb_xstage = 9, cb_h = 12, cb_mask = 17;
constexpr uint32_t XROW = 576;

inline uint32_t row_off(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }

inline void copy_xor(uint32_t dst, uint32_t src, uint32_t mask, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = s[q] ^ m[q];
    }
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t m_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b = get_arg_val<uint32_t>(i++);

    const uint32_t tx = 2048;
    const auto x_acc = TensorAccessor(x_args, x_addr, tx);
    const auto m_acc = TensorAccessor(m_args, m_addr, KT * 64);

    // the block's x rows and masks (2 KB of the one mask page) first, then H_32 while they land
    cb_reserve_back(cb_xstage, 1);
    const uint32_t xst = (get_write_ptr(cb_xstage) + 63) & ~63u;
    for (uint32_t a = 0; a < 32; ++a) {
        noc_async_read(x_acc.get_noc_addr(32 * b + a), xst + a * XROW, XROW);
    }
    cb_reserve_back(cb_mask, 1);
    const uint32_t mk = get_write_ptr(cb_mask);
    noc_async_read(m_acc.get_noc_addr(0) + b * 32 * 64, mk, 32 * 64);

    cb_reserve_back(cb_h, 1);
    {
        volatile tt_l1_ptr uint16_t* h = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_h));
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t f = (r >> 4) * 2 + (c >> 4);
                h[f * 256 + (r & 15) * 16 + (c & 15)] = (__builtin_popcount(r & c) & 1) ? 0xBF80 : 0x3F80;
            }
        }
    }
    cb_push_back(cb_h, 1);

    cb_reserve_back(cb_xb, 1);
    const uint32_t page = get_write_ptr(cb_xb);
    noc_async_read_barrier();
    for (uint32_t a = 0; a < 32; ++a) {
        const uint32_t m = mk + a * 64;
        copy_xor(page + row_off(a), xst + a * XROW, m, 32);
        copy_xor(page + row_off(a) + 512, xst + a * XROW + 512, m + 32, 32);
    }
    cb_push_back(cb_xb, 1);
}
