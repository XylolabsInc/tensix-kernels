// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of the block Hadamard of one token (t2.py hadamard): row a of Y_b becomes row 0 of x''s K
// tile 32b + a, rows 1-31 zero, written as whole bf16 tiles into x' [1, 1, 1, K] (interleaved, L1
// or DRAM). The 32 tiles are built in a scratch CB (zeroed once from the core's zero region),
// then written.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_y = 11, cb_out = 16;

inline uint32_t row_off(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }

inline void copy_words(uint32_t dst, uint32_t src, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = s[q];
    }
}

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b = get_arg_val<uint32_t>(i++);
    const uint32_t tx = 2048;
    const auto o_acc = TensorAccessor(o_args, o_addr, tx);

    cb_reserve_back(cb_out, 32);  // scratch: the 32 output tiles, never pushed
    const uint32_t out = get_write_ptr(cb_out);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < 32 * tx; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, out + off, MEM_ZEROS_SIZE);
    }
    cb_wait_front(cb_y, 1);
    const uint32_t y = get_read_ptr(cb_y);
    noc_async_read_barrier();
    for (uint32_t a = 0; a < 32; ++a) {
        copy_words(out + a * tx, y + row_off(a), 32);
        copy_words(out + a * tx + 512, y + row_off(a) + 512, 32);
        noc_async_write_page(32 * b + a, o_acc, out + a * tx);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_y, 1);
}
