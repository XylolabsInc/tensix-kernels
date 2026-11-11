// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of head_pick (head_pick.py). From the reader the TOP candidates (c_3: the valid count, the keys, the ids), from
// the compute the out tile (c_16: row r holds candidate c's logit in column c; fp32, or bf16 with ROUND). The pick is
// lmhead.pick_token's on the host: the largest logit, a NaN counting as the largest, the lowest token id among equals;
// empty candidates are left out. Row r of the output (uint32, one row-major page a row) gets [the token, the device's own
// best candidate (the first by key, for lmhead's miss count), the valid count]. IDS (ids_out): row 0's token also into
// page 0 of a one-element uint32 tensor (an embedding's index for a draft's next step), as 32 bytes [token, 0, ...]
// within the page's DRAM-aligned slot.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_ids = 3, cb_stg = 4, cb_out = 16;

// a float's bits as an unsigned key in the float order (NaN above everything, -0 equal to +0, as torch compares)
inline uint32_t order32(uint32_t b) {
    if ((b & 0x7FFFFFFFu) == 0) {
        b = 0;
    }
    if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0) {
        return 0xFFFFFFFFu;
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

void kernel_main() {
    constexpr uint32_t TOP = get_compile_time_arg_val(0);
    constexpr bool ROUND = get_compile_time_arg_val(1) != 0;
    constexpr bool IDS = get_compile_time_arg_val(2) != 0;
    constexpr auto o_args = TensorAccessorArgs<3>();
    constexpr auto d_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t r = get_arg_val<uint32_t>(i++);  // this core's row
    const uint32_t d_addr = get_arg_val<uint32_t>(i++);  // ids_out (IDS)
    const auto oa = TensorAccessor(o_args, o_addr, 128);
    const auto da = TensorAccessor(d_args, d_addr, 32);

    cb_wait_front(cb_ids, 1);
    volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ids));
    const uint32_t n = cand[0];
    cb_wait_front(cb_out, 1);
    const uint32_t lo = get_read_ptr(cb_out);
    // row r, columns 0 .. 15: face (r / 16) * 2, row r % 16 (a face is 16 x 16 values)
    uint32_t best = 0xFFFFFFFFu, best_key = 0;
    bool any = false;
    for (uint32_t c = 0; c < n; ++c) {
        const uint32_t face = ((r >> 4) << 1) + (c >> 4);
        uint32_t bits;
        if constexpr (ROUND) {
            const uint32_t at = lo + face * 512 + (r & 15) * 32 + (c & 15) * 2;
            bits = static_cast<uint32_t>(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(at)) << 16;
        } else {
            const uint32_t at = lo + face * 1024 + (r & 15) * 64 + (c & 15) * 4;
            bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at);
        }
        const uint32_t k = order32(bits);
        const uint32_t id = cand[1 + TOP + c];
        if (!any || k > best_key || (k == best_key && id < best)) {
            best_key = k;
            best = id;
            any = true;
        }
    }

    // the page of row r: [token, device best, valid count, 0 ...] (32 words)
    cb_reserve_back(cb_stg, 1);  // the output page's staging
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stg));
    for (uint32_t w = 0; w < 32; ++w) {
        o[w] = 0;
    }
    o[0] = any ? best : 0;
    o[1] = n > 0 ? cand[1 + TOP] : 0xFFFFFFFFu;
    o[2] = n;
    noc_async_write(get_write_ptr(cb_stg), oa.get_noc_addr(r), 128);
    noc_async_write_barrier();
    if constexpr (IDS) {
        if (r == 0) {
            o[1] = 0;
            o[2] = 0;
            noc_async_write(get_write_ptr(cb_stg), da.get_noc_addr(0), 32);  // [token, 0 x 7]
            noc_async_write_barrier();
        }
    }
    cb_pop_front(cb_out, 1);
    cb_pop_front(cb_ids, 1);
}
