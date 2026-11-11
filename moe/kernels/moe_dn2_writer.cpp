// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's down with two loaders (moe_dn2_reader.cpp). With SPLIT it first loads its chunks of the running
// chunk index (split.h) into c_4 (the reader loads the others into c_1, and h's tiles of every chunk), a second
// request stream on the writer's NoC. Then, as moe_dn_writer.cpp,
// the core's ncols output tiles to y's tiles col0 .. (y [1, H]).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "split.h"

constexpr uint32_t cb_w2 = 4, cb_scr = 10, cb_out = 16;
constexpr uint32_t TB = 576;

inline void zero(uint32_t dst, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        const uint32_t n = bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE;
        noc_async_read(z, dst + off, n);
    }
}

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(1);  // split.h: 0 none, 1 every other chunk, else a pattern
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t IT = get_compile_time_arg_val(3);
    constexpr uint32_t KW = get_compile_time_arg_val(4);
    constexpr bool IDS32 = get_compile_time_arg_val(5) != 0;
    constexpr auto y_args = TensorAccessorArgs<6>();
    constexpr auto i_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    uint32_t i = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t ncols = get_arg_val<uint32_t>(i++);
    const uint32_t dn_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t coff = get_arg_val<uint32_t>(i++);
    const uint32_t estride = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    if constexpr (SPLIT) {
        const auto i_acc = TensorAccessor(i_args, ids_addr, 64);
        cb_reserve_back(cb_scr, 1);
        const uint32_t scr = get_write_ptr(cb_scr);
        noc_async_read(i_acc.get_noc_addr(0), scr, 64);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
        volatile tt_l1_ptr uint32_t* ids32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        uint32_t g = 0;
        for (uint32_t s = 0; s < K; ++s) {
            const uint32_t e = IDS32 ? ids32[s] : ids16[s];
            const uint32_t base = dn_addr + e * estride + coff;
            for (uint32_t c = 0; c < IT / KW; ++c, ++g) {
                if (!tk::split_writer<SPLIT_ARG>(g)) {
                    continue;
                }
                cb_reserve_back(cb_w2, KW * NC);
                const uint32_t dst = get_write_ptr(cb_w2);
                const uint32_t src = base + c * KW * ncols * TB;
                for (uint32_t k = 0; k < KW; ++k) {
                    noc_async_read(
                        get_noc_addr_from_bank_id<true>(bank, src + k * ncols * TB),
                        dst + k * NC * TB,
                        ncols * TB,
                        noc_index,
                        vc);
                    if (ncols < NC) {
                        zero(dst + (k * NC + ncols) * TB, (NC - ncols) * TB);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_w2, KW * NC);
            }
        }
    }

    const uint32_t ty = get_tile_size(cb_out);
    const auto y_acc = TensorAccessor(y_args, y_addr, ty);
    cb_wait_front(cb_out, NC);
    const uint32_t rd = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < ncols; ++c) {
        noc_async_write_page(col0 + c, y_acc, rd + c * ty);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, NC);
}
