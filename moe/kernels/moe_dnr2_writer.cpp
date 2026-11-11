// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's down_rows with the second loader (moe8.down_rows(loaders=2); moe_dnr_reader.cpp with SPLIT): it
// builds the union of the R x k ids (moe8_rows.h) as the reader does, loads the weights of its chunks of the running
// chunk index (over the union, IT / KW chunks an expert; split.h) into c_4, a second request stream on the writer's
// NoC, as moe_dn2_writer.cpp does for decode; the reader loads the others into c_1, and h's tiles of every chunk. Then,
// as moe_dn_writer.cpp, the core's ncols output tiles to y's tiles col0 .. (y [1, 1, 32, H]). The output is written
// once, after the K loop, so c_16 holds NC pages as without the second loader.
//
// Scratch (cb_scr): the ids tile, the union's bits and its list (moe_dnr_reader.cpp's offsets).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe8_rows.h"
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
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t IT = get_compile_time_arg_val(2);  // K tiles per expert (I / 32)
    constexpr uint32_t KW = get_compile_time_arg_val(3);
    constexpr bool IDS32 = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t ROWS = get_compile_time_arg_val(5);
    constexpr uint32_t E = get_compile_time_arg_val(6);
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(7);  // split.h: the writer's chunks (1: the odd ones)
    constexpr auto y_args = TensorAccessorArgs<8>();
    constexpr auto i_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    static_assert(IT % KW == 0 && K <= 16 && ROWS >= 1 && ROWS <= 32 && E <= 1024, "moe8 rows");
    constexpr uint32_t OFF_BITS = 4096, OFF_LIST = OFF_BITS + 128;

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

    const auto i_acc = TensorAccessor(i_args, ids_addr, IDS32 ? 4096 : 2048);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, IDS32 ? 4096 : 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_BITS);
    volatile tt_l1_ptr uint16_t* list = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_LIST);
    const uint32_t u = moe8r::build_union<IDS32, K, E>(scr, ROWS, bits, list);

    uint32_t g = 0;  // the running chunk index, as the reader's and the compute's
    for (uint32_t q = 0; q < u; ++q) {
        const uint32_t base = dn_addr + list[q] * estride + coff;
        for (uint32_t c = 0; c < IT / KW; ++c, ++g) {
            if (!tk::split_writer<SPLIT_ARG>(g)) {
                continue;
            }
            cb_reserve_back(cb_w2, KW * NC);
            const uint32_t dst = get_write_ptr(cb_w2);
            for (uint32_t k = 0; k < KW; ++k) {
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank, base + (c * KW + k) * ncols * TB),
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
