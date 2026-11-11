// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of moe8's down_rows (moe_dn_compute.cpp with DYN; moe8.py): core j of bank `bank` holds
// ncols of H's column tiles, as in moe_dn_reader.cpp, and runs K over the union of the R x k ids
// (moe8_rows.h) instead of the k slots: for union index q, h's tiles q * IT .. (gate_up_rows'
// layout) and expert list[q]'s block. The chunk count goes to the compute first (cb_meta, word 0).
// SPLIT (the second loader, moe8.down_rows(loaders=2)): chunk g of the running chunk index (over the union) has its
// weights loaded here when it is the reader's (split.h); the writer loads the others into c_4 (moe_dnr2_writer.cpp),
// as moe_dn2_*.cpp do for decode. h's tiles are read here for every chunk.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe8_rows.h"
#include "split.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 3, cb_scr = 9;
constexpr uint32_t TB = 576;

inline void zero(uint32_t dst, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        const uint32_t n = bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE;
        noc_async_read(z, dst + off, n);
    }
}

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t IT = get_compile_time_arg_val(1);  // K tiles per expert (I / 32)
    constexpr uint32_t KW = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr bool IDS32 = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t ROWS = get_compile_time_arg_val(5);
    constexpr uint32_t E = get_compile_time_arg_val(6);
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(7);  // split.h: the writer's chunks (1: the odd ones)
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr auto h_args = TensorAccessorArgs<8>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    static_assert(IT % KW == 0 && K <= 16 && ROWS >= 1 && ROWS <= 32 && E <= 1024, "moe8 rows");
    constexpr uint32_t OFF_BITS = 4096, OFF_LIST = OFF_BITS + 128;

    uint32_t i = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(i++);
    const uint32_t dn_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t coff = get_arg_val<uint32_t>(i++);
    const uint32_t ncols = get_arg_val<uint32_t>(i++);
    const uint32_t estride = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    const auto h_acc = TensorAccessor(h_args, h_addr, 2048);
    const auto i_acc = TensorAccessor(i_args, ids_addr, IDS32 ? 4096 : 2048);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, IDS32 ? 4096 : 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_BITS);
    volatile tt_l1_ptr uint16_t* list = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_LIST);
    const uint32_t u = moe8r::build_union<IDS32, K, E>(scr, ROWS, bits, list);

    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = u * (IT / KW);
    cb_push_back(cb_meta, 1);

    [[maybe_unused]] uint32_t g = 0;  // the running chunk index (SPLIT: the writer's weights by split.h)
    for (uint32_t q = 0; q < u; ++q) {
        const uint32_t e = list[q];
        const uint32_t base = dn_addr + e * estride + coff;
        for (uint32_t c = 0; c < IT / KW; ++c, ++g) {
            cb_reserve_back(cb_x, KW);
            const uint32_t lx = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < KW; ++k) {
                noc_async_read_page(q * IT + c * KW + k, h_acc, lx + k * 2048);
            }
            bool load = true;
            if constexpr (SPLIT) {
                load = !tk::split_writer<SPLIT_ARG>(g);
            }
            if (load) {
                cb_reserve_back(cb_w, KW * NC);
                const uint32_t dst = get_write_ptr(cb_w);
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
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, KW);
            if (load) {
                cb_push_back(cb_w, KW * NC);
            }
        }
    }
}
