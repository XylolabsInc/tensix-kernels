// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of moe8's down with two loaders (moe_dn_reader.cpp's work; moe8.down(loaders=2)): core j of bank `bank` holds ncols of H's column tiles. For each of the k slots, expert
// ids[s]'s rows, in chunks of KW K rows: h's KW tiles of that slot (h [1, k * I]: slot s's tiles
// s * IT ..) and the core's block of the expert (contiguous, IT x ncols tiles at e * estride + coff
// in the bank).
// SPLIT: chunk c of the running chunk index (over the slots) is loaded here when it is the reader's (split.h); the
// writer loads the others into c_4 (moe_dn2_writer.cpp). h's tiles are read here for every chunk.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "split.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_scr = 9;
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
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(5);  // split.h: 0 none, 1 every other chunk, else a pattern
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr auto h_args = TensorAccessorArgs<6>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    static_assert(IT % KW == 0 && K <= 16, "moe8: chunks; at most 16 experts");

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
    const auto i_acc = TensorAccessor(i_args, ids_addr, 64);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    volatile tt_l1_ptr uint32_t* ids32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);

    uint32_t g = 0;  // the running chunk index
    for (uint32_t s = 0; s < K; ++s) {
        const uint32_t e = IDS32 ? ids32[s] : ids16[s];
        const uint32_t base = dn_addr + e * estride + coff;
        for (uint32_t c = 0; c < IT / KW; ++c, ++g) {
            cb_reserve_back(cb_x, KW);
            const uint32_t lx = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < KW; ++k) {
                noc_async_read_page(s * IT + c * KW + k, h_acc, lx + k * 2048);
            }
            const bool load = !(SPLIT && tk::split_writer<SPLIT_ARG>(g));
            if (load) {
                cb_reserve_back(cb_w, KW * NC);
                const uint32_t dst = get_write_ptr(cb_w);
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
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, KW);
            if (load) {
                cb_push_back(cb_w, KW * NC);
            }
        }
    }
}
