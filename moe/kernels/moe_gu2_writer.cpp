// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's gate_up with two loaders (moe_gu2_reader.cpp). With SPLIT it first loads its chunks of the core's
// running chunk index (split.h) into c_4 (the reader loads the others into c_1), a second request stream on the
// writer's NoC; the compute takes those chunks from c_4. Then, as moe_gu_writer.cpp, per slot s of this core the valid products (the expert's pairs
// held by this bank) to h's tiles s * PAIRS + first pair + p (h [1, k * I], interleaved); c_16
// holds every slot's products (the compute never waits for these writes). With G column groups (moe8.gu_groups)
// each slot is G units: group gi's loads and its products, pairs gi * NC / 2 .. of the expert's here.

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
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t CPB = get_compile_time_arg_val(2);
    constexpr uint32_t BANKS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4);
    constexpr bool IDS32 = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(6);  // split.h: 0 none, 1 every other chunk, else a pattern
    constexpr bool SPLIT = SPLIT_ARG != 0;
    constexpr uint32_t KT = get_compile_time_arg_val(7);
    constexpr uint32_t KW = get_compile_time_arg_val(8);
    constexpr uint32_t G = get_compile_time_arg_val(9);  // column groups per slot
    constexpr auto h_args = TensorAccessorArgs<10>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t H = NC / 2;
    constexpr uint32_t CHUNKS = KT / KW;
    constexpr uint32_t Q = PAIRS / BANKS, R = PAIRS % BANKS;

    uint32_t i = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);
    const uint32_t gu_addr = get_arg_val<uint32_t>(i++);
    const uint32_t t_addr = get_arg_val<uint32_t>(i++);
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    const uint32_t ty = get_tile_size(cb_out);
    const auto h_acc = TensorAccessor(h_args, h_addr, ty);
    const auto i_acc = TensorAccessor(i_args, ids_addr, 64);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    volatile tt_l1_ptr uint32_t* ids32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);

    if constexpr (SPLIT) {
        const auto t_acc = TensorAccessor(t_args, t_addr, 64);
        const uint32_t rows = scr + 64;
        for (uint32_t s = j, u = 0; s < K; s += CPB, ++u) {
            noc_async_read_page(IDS32 ? ids32[s] : ids16[s], t_acc, rows + u * 64);
        }
        noc_async_read_barrier();
        uint32_t g = 0;
        for (uint32_t s = j, u = 0; s < K; s += CPB, ++u) {
            const uint32_t e = IDS32 ? ids32[s] : ids16[s];
            const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + u * 64)[bank];
            const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
            const uint32_t cnt = 2 * (Q + (rank < R ? 1 : 0));
            for (uint32_t gi = 0; gi < G; ++gi) {
                const uint32_t gcnt = cnt > gi * NC ? (cnt - gi * NC < NC ? cnt - gi * NC : NC) : 0;
                const uint32_t goff = off + gi * KT * NC * TB;
                for (uint32_t c = 0; c < CHUNKS; ++c, ++g) {
                    if (!tk::split_writer<SPLIT_ARG>(g)) {
                        continue;
                    }
                    cb_reserve_back(cb_w2, KW * NC);
                    const uint32_t dst = get_write_ptr(cb_w2);
                    const uint32_t src = gu_addr + goff + c * KW * gcnt * TB;
                    for (uint32_t k = 0; k < KW; ++k) {
                        if (gcnt > 0) {
                            noc_async_read(
                                get_noc_addr_from_bank_id<true>(bank, src + k * gcnt * TB),
                                dst + k * NC * TB,
                                gcnt * TB,
                                noc_index,
                                vc);
                        }
                        if (gcnt < NC) {
                            zero(dst + (k * NC + gcnt) * TB, (NC - gcnt) * TB);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_w2, KW * NC);
                }
            }
        }
    }

    for (uint32_t s = j; s < K; s += CPB) {
        const uint32_t e = IDS32 ? ids32[s] : ids16[s];
        const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
        const uint32_t pc = Q + (rank < R ? 1 : 0);
        const uint32_t p0 = rank * Q + (rank < R ? rank : R);
        for (uint32_t gi = 0; gi < G; ++gi) {
            // this group's valid products (G 1: all pc)
            const uint32_t gp = pc > gi * H ? (pc - gi * H < H ? pc - gi * H : H) : 0;
            cb_wait_front(cb_out, H);
            const uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t p = 0; p < gp; ++p) {
                noc_async_write_page(s * PAIRS + p0 + gi * H + p, h_acc, rd + p * ty);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, H);
        }
    }
}
