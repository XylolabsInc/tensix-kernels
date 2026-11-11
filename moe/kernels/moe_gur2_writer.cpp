// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's gate_up_rows with the second loader (moe8.gate_up_rows(loaders=2); moe_gur_reader.cpp with SPLIT):
// per unit n of this core (union index q = j + (n / G) * CPB, moe8_rows.h, group gi = n % G) it loads its chunks of
// the core's running chunk index (split.h) into c_4, a second request stream on the writer's NoC, as
// moe_gu2_writer.cpp does for decode; the reader loads the others into c_1 and the compute takes the writer's from
// c_4. Then, as
// moe_gur_writer.cpp, unit n's valid products (expert list[q]'s pairs held by this bank) to h's tiles
// q * PAIRS + first pair + gi * NC / 2 + p.
//
// The order is load(0), load(1), write(0), load(2), write(1), ..., write(n - 1): unit n's products are written after
// unit n + 1's loads, so the compute, which needs both CBs for unit n + 1, never waits on c_16 while this writer waits
// on c_4. c_16 then holds two units' products (NC pages), whatever the union; decode's writer drains them only after all
// of its loads, which needs c_16 for every unit (k / CPB of them, not u / CPB).
//
// Scratch (cb_scr): the ids tile, the union's bits, its list, the table row of the unit being loaded.

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
    constexpr uint32_t CPB = get_compile_time_arg_val(2);
    constexpr uint32_t BANKS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4);
    constexpr bool IDS32 = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t ROWS = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);  // column groups per union index
    constexpr uint32_t KT = get_compile_time_arg_val(9);
    constexpr uint32_t KW = get_compile_time_arg_val(10);
    constexpr uint32_t SPLIT_ARG = get_compile_time_arg_val(11);  // split.h: 1 every other chunk, else a pattern
    constexpr auto h_args = TensorAccessorArgs<12>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    static_assert(KT % KW == 0 && K <= 16 && ROWS >= 1 && ROWS <= 32 && E <= 1024, "moe8 rows");
    constexpr uint32_t H = NC / 2;
    constexpr uint32_t CHUNKS = KT / KW;
    constexpr uint32_t Q = PAIRS / BANKS, R = PAIRS % BANKS;
    constexpr uint32_t OFF_BITS = 4096, OFF_LIST = OFF_BITS + 128;
    constexpr uint32_t OFF_T = (OFF_LIST + 2 * E + 63) / 64 * 64;  // moe8.py sizes c_10 to OFF_T + 64

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
    const auto i_acc = TensorAccessor(i_args, ids_addr, IDS32 ? 4096 : 2048);
    const auto t_acc = TensorAccessor(t_args, t_addr, 64);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(i_acc.get_noc_addr(0), scr, IDS32 ? 4096 : 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* bits = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_BITS);
    volatile tt_l1_ptr uint16_t* list = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + OFF_LIST);
    const uint32_t u = moe8r::build_union<IDS32, K, E>(scr, ROWS, bits, list);
    const uint32_t units = moe8r::units_of(u, j, CPB) * G;

    uint32_t off = 0;  // the byte offset in this bank of the expert being loaded (its table row)
    for (uint32_t n = 0; n <= units; ++n) {
        if (n < units) {
            // load(n): unit n's chunks of the running chunk index g = n * CHUNKS + c that are the writer's (split.h)
            const uint32_t q = j + (n / G) * CPB, gi = n % G;
            const uint32_t e = list[q];
            if (gi == 0) {
                noc_async_read_page(e, t_acc, scr + OFF_T);
                noc_async_read_barrier();
                off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + OFF_T)[bank];
            }
            const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
            const uint32_t cnt = 2 * (Q + (rank < R ? 1 : 0));
            const uint32_t gcnt = cnt > gi * NC ? (cnt - gi * NC < NC ? cnt - gi * NC : NC) : 0;
            const uint32_t goff = off + gi * KT * NC * TB;
            for (uint32_t c = 0; c < CHUNKS; ++c) {
                if (!tk::split_writer<SPLIT_ARG>(n * CHUNKS + c)) {
                    continue;
                }
                cb_reserve_back(cb_w2, KW * NC);
                const uint32_t dst = get_write_ptr(cb_w2);
                for (uint32_t k = 0; k < KW; ++k) {
                    if (gcnt > 0) {
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(bank, gu_addr + goff + (c * KW + k) * gcnt * TB),
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
        if (n > 0) {
            // write(n - 1): its valid products, as moe_gur_writer.cpp
            const uint32_t m = n - 1;
            const uint32_t q = j + (m / G) * CPB, gi = m % G;
            const uint32_t e = list[q];
            const uint32_t rank = (bank + BANKS - (e % BANKS)) % BANKS;
            const uint32_t pc = Q + (rank < R ? 1 : 0);
            const uint32_t p0 = rank * Q + (rank < R ? rank : R);
            const uint32_t gp = pc > gi * H ? (pc - gi * H < H ? pc - gi * H : H) : 0;
            cb_wait_front(cb_out, H);
            const uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t p = 0; p < gp; ++p) {
                noc_async_write_page(q * PAIRS + p0 + gi * H + p, h_acc, rd + p * ty);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, H);
        }
    }
}
