// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of the W4A16 group-scale matmul (w4a16_compute.cpp), one token:
//   - ones: a bf16 tile with 1.0 in row 0 (the column sum's operand);
//   - x row-spread: activation tile t of a block holds x's K tile t in row t % 32 (the token is
//     row 0 of x's tile t: its face-0 and face-1 row segments, read in 64-byte pieces into a
//     staging page and copied); the other rows stay zero (zeroed once: tile t of every block has
//     its data in the same row);
//   - per (column subblock, pass of 32 K tiles), in the compute's order: the pass's weight chunks
//     (KW x SBW tiles) and the SBW scale tiles of the pass (scale tensor [K/32, N] bf16 TILE,
//     interleaved: tile (pass, column)).
// The weight is a uint8 ROW_MAJOR tensor, DRAM width-sharded: in bank `bank`, row kt holds the
// shard's int4 tiles (512 bytes each: sign + 3-bit magnitude nibbles in the tile's face order, the
// first element in the low nibble, the mantissa section of a BFP4_b tile) at w_addr + kt * pitch
// + col * 512. A K row of the subblock is one read into a staging buffer, then each 512-byte tile
// is copied over the core's own NoC behind the 64 constant exponent bytes (EXP = 129: mantissa m
// reads as m) of its BFP4_b page (576 bytes = 9 * 64).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "w4a16_wload.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_one = 5, cb_stage = 8, cb_xstage = 9;

inline void copy_words(uint32_t dst, uint32_t src, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = s[q];
    }
}

inline void fill_words(uint32_t dst, uint32_t word, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = word;
    }
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t KW = get_compile_time_arg_val(2);
    constexpr uint32_t SBW = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(5);
    constexpr uint32_t EXP = get_compile_time_arg_val(6);
    constexpr bool SPLIT = get_compile_time_arg_val(7) != 0;
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t NKB = KT / KB;
    constexpr uint32_t NS = NC / SBW;
    constexpr uint32_t NP = KB / 32;

    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t pitch = get_arg_val<uint32_t>(i++);  // bytes per weight row in the shard
    const uint32_t col0 = get_arg_val<uint32_t>(i++);   // first column (tile) in the shard
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);  // first global output column tile
    const uint32_t nt = get_arg_val<uint32_t>(i++);  // output column tiles of the scale tensor
    const uint32_t vc = get_arg_val<uint32_t>(i++);

    const uint32_t tx = get_tile_size(cb_x);
    const uint32_t ts = get_tile_size(cb_s);
    const auto x_acc = TensorAccessor(x_args, x_addr, tx);
    const auto s_acc = TensorAccessor(s_args, s_addr, ts);

    w4::init_exponents(cb_w, W_PAGES, EXP);  // the weight pages' constant exponents, once
    // ones in row 0 (faces 0 and 1, row 0), bf16
    cb_reserve_back(cb_one, 1);
    {
        const uint32_t o = get_write_ptr(cb_one);
        fill_words(o, 0, tx);
        fill_words(o, 0x3F803F80u, 32);
        fill_words(o + 512, 0x3F803F80u, 32);
    }
    cb_push_back(cb_one, 1);
    // the x block's slots: zero once
    {
        cb_reserve_back(cb_x, KB);
        fill_words(get_write_ptr(cb_x), 0, KB * tx);
    }
    cb_reserve_back(cb_stage, 1);  // scratch: a K row of a subblock (SBW * 512 B), never pushed
    const uint32_t stage = get_write_ptr(cb_stage);
    cb_reserve_back(cb_xstage, 1);  // scratch: x's row pieces (64-byte aligned), never pushed
    const uint32_t xst = (get_write_ptr(cb_xstage) + 63) & ~63u;

    uint32_t g = 0;  // running chunk index: with SPLIT the writer loads the odd ones
    for (uint32_t b = 0; b < NKB; ++b) {
        cb_reserve_back(cb_x, KB);
        const uint32_t lx = get_write_ptr(cb_x);
        for (uint32_t k0 = 0; k0 < KB; k0 += 16) {
            for (uint32_t k = 0; k < 16; ++k) {
                const uint64_t src = x_acc.get_noc_addr(b * KB + k0 + k);
                noc_async_read(src, xst + k * 128, 64);
                noc_async_read(src + 512, xst + k * 128 + 64, 64);
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t r = (k0 + k) & 31;
                const uint32_t row = lx + (k0 + k) * tx + ((r >> 4) * 2) * 512 + (r & 15) * 32;
                copy_words(row, xst + k * 128, 32);
                copy_words(row + 512, xst + k * 128 + 64, 32);
            }
        }
        cb_push_back(cb_x, KB);

        for (uint32_t s = 0; s < NS; ++s) {
            const uint32_t j0 = s * SBW;
            const uint32_t jn = nvalid > j0 ? (nvalid - j0 < SBW ? nvalid - j0 : SBW) : 0;
            for (uint32_t p = 0; p < NP; ++p) {
                for (uint32_t c = 0; c < 32 / KW; ++c, ++g) {
                    if (!SPLIT || (g & 1) == 0) {
                        w4::chunk<KW, SBW>(
                            cb_w, stage, bank, w_addr, pitch, col0 + j0, jn, b * KB + p * 32 + c * KW, vc);
                    }
                }
                // the pass's scale tiles of the subblock
                cb_reserve_back(cb_s, SBW);
                const uint32_t ls = get_write_ptr(cb_s);
                const uint32_t pass = (b * KB) / 32 + p;
                for (uint32_t j = 0; j < jn; ++j) {
                    noc_async_read_page(pass * nt + n0 + j0 + j, s_acc, ls + j * ts);
                }
                noc_async_read_barrier();
                cb_push_back(cb_s, SBW);
            }
        }
    }
}
