// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Reader of the ternary matmul (t2_compute.cpp; t2.py), one token, x streamed with the weights:
//   - ones: a bf16 tile with 1.0 in row 0 (the column sum's operand), once;
//   - per pass of 32 groups (a group: G inputs = GT K tiles): the scale tiles of the core's columns
//     (scale tensor [K/G, N] bf16 TILE, interleaved: page pass * nst + column);
//   - per chunk (KW K rows of the core's NC columns, KW a multiple of GT): its KW x tiles,
//     row-spread (x's row 0, read with the tile's first 576 bytes, which hold both of its halves,
//     copied into row (K tile / GT) % 32 of a page
//     of the x ring; the row the page held before is zeroed first), and its weights (t2_wload.h;
//     with SPLIT the even chunks only, the writer loads the odd ones).
// NB > 0 (the block Hadamard of a checkpoint whose ternary weights live in a rotated basis, x' = H_1024 (s * x) per
// block of 1024 inputs, NB blocks): first the H_32 tile (+-1, Sylvester order) once, then per block X_b, the
// 32 x 32 tile whose row a is K tile 32b + a's row 0 with the signs applied (an XOR of the sign bits,
// a mask of 16 words per K tile, read once). The compute returns Y_b = H_32 X_b H_32 (= 32 x'_b, row a
// K tile 32b + a of x'; H_1024 = H_32 (x) H_32), and the rows are spread from those tiles instead
// of from x.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "t2_wload.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_s = 2, cb_one = 5, cb_xb = 8, cb_xstage = 9, cb_y = 11, cb_h = 12;
constexpr uint32_t cb_mask = 17;
constexpr uint32_t XROW = 576;  // a tile's first bytes up to the end of row 0's second half

inline void copy_words(uint32_t dst, uint32_t src, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = s[q];
    }
}

inline void copy_xor(uint32_t dst, uint32_t src, uint32_t mask, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = s[q] ^ m[q];
    }
}

inline void fill_words(uint32_t dst, uint32_t word, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t q = 0; q < bytes / 4; ++q) {
        d[q] = word;
    }
}

// the offset of row r's first half (face 0, or face 2 for r >= 16) in a bf16 tile in face order;
// its second half (face 1 or 3) is 512 bytes further
inline uint32_t row_off(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t KW = get_compile_time_arg_val(1);
    constexpr uint32_t GT = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr bool SPLIT = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t XPAGES = get_compile_time_arg_val(5);  // the x ring's pages
    constexpr uint32_t TB = get_compile_time_arg_val(6);      // weight tile bytes
    constexpr uint32_t NB = get_compile_time_arg_val(7);      // Hadamard blocks (0: none)
    constexpr uint32_t DB = get_compile_time_arg_val(8);      // weight tile bytes in DRAM
    constexpr uint32_t W_PAGES = get_compile_time_arg_val(9);
    constexpr uint32_t EXP = get_compile_time_arg_val(10);    // DB < TB: the pages' exponent byte
    // XDMA: x's rows land in the ring by NoC reads straight from x (in L1, 16-byte aligned) or from
    // the Hadamard's Y tiles, and old rows are cleared by reads of the zero region: no RISC-V
    // copies. Without it (x in DRAM, whose reads need 64-byte alignment) x goes through staging.
    constexpr bool XDMA = get_compile_time_arg_val(11) != 0;
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t PT = 32 * GT;  // K tiles per pass
    constexpr uint32_t NCH = KT / KW;
    static_assert(KT % KW == 0 && KW % GT == 0 && PT % KW == 0, "t2: chunks of whole groups");
    static_assert(XPAGES % KW == 0, "t2: the x ring");
    static_assert(NB == 0 || KT == 32 * NB, "t2: the Hadamard covers K in blocks of 1024");

    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bank = get_arg_val<uint32_t>(i++);
    const uint32_t core_off = get_arg_val<uint32_t>(i++);  // the core's first byte in the bank
    const uint32_t nvalid = get_arg_val<uint32_t>(i++);
    const uint32_t n0 = get_arg_val<uint32_t>(i++);   // the core's first global column tile
    const uint32_t nst = get_arg_val<uint32_t>(i++);  // column tiles of the scale tensor
    const uint32_t vc = get_arg_val<uint32_t>(i++);
    const uint32_t m_addr = get_arg_val<uint32_t>(i++);  // NB: the sign masks

    const uint32_t tx = get_tile_size(cb_x);
    const uint32_t ts = get_tile_size(cb_s);
    const auto x_acc = TensorAccessor(x_args, x_addr, tx);
    const auto s_acc = TensorAccessor(s_args, s_addr, ts);
    t2::Chunks<KW, NC, TB, DB> ld{cb_w, bank, w_addr + core_off, nvalid, vc};
    if constexpr (DB < TB) {
        t2::init_exponents(cb_w, W_PAGES, TB, EXP);
        cb_reserve_back(14, 1);  // the staging buffer: scratch, never pushed
        ld.stage = get_write_ptr(14);
    }

    // ones in row 0 (faces 0 and 1, row 0), bf16
    cb_reserve_back(cb_one, 1);
    {
        const uint32_t o = get_write_ptr(cb_one);
        fill_words(o, 0, tx);
        fill_words(o, 0x3F803F80u, 32);
        fill_words(o + 512, 0x3F803F80u, 32);
    }
    cb_push_back(cb_one, 1);
    // the x ring: zeroed once, the first 512 bytes from the core's zero region, then by doubling
    // copies of what is already zero (a DMA, not RISC-V stores, few requests); held[q] is the row
    // page q holds (32: none). With XPAGES a multiple of a pass (32 groups), page q always holds
    // row (q / GT) % 32, so no row is ever cleared again.
    cb_reserve_back(cb_x, XPAGES);
    const uint32_t ring = get_write_ptr(cb_x);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    {
        constexpr uint32_t total = XPAGES * 2048;
        noc_async_read(zeros, ring, MEM_ZEROS_SIZE);
        noc_async_read_barrier();
        for (uint32_t done = MEM_ZEROS_SIZE; done < total;) {
            const uint32_t n = done < total - done ? done : total - done;
            noc_async_read(get_noc_addr(ring), ring + done, n);
            noc_async_read_barrier();
            done += n;
        }
    }
    uint8_t held[XPAGES];
    for (uint32_t q = 0; q < XPAGES; ++q) {
        held[q] = 32;
    }
    cb_reserve_back(cb_xstage, 1);  // scratch: x's row pieces (64-byte aligned), never pushed
    const uint32_t xst = (get_write_ptr(cb_xstage) + 63) & ~63u;

    uint32_t ybase = 0;  // NB: the Y_b tiles (bf16, all NB of them, in order)
    if constexpr (NB > 0) {
        // H_32 in face order: element (r, c) = (-1)^popcount(r & c)
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
        // the masks: one page of KT x 64 bytes, read once
        const auto m_acc = TensorAccessor(m_args, m_addr, KT * 64);
        cb_reserve_back(cb_mask, 1);
        const uint32_t mk = get_write_ptr(cb_mask);
        noc_async_read_page(0, m_acc, mk);
        for (uint32_t b = 0; b < NB; ++b) {
            cb_reserve_back(cb_xb, 1);
            const uint32_t page = get_write_ptr(cb_xb);
            for (uint32_t a = 0; a < 32; ++a) {
                noc_async_read(x_acc.get_noc_addr(32 * b + a), xst + a * XROW, XROW);
            }
            noc_async_read_barrier();
            for (uint32_t a = 0; a < 32; ++a) {
                const uint32_t m = mk + (32 * b + a) * 64;
                copy_xor(page + row_off(a), xst + a * XROW, m, 32);
                copy_xor(page + row_off(a) + 512, xst + a * XROW + 512, m + 32, 32);
            }
            cb_push_back(cb_xb, 1);
        }
        cb_wait_front(cb_y, NB);
        ybase = get_read_ptr(cb_y);
    }

    uint32_t slot = 0;  // the ring page of the chunk's first x tile
    for (uint32_t c = 0; c < NCH; ++c) {
        const uint32_t kt0 = c * KW;
        if (kt0 % PT == 0) {
            cb_reserve_back(cb_s, NC);
            const uint32_t ls = get_write_ptr(cb_s);
            for (uint32_t j = 0; j < nvalid; ++j) {
                noc_async_read_page((kt0 / PT) * nst + n0 + j, s_acc, ls + j * ts);
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, NC);
        }
        if constexpr (NB == 0 && !XDMA) {
            for (uint32_t k = 0; k < KW; ++k) {
                noc_async_read(x_acc.get_noc_addr(kt0 + k), xst + k * XROW, XROW);
            }
        }
        const bool mine = !SPLIT || (c & 1) == 0;
        if (mine) {
            ld.issue(c);
        }
        cb_reserve_back(cb_x, KW);
        if constexpr (XDMA) {
            for (uint32_t k = 0; k < KW; ++k) {
                const uint32_t q = slot + k;
                const uint32_t page = ring + q * tx;
                const uint32_t r = ((kt0 + k) / GT) & 31;
                const uint32_t old = held[q];
                if (old != r) {
                    if (old < 32) {
                        noc_async_read(zeros, page + row_off(old), 32);
                        noc_async_read(zeros, page + row_off(old) + 512, 32);
                    }
                    held[q] = r;
                }
                uint64_t src;
                if constexpr (NB > 0) {
                    const uint32_t t = kt0 + k;
                    src = get_noc_addr(ybase + (t >> 5) * tx + row_off(t & 31));
                } else {
                    src = x_acc.get_noc_addr(kt0 + k);
                }
                noc_async_read(src, page + row_off(r), 32);
                noc_async_read(src + 512, page + row_off(r) + 512, 32);
            }
        }
        noc_async_read_barrier();
        if (mine) {
            if constexpr (DB < TB) {
                ld.place();
                noc_async_read_barrier();
            }
            ld.push();
        }
        for (uint32_t k = 0; k < KW && !XDMA; ++k) {
            const uint32_t q = slot + k;
            const uint32_t page = ring + q * tx;
            const uint32_t r = ((kt0 + k) / GT) & 31;
            const uint32_t old = held[q];
            if (old != r) {
                if (old < 32) {
                    fill_words(page + row_off(old), 0, 32);
                    fill_words(page + row_off(old) + 512, 0, 32);
                }
                held[q] = r;
            }
            if constexpr (NB > 0) {
                const uint32_t t = kt0 + k;
                const uint32_t y = ybase + (t >> 5) * tx + row_off(t & 31);
                copy_words(page + row_off(r), y, 32);
                copy_words(page + row_off(r) + 512, y + 512, 32);
            } else {
                copy_words(page + row_off(r), xst + k * XROW, 32);
                copy_words(page + row_off(r) + 512, xst + k * XROW + 512, 32);
            }
        }
        slot = (slot + KW) % XPAGES;
        cb_push_back(cb_x, KW);
    }
    if constexpr (NB > 0) {
        cb_pop_front(cb_y, NB);
    }
}
