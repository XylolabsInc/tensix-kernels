// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the DeltaNet's conv kernel (gdn_conv_compute.cpp): per group and token t, row j of the
// packed output tile (item j's 32 channels) into row t of that item's output page, the page's rows
// past T zero (as upstream's output has them); the pages into o, [B, T, C], once the group's T tokens
// are in.

#include <cstdint>
// DeviceZoneScopedN marks profiler zones and has no effect when profiling is disabled.
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_wo = 22;  // writer-only: the group's output pages

inline uint32_t tidx(uint32_t i, uint32_t j) { return ((i >> 4) << 9) + ((i & 15) << 4) + ((j >> 4) << 8) + (j & 15); }

// bytes at addr zeroed by NOC reads of the core's zeros region (MEM_ZEROS_SIZE a read; bytes a
// multiple of it); the caller waits (noc_async_read_barrier) before it writes there.
__attribute__((noinline)) void noc_zero(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(z, addr + off, MEM_ZEROS_SIZE);
    }
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    uint32_t i = 0;
    const uint32_t i0 = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t CT = get_arg_val<uint32_t>(i++);
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);

    const uint32_t to = get_tile_size(cb_wo);
    const auto o_acc = TensorAccessor(o_a, o_addr, to);
    const uint32_t wo = get_write_ptr(cb_wo);
// Keep output rows past T zero with NOC reads of the core's zeros region.
// Clear only the min(n, 32) pages allocated to the CB.
    {
        DeviceZoneScopedN("W-czero");
        noc_zero(wo, (n < 32 ? n : 32) * to);
        noc_async_read_barrier();
    }

    for (uint32_t j0 = 0; j0 < n; j0 += 32) {
        const uint32_t m = (n - j0 < 32) ? (n - j0) : 32;
        for (uint32_t t = 0; t < T; t++) {
            DeviceZoneScopedN("W-crows");
            cb_wait_front(cb_out, 1);
            invalidate_l1_cache();
            const uint32_t ro = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < m; j++) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ro + tidx(j, 0) * 2);
                volatile tt_l1_ptr uint32_t* d =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wo + j * to + tidx(t, 0) * 2);
                for (uint32_t w = 0; w < 8; w++) {
                    d[w] = s[w];
                    d[w + 128] = s[w + 128];
                }
            }
            cb_pop_front(cb_out, 1);
        }
        asm volatile("fence" ::: "memory");
        DeviceZoneScopedN("W-cwrite");
        for (uint32_t j = 0; j < m; j++) {
            const uint32_t it = i0 + j0 + j;
            noc_async_write_page((it / CT) * CT + it % CT, o_acc, wo + j * to);
        }
        noc_async_write_barrier();
    }
}
