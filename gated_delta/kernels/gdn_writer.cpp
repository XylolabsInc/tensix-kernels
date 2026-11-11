// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Xylolabs
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the gated delta rule kernel (gdn_compute.cpp): per head its output chunks, then its
// final state. Adapted from tenstorrent/tt-metal v0.81.1 (Apache-2.0)
// chunk_gated_delta_rule/device/kernels/dataflow/writer_chunk_gated_delta_rule.cpp, written for the
// upstream dataflow API.
//
//   o      flat [B, T, HV*V] (o_flat = 1) or head-major [B*HV, T, V]; tile rows past the tensor
//          (chunk padding) are not written
//   state  [B*HV, K, V]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 26, cb_final = 27;

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t h0 = get_arg_val<uint32_t>(i++);
    const uint32_t nh = get_arg_val<uint32_t>(i++);
    const uint32_t NC = get_arg_val<uint32_t>(i++);
    const uint32_t Tt = get_arg_val<uint32_t>(i++);
    const uint32_t HV = get_arg_val<uint32_t>(i++);
    const uint32_t o_flat = get_arg_val<uint32_t>(i++);
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t s_addr = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cv = Ct * Vt, kv = Kt * Vt;
    const uint32_t to = get_tile_size(cb_out);
    const uint32_t ts = get_tile_size(cb_final);
    const auto o_acc = TensorAccessor(o_a, o_addr, to);
    const auto s_acc = TensorAccessor(s_a, s_addr, ts);

    for (uint32_t hh = 0; hh < nh; hh++) {
        const uint32_t bh = h0 + hh;
        const uint32_t b = bh / HV;
        const uint32_t hv = bh % HV;
        const uint32_t o_row = o_flat ? HV * Vt : Vt;
        const uint32_t o_base = o_flat ? (b * Tt * HV * Vt + hv * Vt) : (bh * Tt * Vt);
        for (uint32_t c = 0; c < NC; c++) {
            cb_wait_front(cb_out, cv);
            const uint32_t rd = get_read_ptr(cb_out);
            for (uint32_t r = 0; r < Ct; r++) {
                const uint32_t rt = c * Ct + r;
                if (rt < Tt) {
                    for (uint32_t t = 0; t < Vt; t++) {
                        noc_async_write_page(o_base + rt * o_row + t, o_acc, rd + (r * Vt + t) * to);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, cv);
        }
        cb_wait_front(cb_final, kv);
        const uint32_t rd = get_read_ptr(cb_final);
        for (uint32_t t = 0; t < kv; t++) {
            noc_async_write_page(bh * kv + t, s_acc, rd + t * ts);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_final, kv);
    }
}
