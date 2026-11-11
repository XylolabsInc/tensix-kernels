// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Xylolabs
//
// Writer of moe8's down: the core's ncols output tiles to y's tiles col0 .. (y [1, H]).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr auto y_args = TensorAccessorArgs<1>();
    uint32_t i = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(i++);
    const uint32_t col0 = get_arg_val<uint32_t>(i++);
    const uint32_t ncols = get_arg_val<uint32_t>(i++);
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
