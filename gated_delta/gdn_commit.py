# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Commit speculative state with page-for-page copies in one device launch.

Each destination becomes a bit-for-bit copy of its source. For example, copy each DeltaNet
layer's state slot and conv slot after accepting a draft row. kernels/gdn_commit_{reader,writer}.cpp
lay the pairs' pages end to end, split them evenly over cores, and copy them in batches.

  commit([(src, dst), ...])

Every tensor is interleaved in DRAM; a pair's two tensors have the same shape, dtype and layout. The pair table (src
address, dst address, pages, page bytes per pair) is a small DRAM tensor made once per list of pairs (cached by the
tensors' ids and held with them), so a trace captured around commit() replays it."""

import ttnn
from gdn_scan import source

BATCH = 8  # pages a batch
CB = {"c": 0, "rt": 1, "wt": 2}
_TABLES = {}


def _pages(t):
    """Return the page count and page size: one tile for TILE, one row for ROW_MAJOR."""
    shape = list(t.padded_shape)
    n = 1
    for d in shape[:-1]:
        n *= d
    if t.layout == ttnn.TILE_LAYOUT:
        tb = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[
            t.dtype
        ]
        return (n // 32) * (shape[-1] // 32), tb
    eb = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint32: 4, ttnn.int32: 4}[t.dtype]
    return n, shape[-1] * eb


def _table(device, pairs):
    key = (id(device), *(id(t) for pr in pairs for t in pr))
    if key not in _TABLES:
        import torch

        rows = []
        for src, dst in pairs:
            n, pb = _pages(src)
            assert (n, pb) == _pages(dst) and src.dtype == dst.dtype and src.layout == dst.layout, (
                src.shape,
                dst.shape,
            )
            rows += [src.buffer_address(), dst.buffer_address(), n, pb]
        width = -(-len(rows) // 8) * 8  # 32-byte aligned
        tab = torch.zeros(1, width, dtype=torch.int32)
        tab[0, : len(rows)] = torch.tensor(
            [r if r < 2**31 else r - 2**32 for r in rows], dtype=torch.int32
        )
        dev = ttnn.from_torch(
            tab,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _TABLES[key] = (pairs, dev, width * 4, [_pages(s) for s, _ in pairs])
    return _TABLES[key][1:]


def commit(pairs, ncores=None):
    pairs = [tuple(p) for p in pairs]
    assert pairs, "gdn_commit: no pairs"
    for src, dst in pairs:
        for t in (src, dst):
            mc = t.memory_config()
            assert not mc.is_sharded() and mc.buffer_type == ttnn.BufferType.DRAM, (
                "gdn_commit: interleaved DRAM only"
            )
    device = pairs[0][0].device()
    tab, tbytes, pages = _table(device, pairs)
    total = sum(n for n, _ in pages)
    pmax = max(pb for _, pb in pages)
    g = device.compute_with_storage_grid_size()
    nc = min(ncores or g.x * g.y, g.x * g.y, total)
    base, extra = divmod(total, nc)
    work, start = [], 0
    for c in range(nc):
        cnt = base + (1 if c < extra else 0)
        # the pair and page this core starts at
        p, off = 0, start
        while off >= pages[p][0]:
            off -= pages[p][0]
            p += 1
        work.append((ttnn.CoreCoord(c // g.y, c % g.y), p, off, cnt))
        start += cnt
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, *_ in work])
    sizes = {"c": (2 * BATCH, pmax), "rt": (1, tbytes), "wt": (1, tbytes)}
    cbs = [
        ttnn.CBDescriptor(
            total_size=k * pb,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB[n], data_format=ttnn.bfloat16, page_size=pb)
            ],
        )
        for n, (k, pb) in sizes.items()
    ]
    ct = [BATCH, tbytes, pmax]
    ct.extend(ttnn.TensorAccessorArgs(pairs[0][0]).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(tab).get_compile_time_args())
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, p, off, cnt in work:
        rd[c.x][c.y] = [tab.buffer_address(), p, off, cnt]
        wr[c.x][c.y] = [tab.buffer_address(), p, off, cnt]

    def kernel(name, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("gdn_commit_reader.cpp", rd, ttnn.ReaderConfigDescriptor()),
            kernel("gdn_commit_writer.cpp", wr, ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[],
        cbs=cbs,
    )
    io = [tab] + [t for pr in pairs for t in pr]
    seen = set()
    io = [t for t in io if not (id(t) in seen or seen.add(id(t)))]
    ttnn.generic_op(io, program)
