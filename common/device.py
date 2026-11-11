# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Copyright 2026 Xylolabs
# Modified by Xylolabs, 2026: the DRAM read-VC assignment (read_vcs) reimplemented in Python, next to helpers of
# our own. Adapted from tenstorrent/tt-metal v0.81.1,
# ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp.
# SPDX-License-Identifier: Apache-2.0
"""Device-side helpers shared by the packages: DRAM banks, width-sharded weight layouts, core placement, read VCs and
the split loaders' chunk pattern.

The bank count, the grid and the L1 come from the device; every value a caller may want to change is a parameter
or a module attribute.
"""

import math
from fractions import Fraction

import ttnn

TILE = 32

FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}

# Caps the banks a weight is sharded over (None: every bank the device reports).
MAX_BANKS = None

# Core placement for weights sharded over DRAM banks: "optimal" puts each bank's cores next to the device's preferred
# worker for that bank (get_optimal_dram_bank_to_logical_worker_assignment), else row-major ("rows"). Where the device
# cannot give that assignment for its compute grid, row-major.
PLACE = "optimal"


def dram_banks(device, max_banks=None):
    """The DRAM banks weights are sharded over: the device's, at most max_banks (default MAX_BANKS)."""
    g = device.dram_grid_size()
    n = g.x * g.y
    cap = max_banks if max_banks is not None else MAX_BANKS
    return min(n, cap) if cap else n


def bank_grid(banks):
    """The shard grid of a weight width-sharded over DRAM banks 0 .. banks - 1."""
    assert banks >= 1, banks
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})


def is_dram_sharded(t):
    mc = t.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def check_banks(t, max_banks=None):
    """ValueError for a DRAM width-sharded tensor over more banks than max_banks (default MAX_BANKS; None: no limit)."""
    cap = max_banks if max_banks is not None else MAX_BANKS
    if t is not None and cap and is_dram_sharded(t) and t.memory_config().shard_spec.num_cores() > cap:
        raise ValueError(
            f"a weight sharded over {t.memory_config().shard_spec.num_cores()} DRAM banks (at most {cap}: max_banks)"
        )


def sharded_memcfg(device, k, n, banks=None, max_banks=None):
    """DRAM width-sharded over banks 0 .. banks - 1 (default: dram_banks), one column block of K rows a bank, N padded
    to a multiple of banks * 32."""
    banks = banks or dram_banks(device, max_banks)
    padded = math.ceil(n / (TILE * banks)) * TILE * banks
    spec = ttnn.ShardSpec(bank_grid(banks), (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


_PLACE = {}


def place(device, banks, sharded):
    """A logical core per work item, banks[i] being item i's DRAM bank: with sharded weights each bank's items on the
    free cores nearest to the device's preferred worker of that bank (NoC 1, the reader's), else row-major."""
    g = device.compute_with_storage_grid_size()
    if not sharded or PLACE != "optimal":
        return [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(len(banks))]
    key = (id(device), tuple(banks))
    if key not in _PLACE:
        try:
            best = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_1)
        except (RuntimeError, AttributeError):  # no assignment for this compute grid
            best = None
        if best is None or len(best) <= max(banks):
            _PLACE[key] = [(i % g.x, i // g.x) for i in range(len(banks))]
            return [ttnn.CoreCoord(cx, cy) for cx, cy in _PLACE[key]]
        free = {(cx, cy) for cx in range(g.x) for cy in range(g.y)}
        out = []
        for b in banks:
            o = best[b]
            c = min(free, key=lambda f, o=o: (abs(f[0] - o.x) + abs(f[1] - o.y), f[1], f[0]))
            free.discard(c)
            out.append(c)
        _PLACE[key] = out
    return [ttnn.CoreCoord(cx, cy) for cx, cy in _PLACE[key]]


def read_vcs(work):
    """Each core's DRAM read virtual channel, as tt-metal's DRAM-sharded matmul assigns them
    (matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp): bank & 3, moved on while an earlier core of the
    same row has it. work: [(core, (bank, ...)), ...]."""
    out = []
    for i, (c, (bank, *_)) in enumerate(work):
        vc = bank & 3
        for j in range(i):
            if work[j][0].y == c.y and out[j] == vc:
                vc = (vc + 1) & 3
        out.append(vc)
    return out


def split_arg(split, share=0.5):
    """The SPLIT compile argument (kernels/split.h) for a reader's share of the weight chunks: 0 without split, 1
    for 0.5 (the alternation), else (P << 24) | MASK, the writer's share as a fraction of a period P <= 24 and its
    positions in the period, spread, never 0 and never two in a row. Shares from 0.5 to 0.75 are taken."""
    if not split:
        return 0
    share = float(share)
    assert 0.5 <= share <= 0.75, f"a split share of {share} (the reader's; 0.5 to 0.75)"
    if share == 0.5:
        return 1
    f = Fraction(1 - share).limit_denominator(24)  # the writer's share
    P, nw = f.denominator, f.numerator
    pos = [(2 * i + 1) * P // (2 * nw) for i in range(nw)]
    mask = sum(1 << q for q in pos)
    assert 3 <= P <= 24 and nw >= 1 and mask & 1 == 0 and mask & (mask >> 1) == 0, (share, P, pos)
    return (P << 24) | mask


def l1_free(device):
    """Bytes of L1 a core's circular buffers may take now: the largest free L1 span per bank (L1 tensors are
    allocated from the top), or None where the allocator does not say."""
    try:
        return ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    except (AttributeError, RuntimeError):
        return None
