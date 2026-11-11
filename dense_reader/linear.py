# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""linear.py: a weight-streaming matmul for decode on Tenstorrent Tensix devices, launched through ttnn.generic_op.

y = x @ w (* scale), x bf16 activations, w block-float (bfloat8_b, bfloat4_b) or bfloat16 weights, fp32 accumulation,
bf16 out (or float32). Each DRAM bank's columns of a DRAM width-sharded weight are read by `cores_per_bank` cores, each K row of a core's columns in one contiguous
request straight into its circular buffer; for decode (up to 32 rows) x streams with the weight chunks instead of
waiting in front of them. The reader and the writer RISC of each core can both load weight chunks (split loaders),
one request stream per NoC. Fused output stages: per-column scales, act(gate) * up (SwiGLU, GeGLU), a residual add, the
fp32 accumulator as it is, and a running top-k of each output row kept by the writers (no logits written).

Entry points:
- matmul(x, w, scale, ...): the op; x [1, 1, M, K] bf16 TILE interleaved (DRAM or L1).
- linear(x, w, scale, activation=...): the same for x of any rank, with "swiglu" / "geglu" / "silu".
- native_weight(device, t, dtype): a float [out, in] matrix as a device weight [1, 1, in, out] (DRAM width-sharded)
  and its scale tensor.
- interleave_glu(gate, up): [gate | up] with output tiles alternating, for the fused GLU stage.
- topk_result, topk_records, records_host: the top-k records' tensor and their decode on the host.

The kernels (kernels/) are compiled by tt-metal's JIT at the first call; nothing outside this directory and common/ is
needed beyond ttnn itself.
"""

import itertools
import re
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device as dev  # noqa: E402
from common import source as src  # noqa: E402
from common.device import (  # noqa: E402,F401  (re-exported for the other packages)
    FIDELITY,
    TILE,
    bank_grid,
    check_banks,
    dram_banks,
    is_dram_sharded,
    place,
    read_vcs,
    sharded_memcfg,
    split_arg,
)

KERNELS = Path(__file__).resolve().parent / "kernels"
BFP8_TILE = 1088
NATIVE_TILES = {ttnn.bfloat8_b: BFP8_TILE, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}

# --- settings (module attributes: starting points; retune per device) ---

# Bytes of L1 a core's circular buffers may take: None, the default: the device's largest free L1 span per bank at the
# first plan, less L1_MARGIN; a number caps it. Plans are made for this budget and kept while the L1 free at a call
# holds them (an L1 tensor allocated later lowers the free span; a plan that no longer fits is made again).
L1_BUDGET = None
L1_MARGIN = 32 * 1024
# the x block a core holds in L1 for prefill (bf16 tiles), at most
X_BUDGET = 600_000
# x groups (MT x KW tiles a loader) the reader may run ahead of the compute when x streams with the weights
XS_DEPTH = 3
# The fewest weight chunks per core the default chunk depth aims for, when x streams (decode): 0, the default, off.
# With a value, KW is halved (to the next divisor of K's tiles) until a core has at least MIN_CHUNKS chunks or KW is
# 4. MIN_CHUNKS_SHAPES: a set of (K, N) weight shapes the rule is limited to (None: every shape).
MIN_CHUNKS = 0
MIN_CHUNKS_SHAPES = None
# 1 core per bank first for decode shapes (one tile row) with N at most NARROW_N (0, the default: off)
NARROW_N = 0
# The share of a split plan's weight chunks the reader RISC loads (0.5, the default: the alternation; 0.5 to 0.75,
# kernels/split.h).
SPLIT_SHARE = 0.5
# a weight row's DRAM requests, at most REQ_TILES tiles each (0, the default: one request a row of the core's columns)
REQ_TILES = 0
# the GLU stage's SFPU work on the valid rows only (one tile row of x, at most 16 rows), else on whole tiles
GLU_ROWS = True
# the top-k result tensor as bfloat16 (the same bytes as the uint32 words the writers write; False: uint32)
RECORDS_BF16 = False
# programs cached by everything that shapes them (a hit only rewrites the tensors' addresses)
CACHE = True
# a line per planned shape on stdout
LOG = False

# The configurations tried in order when matmul is not told one: two loaders, full-row chunks, 2 cores per bank
# first; the others for shapes whose buffers do not fit L1 with it (prefill slices, wide N).
CONFIGS = (
    {"cores_per_bank": 2, "split": True, "rows": True},
    {"cores_per_bank": 2, "split": True, "rows": True, "kw": 1},
    {"cores_per_bank": 3, "split": True, "rows": True},
    {"cores_per_bank": 3, "split": True, "rows": True, "kw": 1},
    {"cores_per_bank": 2, "split": True},
    {"cores_per_bank": 3, "split": True},
    {"cores_per_bank": 3},
)

_PROGRAMS = {}
_CHOICE = {}
_FULL = {}  # device id -> the full budget (L1_BUDGET None)
_CAP = {}  # device id -> a lower cap after a circular-buffer clash


def source(name):
    """A kernel's source with its local headers inlined (common/source.py)."""
    return src.load(KERNELS, name)


# --- host side --------------------------------------------------------------------------------------------------------


def bf16_ceil(s):
    """The smallest bf16 value >= s (s > 0, float32)."""
    b = s.to(torch.bfloat16).float()
    up = (b.view(torch.int32) + 0x10000).view(torch.float32)
    return torch.where(b >= s, b, up)


def native_tile(w):
    """Tile bytes of a weight this matmul reads (bfloat8_b, bfloat4_b, bfloat16), else 0."""
    return NATIVE_TILES.get(w.dtype, 0)


def interleave_glu(gate, up):
    """[gate | up] for the fused GLU stage: two float [out, in] matrices as one [2 * out, in] whose output tiles
    alternate, tile 2j the gate's rows 32j..32j+31 and tile 2j + 1 the up's. Whole tiles move, so values already on a
    block-float grid keep their exponent groups."""
    assert gate.shape == up.shape and gate.shape[0] % TILE == 0, (gate.shape, up.shape)
    o, i = gate.shape
    g = gate.reshape(o // TILE, TILE, i)
    u = up.reshape(o // TILE, TILE, i)
    return torch.stack([g, u], dim=1).reshape(2 * o, i)


def scale_tensor(device, s):
    """Per-output-column scales s [N] as [1, 1, 1, N] bf16 TILE in DRAM (row 0 of each tile used)."""
    return ttnn.from_torch(
        s.reshape(1, 1, 1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def native_weight(device, t, dtype, col_scale=False, sharded=True, max_banks=None):
    """A float [out, in] matrix as a weight [1, 1, in, out] of dtype (bfloat8_b, bfloat4_b or bfloat16), DRAM
    width-sharded over the device's banks (at most max_banks) or interleaved, and its [1, 1, 1, out] scales: ones, or
    with col_scale each output column divided by c = max |w| (rounded up to bf16) before the block-float rounding and
    multiplied back after the matmul, so columns of very different size no longer share one coarse exponent."""
    t = t.float()
    c = bf16_ceil(t.abs().amax(dim=1).clamp(min=1e-8)) if col_scale else torch.ones(t.shape[0])
    wt = (t / c[:, None]).T.contiguous()
    k, n = wt.shape
    mc = sharded_memcfg(device, k, n, max_banks=max_banks) if sharded else ttnn.DRAM_MEMORY_CONFIG
    # (from_torch converts float32 to the block-float grid directly; as_tensor would round to bf16 first)
    w = ttnn.from_torch(wt.reshape(1, 1, k, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    return w, scale_tensor(device, c)


def block_float(t, dtype):
    """t (float [..., rows, cols]) rounded to dtype's grid by ttnn's host conversion, as float32 (the values a weight
    of that dtype holds)."""
    h = ttnn.from_torch(t.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(h).float()


def reference(x, w, scale=None):
    """x @ w (* scale) in float32, x as the device sees it (bf16); w [K, N] as the weight holds it (block_float)."""
    y = x.to(torch.bfloat16).float() @ w.float()
    return y * scale.float()[None, :] if scale is not None else y


# --- the plan ---------------------------------------------------------------------------------------------------------


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def row_slices(x, full_sync=False):
    """[(first tile row, tile rows)] of x's rows, at most 4 tile rows each (8 with full_sync): one program per slice,
    each reading all the weights once."""
    mt = -(-x.padded_shape[-2] // TILE)
    cap = 8 if full_sync else 4
    return [(m0, min(cap, mt - m0)) for m0 in range(0, mt, cap)]


def plan(
    device,
    x,
    w,
    cores_per_bank=3,
    full_sync=False,
    kb=None,
    kw=None,
    sbw=None,
    split=False,
    mt=None,
    rows=False,
    budget=None,
    xs=False,
    noscale=False,
    glu=False,
    glu2=False,
    out32=False,
    blocks=None,
    max_banks=None,
):
    """The work split of one row slice of mt tile rows (default: all of x's): {MT, KT, KB, KW, SBW, NC, W_PAGES,
    sharded, split, work: [(core, (bank, pitch, col0, nvalid, n0))], l1, ...}. AssertionError when it does not fit.
    xs (with rows): x streams with the weight chunks (XS_DEPTH groups of MT x KW tiles in L1) instead of an x block of
    KB tiles in front of them. noscale: no scale tensor (unit scales), so no scale tiles in L1. glu: the columns are
    gate and up tiles in turn (interleave_glu); every core's columns are whole pairs and the subblocks even. glu2 (with
    glu, rows and a sharded w): gate and up as two weights of one sharding, w the gate's: planned as the virtual [gate |
    up] of pairs, each bank's ns gate and ns up tiles, each chunk row the gate's columns then the up's
    (dense_wload.h HALF). blocks (matmul's block list; w the first): weights of one K, dtype and banks (the gates
    with glu2), each DRAM-sharded on its own: each core runs one segment per block, its slice of the bank's shard of
    that block (cores_per_bank slices a bank, each block cut on its own), in sequence (SEGS); NC is the widest
    segment's (the others padded); p["seg_work"][i]: core i's (pitch, col0, nvalid, n0) of the blocks after the
    first, whose own are in work."""
    blocks = tuple(blocks) if blocks else (w,)
    SEGS = len(blocks)
    if SEGS > 1:
        assert 2 <= SEGS <= 4, f"dense_reader: {SEGS} blocks (at most 4)"
        assert xs and rows and noscale and not out32 and (glu2 or not glu), (
            "dense_reader: a block list streams x (up to 32 rows) with unit-scale weights, plain or (gate, up) pairs"
        )
        assert blocks[0] is w and all(is_dram_sharded(b) for b in blocks), "dense_reader: blocks are DRAM-sharded"
        assert len({(b.shape[-2], b.dtype, b.memory_config().shard_spec.num_cores()) for b in blocks}) == 1, (
            "dense_reader: blocks of one K, dtype and bank count"
        )
        # the kernels read shard b of every block from bank b (one bank argument a core): every block on the same
        # shard grid in the same orientation
        specs = [b.memory_config().shard_spec for b in blocks]
        assert len({(str(sp.grid), str(sp.orientation)) for sp in specs}) == 1, (
            "dense_reader: blocks sharded on one grid in one orientation"
        )
    assert not glu2 or (glu and rows), "dense_reader: glu2 needs glu and rows"
    assert not (out32 and (glu or not noscale)), "dense_reader: out32 needs unit scales, no GLU stage"
    native = native_tile(w)
    assert native, f"dense_reader: weights are bfloat8_b, bfloat4_b or bfloat16, not {w.dtype}"
    K, N = w.shape[-2], w.shape[-1]
    KT, NT = K // TILE, -(-N // TILE)
    M = x.shape[-2]
    MT = mt or -(-M // TILE)
    assert K % TILE == 0 and x.shape[-1] == K, (x.shape, w.shape)
    g = device.compute_with_storage_grid_size()
    ncores = g.x * g.y
    sharded = is_dram_sharded(w)
    seg_layout = []  # SEGS > 1: each block's (pitch, ns, groups)
    if sharded:
        for wb in blocks:
            spec = wb.memory_config().shard_spec
            ns = spec.shape[1] // TILE
            banks = spec.num_cores()
            assert spec.shape[0] == K, (spec.shape, K)
            pitch = ns  # the weight's own tiles per shard row (glu2: the gate's and the up's)
            NT = -(-wb.shape[-1] // TILE)
            if glu2:
                ns, NT = 2 * ns, 2 * NT  # the virtual [gate | up]: whole pairs in every bank
            groups = [(b, ns, b * ns, max(0, min(ns, NT - b * ns))) for b in range(banks)]
            seg_layout.append((pitch, ns, groups))
        pitch, ns, groups = seg_layout[0]
        NT = -(-N // TILE) * (2 if glu2 else 1)  # (the first block's, as without blocks)
    else:
        banks = dram_banks(device, max_banks)
        ns = -(-NT // banks)
        groups = [(0, ns, b * ns, max(0, min(ns, NT - b * ns))) for b in range(banks)]
        pitch = NT
    cpb = max(1, min(cores_per_bank, ncores // len(groups)))

    def per_core(ns_):
        per_ = -(-ns_ // cpb)
        if glu:
            assert sharded and ns_ % 2 == 0, f"dense_reader: glu needs whole gate/up pairs per bank ({ns_})"
            per_ += per_ % 2  # whole pairs per core
        return per_

    pers = [per_core(ns_) for _, ns_, _ in seg_layout] if SEGS > 1 else [per_core(ns)]
    per = max(pers)  # (SEGS > 1: the widest segment sets SBW and NC)
    cap = (8 if full_sync else 4) // MT
    assert cap >= 1, f"dense_reader: {M} rows need more dest than {8 if full_sync else 4} fp32 tiles"
    if sbw is None:
        # unpack work per column: the weight tile plus 1/SBW of an x tile (srcB reuse); padded columns are computed
        # but never read from DRAM
        widths = [w_ for w_ in range(1, cap + 1) if not glu or w_ % 2 == 0]
        assert widths, "dense_reader: glu needs subblocks of 2 or 4 tiles"
        sbw = min(widths, key=lambda w_: (-(-per // w_) * w_ * (1 + 1 / w_), -w_))
    NC = -(-per // sbw) * sbw
    work = []
    seg_work = []
    if SEGS > 1:
        # core (bank b, j): slice j of bank b's shard of every block, each block cut by its own per-core width; a core
        # with columns in any block runs every segment (one with none reads no weights and writes nothing)
        for b in range(len(groups)):
            for j in range(cpb):
                segs = []
                for (pitch_, _, groups_), per_ in zip(seg_layout, pers, strict=True):
                    g0_, valid_ = groups_[b][2], groups_[b][3]
                    segs.append((pitch_, j * per_, max(0, min(per_, valid_ - j * per_)), g0_ + j * per_))
                if any(sg[2] for sg in segs):
                    work.append((b, *segs[0]))
                    seg_work.append(tuple(segs[1:]))
    for bank, _, g0, valid in groups if SEGS == 1 else ():
        for j in range(cpb):
            nvalid = max(0, min(per, valid - j * per))
            if nvalid == 0:
                continue
            col0 = j * per if sharded else g0 + j * per
            work.append((bank, pitch, col0, nvalid, g0 + j * per))
    assert len(work) <= ncores, (len(work), ncores)
    cores = place(device, [wk[0] for wk in work], sharded)
    assert not glu2 or sharded, "dense_reader: glu2 needs sharded weights"
    page = native
    budget = budget or full_budget(device)
    width = NC if rows else sbw  # tiles per weight row of a chunk
    loaders = 2 if split else 1
    xs = xs and rows

    def sized(kb_, kw_):
        """(kw, W_PAGES, l1) of chunks of kw rows."""
        ch_ = kw_ * width
        depth = 2 if ch_ > 64 else (3 if ch_ > 16 else 4)
        w_pages_ = depth * ch_
        x_tiles = MT * kw_ * loaders * XS_DEPTH if xs else MT * kb_
        l1_ = x_tiles * 2048 + loaders * w_pages_ * page + (0 if noscale else NC * 2048)
        l1_ += MT * NC * SEGS * 4096  # (the accumulator: every segment's columns)
        l1_ += 2 * sbw * (4096 if out32 else 2048)
        return kw_, w_pages_, l1_

    def sizes(kb_):
        kw_ = kw
        if kw_ is None:
            # a chunk of at most 128 BFP8-sized pages (rows) or 32; smaller tiles take more
            cap_tiles = (128 if rows else 32) * BFP8_TILE // page
            kw_ = max([d for d in _divisors(kb_) if d * width <= cap_tiles and d <= 16] or [1])
            # decode (x streamed): smaller chunks, so a core has at least MIN_CHUNKS of them (MIN_CHUNKS_SHAPES)
            while (
                MIN_CHUNKS
                and xs
                and (MIN_CHUNKS_SHAPES is None or (w.shape[-2], w.shape[-1]) in MIN_CHUNKS_SHAPES)
                and kb_ // kw_ < MIN_CHUNKS
                and kw_ > 4
            ):
                smaller = [d for d in _divisors(kb_) if d <= kw_ // 2]
                if not smaller:
                    break
                kw_ = max(smaller)
        return sized(kb_, kw_)

    if xs:
        kb = KT  # one block: x streams per chunk
    if kb is None:
        # the largest x block (fewest packer L1-accumulation passes) whose buffers fit
        cands = [d for d in sorted(_divisors(KT), reverse=True) if MT * d * 2048 <= X_BUDGET]
        kb = next((d for d in cands if sizes(d)[2] <= budget), cands[-1])
    if KT % kb:
        raise ValueError(f"dense_reader: kb {kb} does not divide K's {KT} tiles")
    kw, w_pages, l1 = sizes(kb)
    assert l1 <= budget, f"dense_reader: {l1} B of circular buffers per core (budget {budget})"
    return {
        "MT": MT,
        "KT": KT,
        "NT": NT,
        "KB": kb,
        "KW": kw,
        "SBW": sbw,
        "NC": NC,
        "W_PAGES": w_pages,
        "sharded": sharded,
        "split": split,
        "rows": rows,
        "native": native,
        "xs": xs,
        "noscale": noscale,
        "glu": glu,
        "glu2": glu2,
        "out32": out32,
        "SEGS": SEGS,
        "seg_work": tuple(seg_work),
        "work": list(zip(cores, work)),
        "l1": l1,
    }


def full_budget(device):
    """The circular buffers' L1 budget plans are made for: L1_BUDGET, or (None) the device's free L1 span at the first
    plan less L1_MARGIN."""
    if L1_BUDGET is not None:
        return L1_BUDGET
    key = id(device)
    if key not in _FULL:
        free = dev.l1_free(device)
        assert free is not None, "dense_reader: the allocator does not report free L1; set linear.L1_BUDGET"
        _FULL[key] = free - L1_MARGIN
    return _FULL[key]


def l1_free(device):
    """Bytes of L1 the circular buffers may take now: below the lowest L1 buffer, within the full budget, less
    L1_MARGIN, and under the cap a circular-buffer clash set."""
    free = dev.l1_free(device)
    full = full_budget(device)
    free = full if free is None else free - L1_MARGIN
    return min(full, _CAP.get(id(device), full), free)


def _configs(w, mt_all, forced):
    """CONFIGS updated by forced; for a decode shape (one tile row) of N <= NARROW_N, unless forced sets
    cores_per_bank, CONFIGS[0] and [1] with 1 core per bank first."""
    configs = list(CONFIGS)
    if NARROW_N and mt_all == 1 and w.shape[-1] <= NARROW_N and "cores_per_bank" not in forced:
        configs = [{**c, "cores_per_bank": 1} for c in CONFIGS[:2]] + configs
    return [{**c, **forced} for c in configs]


def _search(device, x, w, full_sync, forced, budget, why=None):
    """[(first tile row, plan)] of the first of _configs whose buffers fit budget, with row slices of 4, 2 or 1 tile
    rows (8 first with full_sync), taller first; None if none fits. why, a list: each distinct reason plan() gave for a
    refusal is appended."""
    mt_all = -(-x.padded_shape[-2] // TILE)
    configs = _configs(w, mt_all, forced)
    for h in (8, 4, 2, 1) if full_sync else (4, 2, 1):
        for cfg0 in configs:
            cfg = dict(cfg0)
            cpb = cfg.pop("cores_per_bank", 3)
            try:
                return [
                    (m0, plan(device, x, w, cpb, full_sync, mt=min(h, mt_all - m0), budget=budget, **cfg))
                    for m0 in range(0, mt_all, h)
                ]
            except AssertionError as e:
                reason = str(e) or "an assertion without a message"
                if why is not None and reason not in why:
                    why.append(reason)
    return None


def _log_shape(x, w, forced):
    blocks = forced.get("blocks")
    n = " + ".join(str(b.shape[-1]) for b in blocks) if blocks else w.shape[-1]
    return f"K {w.shape[-2]} N {n} rows {x.shape[-2]}"


def _shape_key(w):
    """A weight's address-free identity for the caches: shape, layout and dtype."""
    return (tuple(w.padded_shape), str(w.memory_config()), w.dtype)


def _settings():
    """The module settings a plan depends on (part of the plan cache's key)."""
    mcs = None if MIN_CHUNKS_SHAPES is None else tuple(sorted(MIN_CHUNKS_SHAPES))
    return (L1_BUDGET, X_BUDGET, XS_DEPTH, MIN_CHUNKS, mcs, NARROW_N, tuple(map(str, CONFIGS)), dev.PLACE)


def choose(device, x, w, full_sync=False, **forced):
    """[(first tile row, plan)] for x's rows. The plan is made for the full budget and kept as long as the L1 free at
    the call holds it, so a call whose free L1 moves a little does not get new block sizes (a new program and a JIT
    build); only when it does not fit is one made for the L1 free now (and kept for that budget)."""
    # the weight's shape and layout too: a freed weight's id and address come back for the next one, which must not
    # get the old plan. A block list (forced blocks=) is keyed by its blocks' shapes, layouts and dtype only.
    blocks = forced.get("blocks")
    key = (
        ("blocks", tuple(_shape_key(b) for b in blocks)) if blocks else (id(w), w.buffer_address(), *_shape_key(w)),
        tuple(x.padded_shape),
        full_sync,
        tuple(sorted((k, v) for k, v in forced.items() if k != "blocks")),
        _settings(),
    )
    full = full_budget(device)
    free = l1_free(device)
    for budget in sorted({b for k, b in _CHOICE if k == key} | {full}, reverse=True):
        got = _CHOICE.get((key, budget))
        if got is None and budget == full:
            got = _CHOICE[(key, budget)] = _search(device, x, w, full_sync, forced, budget)
            if LOG and got:
                p0 = got[0][1]
                print(
                    f"dense_reader: {_log_shape(x, w, forced)}: plan cores {len(p0['work'])} SEGS {p0['SEGS']}"
                    f" split {p0['split']} xs {p0['xs']} KW {p0['KW']} NC {p0['NC']} W_PAGES {p0['W_PAGES']}"
                    f" l1 {p0['l1']}",
                    flush=True,
                )
        if got is not None and max(p["l1"] for _, p in got) <= free:
            return got
    why = []
    out = _search(device, x, w, full_sync, forced, free, why)
    if out is None:
        raise RuntimeError(
            f"dense_reader: no configuration fits the {free} B of L1 free for x {x.padded_shape}, w {w.shape}; the planner's"
            " refusals: "
            + ("; ".join(why[:6]) or "none recorded")
            + (f"; and {len(why) - 6} more" if len(why) > 6 else "")
        )
    _CHOICE[(key, free)] = out
    return out


class BlockListMismatch(ValueError):
    """A block list refused before its launch: its one call's K chunk depth KW differs from a block's own call's, so
    its output would not equal one call per block (out=, col0=) bit for bit. The message names the list's KW and each
    block's; the caller falls back to one call per block."""


def _block_kw_check(device, x, w, entries, full_sync, opts, slices):
    """BlockListMismatch unless every block's own call (choose on that block alone, as matmul would plan it: the
    same options, cached like any plan) has the block list's KW."""
    single = {k: v for k, v in opts.items() if k != "blocks"}
    kws = [choose(device, x, g, full_sync, **single)[0][1]["KW"] for g, _, _ in entries]
    kw = slices[0][1]["KW"]
    if any(k != kw for k in kws):
        msg = (
            f"dense_reader: {_log_shape(x, w, opts)}: block list refused: its KW {kw}, the blocks' own calls' KW {kws}"
            " (its output would not equal one call per block); one call per block instead"
        )
        if LOG:
            print(msg, flush=True)
        raise BlockListMismatch(msg)


def is_block_list(w):
    """True for matmul's block list [(w | (gate, up), col0_tiles), ...]: a list or tuple of 2 to 4 entries, each
    a (weight or (gate, up), int col0_tiles) pair; False for one weight and for linear's (gate, up) pair of weights."""
    return (
        isinstance(w, (list, tuple))
        and 2 <= len(w) <= 4
        and all(
            isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[1], int) and not isinstance(e[1], bool)
            for e in w
        )
    )


def block_entries(w):
    """A block list [(w | (gate, up), col0_tiles), ...] as [(gate, up or None, col0_tiles)], checked: two to four
    entries, all plain weights or all pairs, of one K, dtype, bank count, shard grid and orientation."""
    out = []
    for item in w:
        assert isinstance(item, (tuple, list)) and len(item) == 2, f"dense_reader: a block is (weight, col0_tiles): {item!r}"
        wt, c0 = item
        gate, up = (wt[0], wt[1]) if isinstance(wt, (tuple, list)) else (wt, None)
        assert isinstance(c0, int) and c0 >= 0, f"dense_reader: col0_tiles {c0!r}"
        if up is not None:
            assert up.dtype == gate.dtype and tuple(up.padded_shape) == tuple(gate.padded_shape), "gate and up differ"
            assert str(up.memory_config()) == str(gate.memory_config()), "dense_reader: gate and up differ"
        out.append((gate, up, c0))
    assert 2 <= len(out) <= 4, f"dense_reader: {len(out)} blocks (2 to 4; one weight goes in on its own)"
    assert len({up is None for _, up, _ in out}) == 1, "dense_reader: blocks all plain or all (gate, up) pairs"
    assert all(is_dram_sharded(g) for g, _, _ in out), "dense_reader: blocks are DRAM-sharded"
    layouts = set()
    for g, _, _ in out:
        sp = g.memory_config().shard_spec
        layouts.add((g.shape[-2], g.dtype, sp.num_cores(), str(sp.grid), str(sp.orientation)))
    assert len(layouts) == 1, (
        "dense_reader: blocks of one K, dtype, bank count, shard grid and orientation (shard b of every block in bank b)"
    )
    return out


def choose_blocks(device, x, w, full_sync=False, **forced):
    """choose for a block list (w = [(w | (gate, up), col0_tiles), ...]): the plan of its one call."""
    entries = block_entries(w)
    return choose(device, x, entries[0][0], full_sync, blocks=tuple(g for g, _, _ in entries), **forced)


# --- the op -----------------------------------------------------------------------------------------------------------


def matmul(
    x,
    w,
    scale,
    *,
    memory_config=None,
    fidelity="hifi2",
    full_sync=False,
    topk=None,
    out=None,
    col0=0,
    w_up=None,
    max_banks=None,
    **opts,
):
    """y = x @ w (* scale) on the device: x [1, 1, M, K] bf16 TILE interleaved (DRAM or L1; L1 keeps the per-core x
    reads off DRAM), w a [1, 1, K, N] bfloat8_b, bfloat4_b or bfloat16 TILE weight (native_weight; DRAM width-sharded,
    or interleaved), scale [1, 1, 1, N] bf16 TILE (scale_tensor) or None (unit scales: no scale stage). Returns [1, 1,
    M, N] bf16 TILE in memory_config (DRAM interleaved by default). max_banks: refuse a weight sharded over more DRAM
    banks (default common.device.MAX_BANKS; None: no limit).

    topk (one row slice, M <= 32): {"r": result tensor (topk_result), "k": K, "rows": T, "slot0": first record,
    "base": index of output element 0, "n": valid output elements, "write_y": bool}: each core's writer keeps a running
    top-K of each of the first T (<= 16) rows over its columns and writes it as record slot0 + its place in the plan of
    r ([1, 1, records, 2 * T * K] uint32 ROW_MAJOR, DRAM interleaved; keys, then indices; topk_records decodes them); y
    is still written with write_y. out: a [1, 1, M, N] bf16 TILE tensor to write into instead of a new one (with topk
    and write_y False it is never written: a fixed placeholder saves an allocation per call). col0 (with out, no topk):
    out is wider, [1, 1, M, W], and output column tile c goes to its tile c + col0.

    The work split (plan, chosen by choose from CONFIGS unless opts force it): each DRAM bank's columns go to
    cores_per_bank cores; each core accumulates its columns over K in fp32. Options (opts):
    - cores_per_bank: cores per DRAM bank;
    - split: the writer RISC loads some of the weight chunks into a second CB (two request streams per core, one per
      NoC): every other one, or SPLIT_SHARE's pattern (split_arg);
    - rows: each weight chunk is a few full rows of the core's columns, K outermost, the partial sums added up by the
      packer (L1 accumulation);
    - xs: x streams with the weight chunks (the default at up to 32 rows);
    - kb, kw, sbw: the x block, weight chunk depth and subblock width in tiles;
    - full_sync: 8 dest tiles per subblock instead of 4;
    - glu (scale None): the weight is interleave_glu's [gate | up] and the output act(gate) * up, N / 2 wide;
      glu_act: 0 silu (SwiGLU), 1 gelu_tanh (GeGLU); w_up (with glu): the up weight apart, w then the gate's ([1, 1,
      K, N] each, one format and sharding), the output N wide;
    - residual ([1, 1, M, N] bf16 TILE, scale None, no GLU or top-k): y = bf16(x @ w + residual); residual_mode
      "epilogue" (added in the output stage) or "preload" (in dest before the first chunk); out may be the residual
      itself (in place);
    - out32: y in float32, the fp32 accumulator as it is; scale None, no GLU or top-k.

    A block list in place of w: matmul(x, [(w_or_(gate, up), col0_tiles), ...], None, out=None, **opts) is one
    call, one output, over two to four column blocks. Each entry is a [1, 1, K, N_i] weight DRAM-sharded over the banks
    on its own (or a (gate, up) pair of such, for the GLU: every entry a pair or none), and col0_tiles its block's first
    output column in whole tiles of 32. All blocks share K, dtype, bank count, shard grid and orientation; scale is
    None; x has up to 32 rows. out: a [1, 1, M, W] bf16 TILE tensor the blocks are written into, each at its
    col0_tiles; by default one as wide as the furthest block's end (columns no block covers are not written). Each core
    takes its slice of every block's shard in its bank and runs the blocks one after the other in one dispatch. The
    output equals one call per block (out=, col0=) bit for bit when each block's K chunk depth KW equals its own call's;
    a list whose KW differs from any block's own call's is refused before its launch with BlockListMismatch (a
    ValueError naming the KWs), and the caller then makes one call per block. Block output spans must not overlap."""
    device = x.device()
    # weights sharded over more banks than allowed are refused before anything else
    if not isinstance(w, (list, tuple)):
        check_banks(w, max_banks)
    elif is_block_list(w):
        for wt, _ in w:
            for u in wt if isinstance(wt, (list, tuple)) else (wt,):
                check_banks(u, max_banks)
    check_banks(w_up, max_banks)
    segs = ()
    if isinstance(w, (list, tuple)):
        assert is_block_list(w), "dense_reader: w is one weight or a block list [(w | (gate, up), col0_tiles), ...]"
        entries = block_entries(w)
        assert scale is None and topk is None and col0 == 0 and w_up is None, (
            "dense_reader: a block list takes no scale, topk, col0 or w_up (each block carries its own col0_tiles)"
        )
        assert "residual" not in opts and not opts.get("out32"), "dense_reader: a block list: no residual or out32"
        assert x.shape[-2] <= 32, "dense_reader: a block list is a decode call: up to 32 rows"
        w, w_up, col0 = entries[0]
        segs = tuple(entries[1:])
        if w_up is not None:
            opts = {**opts, "glu": True}
        opts = {**opts, "blocks": tuple(g for g, _, _ in entries)}
    res_mode = {"epilogue": 1, "preload": 2}[opts.pop("residual_mode", "epilogue")]
    residual = opts.pop("residual", None)
    assert not (opts.get("out32") and topk is not None), "dense_reader: out32 without top-k"
    assert not (opts.get("out32") and out is not None and out.dtype != ttnn.float32), out.dtype
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
    assert x.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, x.memory_config()
    assert native_tile(w), f"dense_reader: weights are bfloat8_b, bfloat4_b or bfloat16, not {w.dtype}"
    glu = opts.get("glu", False)
    act = opts.pop("glu_act", 0)  # the GLU's activation: 0 silu, 1 gelu_tanh
    assert act in (0, 1) and (act == 0 or glu), "dense_reader: glu_act (0 silu, 1 gelu_tanh) needs glu"
    if glu:
        assert scale is None, "dense_reader: glu takes weights without scales"
    if x.shape[-2] <= 32 and "xs" not in opts:
        opts = {**opts, "xs": True}  # decode: x streams with the weight chunks
    if scale is None:
        opts = {**opts, "noscale": True}
    else:
        assert scale.dtype == ttnn.bfloat16 and scale.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    if max_banks is not None:
        opts = {**opts, "max_banks": max_banks}
    if w_up is not None:
        assert glu, "dense_reader: w_up is the GLU's up weight (glu=True)"
        assert w_up.dtype == w.dtype and tuple(w_up.padded_shape) == tuple(w.padded_shape)
        assert str(w_up.memory_config()) == str(w.memory_config()), "dense_reader: gate and up differ"
        opts = {**opts, "glu2": True}
        N = w.shape[-1]
    else:
        N = w.shape[-1] // 2 if glu else w.shape[-1]
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if residual is not None:
        assert scale is None and not glu and topk is None and not opts.get("out32"), (
            "dense_reader: residual: unit-scale matmul only"
        )
        assert residual.dtype == ttnn.bfloat16 and residual.layout == ttnn.TILE_LAYOUT
        assert residual.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        assert tuple(residual.padded_shape) == (1, 1, -(-x.shape[-2] // TILE) * TILE, -(-N // TILE) * TILE), (
            residual.shape,
            N,
        )
        # out may be the residual itself (in place): each core reads its residual tiles into L1 before it writes any
        # output tile, and writes only the tiles it read, so the aliasing is safe within the op
        assert col0 == 0 and (out is None or out is residual), "dense_reader: residual with a fresh output or in place"
    ends = [(c0, -(-g.shape[-1] // TILE)) for g, _, c0 in ((w, w_up, col0), *segs)]  # (first tile, tiles)
    if segs:
        N = max(TILE * c0 + g.shape[-1] for g, _, c0 in ((w, w_up, col0), *segs))  # the furthest block's end
    y = out
    if y is None:
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, x.shape[-2], N]),
            ttnn.float32 if opts.get("out32") else ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            mc,
        )
    assert segs or col0 == 0 or (out is not None and topk is None), "dense_reader: col0 needs out, no top-k"
    for c0, nt_ in ends if segs else [(col0, -(-N // TILE))]:
        assert nt_ + c0 <= -(-y.shape[-1] // TILE), (nt_, c0, y.shape)
    if segs:
        # the blocks' output tiles [col0_tiles, col0_tiles + tiles) must not overlap: two cores would write one tile
        spans = sorted(ends)
        assert all(a0 + an <= b0 for (a0, an), (b0, _) in itertools.pairwise(spans)), (
            f"dense_reader: block output spans overlap (col0_tiles, tiles): {spans}"
        )
    for attempt in range(3):
        try:
            slices = choose(device, x, w, full_sync, **opts)
            if segs:
                _block_kw_check(device, x, w, [(w, w_up, col0), *segs], full_sync, opts, slices)
            assert topk is None or len(slices) == 1, "dense_reader: top-k needs one row slice"
            for mt0, p in slices:
                _launch(
                    p,
                    x,
                    w,
                    scale,
                    y,
                    mt0,
                    fidelity,
                    full_sync,
                    p["split"],
                    topk,
                    ycol0=col0,
                    w2=w_up,
                    act=act,
                    res=residual,
                    res_mode=res_mode,
                    segs=segs,
                )
            return y
        except RuntimeError as e:
            # an L1 tensor below the free span's top: lower the cap under it and plan again
            m = re.search(r"allocated at (\d+) and static circular buffer region ends at (\d+)", str(e))
            if m is None or attempt == 2:
                raise
            over = int(m.group(2)) - int(m.group(1))
            _CAP[id(device)] = l1_free(device) - over - 64 * 1024
    return y


def topk_records(r, rows, k):
    """The records of a top-k result tensor (matmul's topk) on the host: (values [rows, records * k] float32,
    indices [rows, records * k] int64); empty slots hold -inf and -1."""
    return records_host(ttnn.to_torch(r), rows, k)


def records_host(t, rows, k):
    """topk_records on a host copy of the result tensor (torch: int32/uint32 words, or the same bytes as bfloat16)."""
    if t.dtype == torch.bfloat16:
        t = t.contiguous().view(torch.int32)  # the bits, two bf16 per uint32 word
    t = t.to(torch.int64).reshape(-1, 2, rows, k) & 0xFFFFFFFF
    keys, idx = (
        t[:, 0].permute(1, 0, 2).reshape(rows, -1),
        t[:, 1].permute(1, 0, 2).reshape(rows, -1),
    )
    bits = torch.where(keys >= 0x8000, keys & 0x7FFF, ~keys & 0xFFFF)
    v = bits << 16
    vals = torch.where(v >= 1 << 31, v - (1 << 32), v).to(torch.int32).view(torch.float32)
    empty = keys == 0
    vals = torch.where(empty, torch.full_like(vals, float("-inf")), vals)
    idx = torch.where(empty | (idx == 0xFFFFFFFF), torch.full_like(idx, -1), idx)
    return vals, idx


def topk_result(device, records, rows, k):
    """A result tensor for records top-k records of rows x k, zeroed (every slot empty): [1, 1, records, 2 * rows * k]
    uint32, or with RECORDS_BF16 the same bytes as [1, 1, records, 4 * rows * k] bfloat16 (one record a row-major page
    either way)."""
    if RECORDS_BF16:
        z, dt, width = torch.bfloat16, ttnn.bfloat16, 4 * rows * k
    else:
        z, dt, width = torch.int32, ttnn.uint32, 2 * rows * k
    return ttnn.from_torch(
        torch.zeros(1, 1, records, width, dtype=z),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def glu_iter(x, mt, glu):
    """The GLU stage's SFPU iterations on the valid rows only (GLU_ROWS; one tile row of x, at most 16 rows): two
    iterations cover four rows of a face; 0: whole tiles."""
    rows = x.shape[-2]
    return 2 * -(-rows // 4) if glu and GLU_ROWS and mt == 1 and rows <= 16 else 0


def _key(p, x, w, scale, y, mt0, fidelity, full_sync, split, col0=0, w2=None, act=0, res=None, res_mode=1, entries=None):
    """The program cache's key. entries (a block list: [(gate, up, col0_tiles)]): its blocks' shapes, layouts, dtype and
    output columns, no address (a hit rewrites the weights' addresses in the runtime arguments)."""
    plan_items = tuple((k, v) for k, v in p.items() if k != "work")
    common = (
        tuple(x.padded_shape),
        str(x.memory_config()),
        str(y.memory_config()),
        tuple(y.padded_shape),
        mt0,
        x.shape[-2] if p.get("glu") else 0,  # GLU: the valid rows set its SFPU stage
        act,
        plan_items,
        fidelity,
        full_sync,
        split,
        GLU_ROWS,
    )
    if entries:
        return (("blocks", tuple((_shape_key(g), u is not None, c0) for g, u, c0 in entries)), *common)
    return (
        id(w),
        w.buffer_address(),
        w2.buffer_address() if w2 is not None else 0,
        tuple(w.padded_shape),
        str(w.memory_config()),
        scale.buffer_address() if scale is not None else 0,
        col0,
        (str(res.memory_config()), res_mode) if res is not None else "",
        *common,
    )


def _launch(
    p,
    x,
    w,
    scale,
    y,
    mt0,
    fidelity,
    full_sync,
    split,
    topk=None,
    ycol0=0,
    w2=None,
    act=0,
    res=None,
    res_mode=1,
    segs=(),
):
    """One generic_op of the plan p: tile rows mt0 .. mt0 + p["MT"] of x into the same rows of y. w2: the up weight with
    p["glu2"] (w the gate's). segs (a block list): its blocks after the first, [(gate, up or None, col0_tiles)], p's
    segments 1 .. (w, w2 and ycol0 are segment 0's)."""
    assert bool(p.get("glu2")) == (w2 is not None), "dense_reader: glu2 takes the up weight as w2"
    assert len(segs) + 1 == p["SEGS"], (len(segs), p["SEGS"])
    seg_t = [t for g, u, _ in segs for t in (g, u) if t is not None]  # (their tensors)
    sa = split_arg(split, SPLIT_SHARE)  # the chunks' pattern between the two loaders (kernels/split.h)
    rq = REQ_TILES
    tk = None
    if topk is not None:
        tk = (
            topk["k"],
            topk["rows"],
            bool(topk.get("write_y", False)),
            topk["r"].buffer_address(),
            topk["slot0"],
            topk["base"],
            topk["n"],
        )
    key = (
        (
            _key(
                p,
                x,
                w,
                scale,
                y,
                mt0,
                fidelity,
                full_sync,
                sa,
                ycol0,
                w2,
                act,
                res,
                res_mode,
                [(w, w2, ycol0), *segs] if segs else None,
            ),
            tk,
            rq,
        )
        if CACHE
        else None
    )
    ios = [x, w, scale, y] if scale is not None else [x, w, y]
    if w2 is not None:
        ios.insert(2, w2)
    ios[2 + (w2 is not None) : 2 + (w2 is not None)] = seg_t  # (a block list's other weights, before y)
    if res is not None:
        ios.insert(len(ios) - 1, res)  # before y: the output stays last
    if topk is not None:
        ios.append(topk["r"])
    hit = _PROGRAMS.get(key) if CACHE else None
    if hit is not None:
        program = hit
        xa, ya = x.buffer_address(), y.buffer_address()
        rd, wr = program.kernels[0].runtime_args, program.kernels[1].runtime_args
        if segs:
            # (a block list is keyed without addresses: every block's own, as laid out below)
            wa, ua = w.buffer_address(), w2.buffer_address() if w2 is not None else 0
            sa_ = [(g.buffer_address(), u.buffer_address() if u is not None else 0) for g, u, _ in segs]
        for c, _ in p["work"]:
            rd[c.x][c.y][0] = xa
            wr[c.x][c.y][0] = ya
            if res is not None:
                wr[c.x][c.y][11] = res.buffer_address()
            if segs:
                r_, w_ = rd[c.x][c.y], wr[c.x][c.y]
                r_[2], r_[10], w_[4], w_[10] = wa, ua, wa, ua
                for i, (ga, gu) in enumerate(sa_):
                    r_[11 + 5 * i], r_[11 + 5 * i + 4] = ga, gu
                    w_[12 + 6 * i + 2], w_[12 + 6 * i + 5] = ga, gu
        ttnn.generic_op(ios, program)
        return
    MT, KT, KB, KW, SBW, NC, WP = (p[k] for k in ("MT", "KT", "KB", "KW", "SBW", "NC", "W_PAGES"))

    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in p["work"]])

    def cb(index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    tb = p["native"]
    x_pages = MT * KW * (2 if split else 1) * XS_DEPTH if p["xs"] else MT * KB
    ns = int(p["noscale"])
    cbs = [
        cb(0, ttnn.bfloat16, 2048, x_pages),
        cb(1, w.dtype, tb, WP),
        cb(3, ttnn.float32, 4096, MT * NC * p["SEGS"]),  # (every segment's accumulator)
        # out: bf16, or float32 (out32: the accumulator's fp32 packed as it is)
        cb(16, ttnn.float32, 4096, 2 * SBW) if p["out32"] else cb(16, ttnn.bfloat16, 2048, 2 * SBW),
    ]
    if not ns:
        cbs.append(cb(2, ttnn.bfloat16, 2048, NC))
    if split:
        cbs.append(cb(4, w.dtype, tb, WP))
    if res is not None:
        cbs.append(cb(8, ttnn.bfloat16, 2048, MT * NC))

    sh = int(p["sharded"])
    rows = int(p["rows"])
    xs = int(p["xs"])
    glu2 = NC // 2 if p["glu2"] else 0
    reader_ct = [MT, KT, KB, KW, SBW, NC, WP, sh, sa, rows, tb, xs, ns, glu2, rq]
    for t in (x, x if ns else scale, w):  # (x's accessor in the unused scale slot)
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_ct.append(p["SEGS"])
    glu = int(p["glu"])
    writer_ct = [MT, NC, SBW, KT, KB, KW, WP, sh, sa, rows, tb]
    writer_ct += [tk[0], tk[1], int(tk[2])] if tk else [0, 1, 1]
    writer_ct += [glu, glu2, int(res is not None), rq]
    for t in (y, w, topk["r"] if tk else y, res if res is not None else y):  # (y's accessor in the unused slots)
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct.append(p["SEGS"])
    if tk:
        cbs.append(cb(7, ttnn.bfloat16, max(64, 8 * tk[0] * tk[1]), 1))
    gmode = 2 if p["glu2"] else glu
    compute_ct = [
        MT,
        KT,
        KB,
        KW,
        SBW,
        NC,
        sa,
        rows,
        xs,
        ns,
        gmode,
        glu_iter(x, MT, glu),
        act,
        res_mode if res is not None else 0,
        p["SEGS"],
    ]
    w2a = w2.buffer_address() if w2 is not None else 0
    rcol = (lambda c: c // 2) if p["glu2"] else (lambda c: c)  # glu2: the weights' own columns

    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    vcs = read_vcs(p["work"])
    for place_, ((c, (bank, pitch, col0, nvalid, n0)), vc) in enumerate(zip(p["work"], vcs)):
        ra = [
            x.buffer_address(),
            scale.buffer_address() if not ns else 0,
            w.buffer_address(),
            bank,
            pitch,
            rcol(col0),
            nvalid,
            n0,
            mt0,
            vc,
            w2a,
        ]
        # a block list: per block after the first, its weight, pitch, columns of the core and the up weight
        sw_ = p["seg_work"][place_] if segs else ()
        for (g, u, _), (pitch_, col0_, nvalid_, _) in zip(segs, sw_, strict=True):
            ra += [g.buffer_address(), pitch_, rcol(col0_), nvalid_, u.buffer_address() if u is not None else 0]
        rd[c.x][c.y] = ra
        # y's own width in tiles (NT, NT / 2 with glu, or a wider out), and the core's first output tile shifted by
        # col0 (twice that with glu, whose writer halves it)
        nt_y = -(-y.padded_shape[-1] // TILE)
        wargs = [
            y.buffer_address(),
            nt_y,
            n0 + (2 if glu else 1) * ycol0,
            nvalid,
            w.buffer_address(),
            bank,
            pitch,
            rcol(col0),
            mt0,
            vc,
            w2a,
            res.buffer_address() if res is not None else 0,
        ]
        if tk:
            wargs += [tk[3], tk[4] + place_, tk[5], tk[6]]
        for (g, u, c0_), (pitch_, col0_, nvalid_, n0_) in zip(segs, sw_, strict=True):
            wargs += [
                n0_ + (2 if glu else 1) * c0_,
                nvalid_,
                g.buffer_address(),
                pitch_,
                rcol(col0_),
                u.buffer_address() if u is not None else 0,
            ]
        wr[c.x][c.y] = wargs

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    ccfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=FIDELITY[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=full_sync,
    )
    if p["out32"] or res is not None:
        # the accumulator (c_3) to dest by copy_tile in fp32: through srcA it would keep tf32's 10 mantissa bits, which
        # a bf16 output hides and a float32 one (or an fp32 residual sum before its one bf16 rounding) does not
        unpack = [ttnn.UnpackToDestMode.Default] * 64  # one per CB index
        unpack[3] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = unpack  # a field, not a constructor argument
    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("dense_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("dense_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel("dense_compute.cpp", compute_ct, ttnn.RuntimeArgs(), ccfg),
        ],
        semaphores=[],
        cbs=cbs,
    )
    if CACHE:
        _PROGRAMS[key] = program
    ttnn.generic_op(ios, program)


def linear(x, w, scale, memory_config=None, activation=None, out_dtype=None, **opts):
    """matmul for x of any rank: x [..., rows, K] (bf16 TILE; a sharded x is made interleaved in L1 first, one
    op), the [1, 1, K, N] weight and its [1, 1, 1, N] scales or None, the output in memory_config (DRAM by default),
    the input's rank back. activation: None, "silu" (ttnn.silu after the matmul), "swiglu" or "geglu" (w is
    interleave_glu's [gate | up] weight with scale None, the output act(gate) * up N / 2 wide, made in the matmul's
    output stage; or w is (gate, up), two weights [1, 1, K, N] apart, the output N wide). out_dtype ttnn.float32: the
    output in float32 (out32). out= and col0= write into a wider preallocated output at a column offset; it is returned
    as it is. A block list for w ([(w_or_(gate, up), col0_tiles), ...]): matmul's one call over column blocks,
    scale None; plain weights with activation None, (gate, up) pairs with "swiglu" or "geglu". opts go to
    matmul."""
    shape = list(x.shape)
    if out_dtype is not None and out_dtype != ttnn.bfloat16:
        assert out_dtype == ttnn.float32, f"dense_reader.linear: out_dtype {out_dtype}"
        opts = {**opts, "out32": True}
    blocks = is_block_list(w)
    if blocks:
        pairs = isinstance(w[0][0], (tuple, list))
        assert activation != "silu", "dense_reader.linear: no silu after a block list"
        takes = ("swiglu", "geglu") if pairs else (None,)
        assert activation in takes, (
            f"dense_reader.linear: a block list of {'(gate, up) pairs' if pairs else 'plain weights'} takes activation"
            f" {' or '.join(map(repr, takes))}, not {activation!r}"
        )
        if activation == "geglu":
            opts = {**opts, "glu_act": 1}
        n = max(TILE * c0 + (wt[0] if pairs else wt).shape[-1] for wt, c0 in w)  # the default output's width
    elif isinstance(w, (tuple, list)):
        assert activation in ("swiglu", "geglu"), "dense_reader.linear: two weights are the GLU's gate and up"
        w, w_up = w
        opts = {**opts, "w_up": w_up}
    rows, kd = shape[-2], shape[-1]
    if not blocks:
        n = w.shape[-1]
    if activation in ("swiglu", "geglu") and not blocks:
        opts = {**opts, "glu": True}
        if activation == "geglu":
            opts["glu_act"] = 1
        if opts.get("w_up") is None:
            n //= 2
    x4 = x if len(shape) == 4 else ttnn.reshape(x, [1, 1, rows, kd])
    tmp = None
    if x4.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        tmp = x4 = ttnn.sharded_to_interleaved(x4, ttnn.L1_MEMORY_CONFIG)
    y = matmul(x4, w, scale, memory_config=memory_config, **opts)
    if tmp is not None:
        ttnn.deallocate(tmp)
    if activation is not None and activation not in ("swiglu", "geglu"):
        assert activation == "silu", f"dense_reader.linear: activation {activation!r}"
        y = ttnn.silu(y)
    if len(shape) != 4 and opts.get("out") is None:  # (a given out is returned as it is)
        y = ttnn.reshape(y, [*shape[:-1], n])
    return y
