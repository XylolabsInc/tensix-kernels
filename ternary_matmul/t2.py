# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""t2.py: a ternary-weight matmul with group scales along K (trits -1, 0, +1 and one scale per output channel and G
inputs), launched through ttnn.generic_op: one token (t2_linear), many rows (t2m_linear), and linear() choosing by
rows.

The weight: the trits as hardware block-float tiles whose shared exponents read every value exactly, made on the host
(tiles):
- "bfp2": BFP2_b, 320 bytes per 32 x 32 tile: 64 exponent bytes of 127, then the 1024 values in the tile's face
  order, 2 bits each (bit 1 the sign, bit 0 the magnitude; 16 per 32-bit word, the first in the low bits). 2.5 bits
  per weight read from DRAM.
- "bfp2c": the same tiles with only their 256 code bytes in DRAM (2 bits per weight); the kernel keeps the exponent
  bytes in L1 and copies the codes behind them (kernels/t2_wload.h).
- "bfp4": BFP4_b, 576 bytes: exponents of 129, a sign and 3-bit magnitude per value. 4.5 bits per weight.
ttnn has no BFP2 dtype, so the weight is a uint8 ROW_MAJOR tensor, DRAM width-sharded over the banks; in each bank
every core's tiles are contiguous, K row after K row (ternary_weight), and the kernel's weight CBs are declared
bfloat4_b with 320-byte pages, the compute switching srcA's unpacker to BFP2_b (kernels/t2_compute.cpp). The scales:
[K/G, N] bf16 TILE in DRAM (group_scales).

One token: the token is spread over the activation tile's rows, each group's partial sums in their own row of D, D
times the scale tensor's own tile once per 32 groups, then the column sum. The weight is exactly q * s (s in bf16),
the kernel's error its arithmetic alone. Many rows: each weight tile is dequantized in L1 by its group's scale row,
then an ordinary matmul with fp32 accumulation (kernels/t2m_*.cpp).

Some ternary checkpoints store their weights in a rotated basis: the input x of every ternary projection must first
become x' = H_1024 (s * x), blockwise over 1024 inputs (the normalized Sylvester Hadamard, fixed +-1 signs s per input
width). With hadamard=hadamard_signs(device, s) the one-token kernel does that itself: each core transforms x once (per
block H_32 X H_32 with X the block as a 32 x 32 tile, H_1024 = H_32 (x) H_32: two tile matmuls), unnormalized, so the
scales must carry the 1/32 (group_scales(..., factor=HADAMARD_SCALE)). hadamard() does the same transform as its own
op, rotate_rows for many rows in ttnn ops.

Entry points: ternary_weight(device, trits), group_scales(device, scales), hadamard_signs(device, signs),
t2_linear(x, w, scale), t2m_linear(x, w, scale), linear(x, w, scale), hadamard(x, mask), rotate_rows(x, mask),
reference(x, trits, scales, g), rotate(x, signs).
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device as dev  # noqa: E402
from common import source as src  # noqa: E402

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32
# tile bytes, the exponent that reads a code's magnitude m as m, bits per value, tile bytes in DRAM ("bfp2c": the codes
# alone, the kernel sets the exponent bytes in L1)
FORMATS = {"bfp2": (320, 127, 2, 320), "bfp2c": (320, 127, 2, 256), "bfp4": (576, 129, 4, 576)}
# Bytes of L1 a core's circular buffers may take: None, the default: the device's free L1 span per bank at the plan,
# less L1_MARGIN; a number caps it.
L1_BUDGET = None
L1_MARGIN = 32 * 1024
DEPTH = 3  # weight chunks per loader in L1
# the default chunk: the most K rows (at most 16) whose tiles stay within this many; wider chunks pack D less often
# (once per KW K rows), the L1 budget permitting (plan falls back)
CHUNK_TILES = 256
# x's rows placed by NoC reads straight into the x ring when x is in L1 (or comes from the fused Hadamard), and the
# ring a whole pass when it fits, so its rows never need clearing
XDMA = True
RING_FIXED = True
HADAMARD = 1024  # the rotation's block
HADAMARD_SCALE = 1 / 32  # the kernel's H_32 X H_32 is 32 times the normalized transform


def source(name):
    return src.load(KERNELS, name)


def l1_budget(device):
    """The circular buffers' budget per core: L1_BUDGET, or the device's free L1 span less L1_MARGIN."""
    if L1_BUDGET is not None:
        return L1_BUDGET
    free = dev.l1_free(device)
    assert free is not None, "t2: the allocator does not report free L1; set t2.L1_BUDGET"
    return free - L1_MARGIN


def tiles(q, fmt="bfp2"):
    """trits q [K, N] (-1, 0, 1) -> uint8 [KT, NT, TB]: per tile its exponent bytes, then the codes in face order."""
    tb, exp, bits, _ = FORMATS[fmt]
    k, n = q.shape
    assert k % TILE == 0 and n % TILE == 0, q.shape
    assert q.abs().max() <= 1, "t2: the weight's values must be -1, 0 or +1"
    v = q.to(torch.int16).reshape(k // 32, 32, n // 32, 32).permute(0, 2, 1, 3)
    v = v.reshape(k // 32, n // 32, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5)
    v = v.reshape(k // 32, n // 32, 1024)
    if bits == 2:
        c = ((v < 0).to(torch.int16) * 2 + (v != 0).to(torch.int16)).to(torch.uint8)
        body = c[..., 0::4] | (c[..., 1::4] << 2) | (c[..., 2::4] << 4) | (c[..., 3::4] << 6)
    else:
        c = ((v < 0).to(torch.int16) * 8 + v.abs()).to(torch.uint8)
        body = c[..., 0::2] | (c[..., 1::2] << 4)
    head = torch.full((k // 32, n // 32, 64), exp, dtype=torch.uint8)
    out = torch.cat([head, body], dim=-1)
    assert out.shape[-1] == tb
    return out


def _layout(device, n, cores_per_bank, sbw, banks):
    """The column split: tiles per bank (ns), cores per bank (cpb), tiles per core (per), the subblock width (SBW) and
    the padded core width (NC)."""
    g = device.compute_with_storage_grid_size()
    cpb = max(1, min(cores_per_bank, (g.x * g.y) // banks))
    nt = -(-n // TILE)
    ns = -(-nt // banks)
    per = -(-ns // cpb)
    if sbw is None:
        # dest holds 4 fp32 tiles (half sync); unpack work per column: the weight tile plus 1/SBW of an x tile; padded
        # columns cost compute only (they are not stored)
        sbw = min(range(1, 5), key=lambda w_: (-(-per // w_) * w_ * (1 + 1 / w_), -w_))
    return cpb, nt, ns, per, sbw, -(-per // sbw) * sbw


def ternary_weight(device, trits, *, fmt="bfp2", cores_per_bank=3, sbw=None, banks=None, max_banks=None):
    """trits [N_out, K] (the [out, in] orientation of a linear layer's weight; -1, 0, 1) as the kernel's weight: a dict
    with the device tensor t (uint8 ROW_MAJOR [KT, banks * W], DRAM width-sharded over banks, default the device's, at
    most max_banks) and the work split. In bank b core j's columns (nvalid tiles, the next per of the bank's ns) are KT
    K rows of nvalid tiles, contiguous from byte off_j; no padding is stored, so every byte read is a weight."""
    tb, db = FORMATS[fmt][0], FORMATS[fmt][3]
    q = trits.T.contiguous()
    k, n = q.shape
    assert k % TILE == 0, f"t2: K = {k} must be a multiple of 32"
    banks = banks or dev.dram_banks(device, max_banks)
    cpb, nt, ns, per, sbw, nc = _layout(device, n, cores_per_bank, sbw, banks)
    if n != nt * TILE:
        q = torch.cat([q, torch.zeros(k, nt * TILE - n, dtype=q.dtype)], dim=1)
    t = tiles(q, fmt)[..., tb - db :]  # [KT, nt, DB]
    kt = k // TILE
    work, flat = [], []
    for b in range(banks):
        parts, off = [], 0
        for j in range(cpb):
            c0 = b * ns + j * per
            nvalid = max(0, min(per, ns - j * per, nt - c0))
            if nvalid == 0:
                continue
            parts.append(t[:, c0 : c0 + nvalid].reshape(-1))
            work.append((b, off, nvalid, c0))
            off += kt * nvalid * db
        flat.append(torch.cat(parts) if parts else torch.zeros(0, dtype=torch.uint8))
    size = max(f.numel() for f in flat)
    size = -(-size // (kt * 64)) * kt * 64  # whole rows of a 64-byte multiple
    w_row = size // kt
    data = torch.stack([torch.cat([f, torch.zeros(size - f.numel(), dtype=torch.uint8)]) for f in flat])
    data = data.reshape(banks, kt, w_row).permute(1, 0, 2).reshape(kt, banks * w_row)
    spec = ttnn.ShardSpec(dev.bank_grid(banks), (kt, w_row), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    dt = ttnn.from_torch(data.contiguous(), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    return {
        "t": dt,
        "k": k,
        "n": n,
        "fmt": fmt,
        "tb": tb,
        "db": db,
        "banks": banks,
        "cpb": cpb,
        "NT": nt,
        "SBW": sbw,
        "NC": nc,
        "work": work,
        "bytes": banks * size,
    }


def group_scales(device, scales, factor=1.0):
    """scales [N_out, K/G] (any float dtype) times factor (a power of two: HADAMARD_SCALE with the Hadamard) as [K/G, N]
    bf16 TILE in DRAM; the kernel's weight is q * bf16(s * factor)."""
    return ttnn.from_torch(
        (scales.float().T * factor).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


_SIGNS = {}  # sign mask's address -> its host signs (rotate_rows)
_ROT = {}  # (device, mask address) -> (the signs as a bf16 row, H_32)


def hadamard_signs(device, signs):
    """The +-1 signs [K] of the block Hadamard as the reader's sign masks: per K tile 16 words, word q holding bit 15
    for value 2q and bit 31 for value 2q + 1 when negative (an XOR flips the bf16 pair's signs), as uint8 ROW_MAJOR
    [1, KT x 64] in DRAM: one page, read once per call."""
    k = signs.numel()
    assert k % HADAMARD == 0, f"t2: the Hadamard needs K a multiple of {HADAMARD}, got {k}"
    neg = (signs.reshape(k // 2, 2) < 0).to(torch.int64)
    words = neg[:, 0] << 15 | neg[:, 1] << 31
    b = torch.stack([(words >> (8 * i)) & 255 for i in range(4)], dim=-1).to(torch.uint8)
    mask = ttnn.from_torch(
        b.reshape(1, k // TILE * 64).contiguous(),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _SIGNS[mask.buffer_address()] = signs.float().clone()
    return mask


def sylvester(n):
    """The +-1 Sylvester Hadamard matrix of order n (a power of two), float64."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def rotate(x, signs, block=HADAMARD):
    """x' = H_block (signs * x) / sqrt(block) per block of the last dimension, in float64 (the reference)."""
    h = sylvester(block)
    v = (x.double() * signs.double()).reshape(*x.shape[:-1], -1, block)
    return (v @ h.T / block**0.5).reshape(x.shape)


def rotate_rows(x, hadamard):
    """x' = 32 H_1024 (s * x) per block for many rows, in ttnn ops: the rows times the signs, the blocks as 32 x 32 tiles
    X, then H_32 X H_32 as two batched matmuls (the left product through transposes, Y^T = Z^T H_32 for Z = X H_32).
    Unnormalized, as the one-token kernel's (the scales carry the 1/32). x [1, 1, M, K] bf16 TILE; hadamard: the
    width's sign mask (hadamard_signs)."""
    device = x.device()
    m, k = x.shape[-2], x.shape[-1]
    key = (id(device), hadamard.buffer_address())
    if key not in _ROT:
        signs = _SIGNS[hadamard.buffer_address()]
        row, h32 = signs.reshape(1, 1, 1, k), sylvester(32).reshape(1, 1, 32, 32)
        _ROT[key] = tuple(
            ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in (row, h32)
        )
    row, h32 = _ROT[key]
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    xs = ttnn.multiply(x, row)
    xb = ttnn.reshape(xs, [1, m * (k // HADAMARD), 32, 32])
    z = ttnn.matmul(xb, h32, compute_kernel_config=cfg)
    zt = ttnn.transpose(z, -2, -1)
    yt = ttnn.matmul(zt, h32, compute_kernel_config=cfg)
    y = ttnn.transpose(yt, -2, -1)
    for t in (z, zt, yt):  # (xs, xb and y may share buffers with the reshapes: left to Python)
        ttnn.deallocate(t)
    return ttnn.reshape(y, [1, 1, m, k])


_HAD = {}


def hadamard(x, mask, memory_config=None):
    """x' = 32 H_1024 (s * x) per block for one token, on the device, as its own op: one block of 1024 inputs per core
    (kernels/had_*.cpp). x [1, 1, 1, K] bf16 TILE interleaved; mask: the width's sign masks (hadamard_signs). Returns x'
    [1, 1, 1, K] bf16 TILE (rows 1-31 zero) in memory_config (L1 by default), unnormalized like the fused path, so the
    scales carry the 1/32 and t2_linear takes it without hadamard=. One transform serves every projection of that
    input."""
    device = x.device()
    k = x.shape[-1]
    assert x.shape[-2] == 1 and k % HADAMARD == 0, x.shape
    nb, kt = k // HADAMARD, k // TILE
    mc = memory_config or ttnn.L1_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    key = (id(device), k, mask.buffer_address(), str(x.memory_config()), str(mc))
    hit = _HAD.get(key)
    g = device.compute_with_storage_grid_size()
    assert nb <= g.x * g.y, f"t2: {nb} Hadamard blocks, {g.x * g.y} cores"
    cores = [ttnn.CoreCoord(b % g.x, b // g.x) for b in range(nb)]
    if hit is None:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [
            cb(8, bf, 2048, 1),
            cb(9, bf, 32 * 576 + 64, 1),
            cb(10, f32, 4096, 1),
            cb(11, bf, 2048, 1),
            cb(12, bf, 2048, 1),
            cb(16, bf, 2048, 32),
            cb(17, bf, 2048, 1),
        ]
        reader_ct = [kt]
        for t in (x, mask):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for b, c in enumerate(cores):
            rd[c.x][c.y] = [x.buffer_address(), mask.buffer_address(), b]
            wr[c.x][c.y] = [y.buffer_address(), b]

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=source(name),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        hit = _HAD[key] = ttnn.ProgramDescriptor(
            kernels=[
                kernel("had_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
                kernel("had_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
                kernel(
                    "had_compute.cpp",
                    [],
                    ttnn.RuntimeArgs(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
    else:
        rd, wr = hit.kernels[0].runtime_args, hit.kernels[1].runtime_args
        for c in cores:
            rd[c.x][c.y][0] = x.buffer_address()
            wr[c.x][c.y][0] = y.buffer_address()
    ttnn.generic_op([x, mask, y], hit)
    return y


def dequantize(trits, scales, g):
    """The float [N_out, K] weight q * s (s as given; s.to(torch.bfloat16) for the kernel's)."""
    return trits.float() * scales.float().repeat_interleave(g, dim=1)


def reference(x, trits, scales, g):
    """x @ (q * s).T in float32, x as the device sees it (bf16), s as the device holds it (bf16)."""
    return x.to(torch.bfloat16).float() @ dequantize(trits, scales.to(torch.bfloat16), g).T


def plan(device, w, g, kw=None, split=True, hadamard=False, budget=None):
    """{KT, GT, KW, SBW, NC, W_PAGES, XPAGES, work: [(core, (bank, off, nvalid, n0))], l1} of the one-token kernel."""
    k, tb, db, sbw, nc = w["k"], w["tb"], w["db"], w["SBW"], w["NC"]
    assert g % TILE == 0 and k % g == 0, f"t2: K = {k} must be a multiple of G = {g}"
    budget = budget or l1_budget(device)
    kt, gt = k // TILE, g // TILE
    pt = 32 * gt
    cores = dev.place(device, [wk[0] for wk in w["work"]], True)
    cands = [d for d in range(gt, pt + 1, gt) if pt % d == 0 and kt % d == 0]
    if kw is None:
        kw = max([d for d in cands if d * nc <= CHUNK_TILES and d <= 16] or [gt])
    assert kw in cands, f"t2: KW {kw} must be a multiple of {gt} dividing {pt} and {kt}"
    loaders = 2 if split else 1

    nb = k // HADAMARD if hadamard else 0

    def size(kw_, depth, xp):
        return (
            (nb * 2048 + 2 * 2048 + 4096 + 2048 + kt * 64 if nb else 0)  # Y, X_b, Z, H_32, masks
            + xp * 2048  # the x ring
            + loaders * depth * kw_ * nc * tb
            + (loaders * kw_ * nc * db if db < tb else 0)  # codes only: the staging buffers
            + 2 * nc * 2048  # scales: this pass's and the next
            + 2 * nc * 4096  # D and the output accumulator
            + sbw * 4096  # D * S
            + 2048  # ones
            + max(32, kw_) * 576
            + 64  # x staging (a block's 32 K tiles, or a chunk's)
            + 2 * sbw * 2048  # out
        )

    # the widest chunk that fits, three in flight per loader, else two; first with a ring of a whole pass (fixed rows),
    # then of three chunks
    rings = ((lambda k_: pt), (lambda k_: 3 * k_)) if RING_FIXED else ((lambda k_: 3 * k_),)
    fits = [(d, k_, r(k_)) for r in rings for d in (DEPTH, 2) for k_ in sorted(cands, reverse=True) if k_ <= kw]
    depth, kw, xpages = next(((d, k_, xp) for d, k_, xp in fits if size(k_, d, xp) <= budget), (2, gt, 3 * gt))
    l1 = size(kw, depth, xpages)
    assert l1 <= budget, f"t2: {l1} B of circular buffers per core (budget {budget})"
    return {
        "KT": kt,
        "GT": gt,
        "KW": kw,
        "SBW": sbw,
        "NC": nc,
        "W_PAGES": depth * kw * nc,
        "XPAGES": xpages,
        "split": split,
        "NB": nb,
        "work": list(zip(cores, w["work"])),
        "l1": l1,
    }


_PROGRAMS = {}


def t2_linear(
    x,
    w,
    scale,
    g=128,
    *,
    hadamard=None,
    memory_config=None,
    fidelity="hifi2",
    kw=None,
    split=True,
    topk=None,
    out=None,
    col0=0,
    gap=None,
):
    """y = x @ (q * s).T for one token: x [1, 1, 1, K] bf16 TILE interleaved (L1 or DRAM), w from ternary_weight,
    scale from group_scales (G inputs per scale); with hadamard (hadamard_signs) x is first rotated (see the module
    docstring). Returns [1, 1, 1, N] bf16 TILE (rows 1-31 of the tile zero) in memory_config (DRAM by default).
    split: the writer RISC loads every other weight chunk (a second request stream). topk (an LM head; the dense
    reader's contract): {"r": result tensor (dense_reader.linear.topk_result made for 1 row), "k", "slot0", "base", "n",
    "write_y"}: each core's writer keeps the row's top k over its columns and writes one record, slot0 + its place; y
    (out, a placeholder of the right shape, when write_y is False) is then not written. out (one row): write into this
    preallocated [1, 1, 1, W] bf16 TILE tensor instead, output column tile c at tile c + col0, and with gap = (split,
    shift) tiles c >= split at c + col0 + shift (the output's columns in two runs). Returns out."""
    device = x.device()
    assert x.shape[-2] == 1, f"t2: one token, got {x.shape[-2]} rows"
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
    assert x.shape[-1] == w["k"], (x.shape, w["k"])
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    y = out
    if y is None:
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, w["n"]]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    if out is not None and (col0 or gap is not None):
        last = -(-w["n"] // TILE) - 1
        last += col0 + (gap[1] if gap is not None and last >= gap[0] else 0)
        assert last < -(-out.shape[-1] // TILE), f"t2: out has {out.shape[-1]} columns, tile {last}"
    tk = None
    if topk is not None:
        tk = (
            topk["k"],
            bool(topk.get("write_y", False)),
            topk["r"].buffer_address(),
            topk["slot0"],
            topk["base"],
            topk["n"],
        )
    # the weight's shape and split too: a freed weight's id and addresses come back for the next
    key = (
        id(w["t"]),
        w["t"].buffer_address(),
        tuple(w["t"].padded_shape),
        (w["k"], w["n"], w["fmt"], w["NC"], w["SBW"], tuple(w["work"])),
        scale.buffer_address(),
        tuple(scale.padded_shape),
        hadamard.buffer_address() if hadamard is not None else 0,
        str(x.memory_config()),
        str(mc),
        g,
        fidelity,
        kw,
        split,
        tk,
        str(y.memory_config()),
        col0,
        gap,
        XDMA,
        RING_FIXED,
        DEPTH,
        CHUNK_TILES,
    )
    hit = _PROGRAMS.get(key)
    if hit is None:
        program, work = _PROGRAMS[key] = _program(x, w, scale, y, g, fidelity, kw, split, hadamard, tk, topk, col0, gap)
    else:
        program, work = hit
        xa, ya = x.buffer_address(), y.buffer_address()
        rd, wr = program.kernels[0].runtime_args, program.kernels[1].runtime_args
        for c, _ in work:
            rd[c.x][c.y][0] = xa
            wr[c.x][c.y][0] = ya
    ios = [x, w["t"], scale, y] + ([hadamard] if hadamard is not None else [])
    ios += [topk["r"]] if topk is not None else []
    ttnn.generic_op(ios, program)
    return y


def _program(x, w, scale, y, g, fidelity, kw, split, hadamard, tk, topk, col0, gap):
    device = x.device()
    p = plan(device, w, g, kw=kw, split=split, hadamard=hadamard is not None)
    nb = p["NB"]
    KT, GT, KW, SBW, NC, WP, XP = (p[k] for k in ("KT", "GT", "KW", "SBW", "NC", "W_PAGES", "XPAGES"))
    tb, db = w["tb"], w["db"]
    exp = FORMATS[w["fmt"]][1]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in p["work"]])

    def cb(index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    bf, f32 = ttnn.bfloat16, ttnn.float32
    sp = int(split)
    cbs = [
        cb(0, bf, 2048, XP),
        cb(1, ttnn.bfloat4_b, tb, WP),
        cb(2, bf, 2048, 2 * NC),
        cb(3, f32, 4096, NC),
        cb(5, bf, 2048, 1),
        cb(6, f32, 4096, NC),
        cb(7, f32, 4096, SBW),
        cb(9, bf, max(32, KW) * 576 + 64, 1),
        cb(16, bf, 2048, 2 * SBW),
    ]
    if sp:
        cbs.append(cb(4, ttnn.bfloat4_b, tb, WP))
    if db < tb:
        cbs += [cb(14 + j, bf, KW * NC * db, 1) for j in range(1 + sp)]
    if nb:
        cbs += [cb(8, bf, 2048, 2), cb(10, f32, 4096, 1), cb(11, bf, 2048, nb), cb(12, bf, 2048, 1)]
        cbs.append(cb(17, bf, KT * 64, 1))
    nst = -(-w["n"] // TILE)
    # XDMA: x's rows straight into the ring (x in L1, or the Hadamard's own Y tiles)
    xdma = int(XDMA and (nb > 0 or x.memory_config().buffer_type == ttnn.BufferType.L1))
    reader_ct = [KT, KW, GT, NC, sp, XP, tb, nb, db, WP, exp, xdma]
    for t in (x, scale, hadamard if nb else x):  # (x's accessor in the unused mask slot)
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [KT, KW, NC, SBW, sp, tb, tk[0] if tk else 0, int(tk[1]) if tk else 1, db, WP, exp]
    for t in (y, topk["r"] if tk else y):  # (y's accessor in the unused result slot)
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    if tk:
        cbs.append(cb(13, bf, max(64, 8 * tk[0]), 1))
    compute_ct = [KT, KW, GT, SBW, NC, sp, int(w["fmt"].startswith("bfp2")), nb]
    vcs = dev.read_vcs([(c, (bnk,)) for c, (bnk, *_r) in p["work"]])
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    wa = w["t"].buffer_address()
    assert col0 >= 0 and (gap is None or gap[1] >= 0), (col0, gap)
    gsplit, gshift = gap if gap is not None else (1 << 30, 0)
    for place_, ((c, (bnk, off, nvalid, n0)), vc) in enumerate(zip(p["work"], vcs)):
        rd[c.x][c.y] = [
            x.buffer_address(),
            scale.buffer_address(),
            wa,
            bnk,
            off,
            nvalid,
            n0,
            nst,
            vc,
            hadamard.buffer_address() if nb else 0,
        ]
        wargs = [y.buffer_address(), n0, nvalid, wa, bnk, off, vc, col0, gsplit, gshift]
        if tk:
            wargs += [tk[2], tk[3] + place_, tk[4], tk[5]]
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

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("t2_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("t2_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "t2_compute.cpp",
                compute_ct,
                ttnn.RuntimeArgs(),
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=dev.FIDELITY[fidelity],
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    return program, p["work"]


MT_MAX = 4  # tile rows per multi-row call (each call reads the weights once)


def plan_m(device, w, g, mt, kw=None, split=True, budget=None):
    """The multi-row kernel's plan for mt tile rows, or None when its buffers do not fit L1: {MT, KT, GT, KW, SBW, NC,
    W_PAGES, work, l1}. SBW: the widest divisor of NC with MT x SBW output tiles in half of dest (4 fp32 tiles)."""
    k, tb, db, nc = w["k"], w["tb"], w["db"], w["NC"]
    assert g % TILE == 0 and k % g == 0, f"t2: K = {k} must be a multiple of G = {g}"
    budget = budget or l1_budget(device)
    kt, gt = k // TILE, g // TILE
    pt = 32 * gt
    widths = [d for d in range(1, 5) if nc % d == 0 and mt * d <= 4]
    if not widths:
        return None
    sbw = max(widths)
    cores = dev.place(device, [wk[0] for wk in w["work"]], True)
    cands = [d for d in range(gt, pt + 1, gt) if pt % d == 0 and kt % d == 0]
    if kw is None:
        kw = max([d for d in cands if d * nc <= CHUNK_TILES and d <= 16] or [gt])
    loaders = 2 if split else 1

    def size(kw_, depth):
        return (
            3 * mt * kw_ * 2048  # x
            + loaders * depth * kw_ * nc * tb
            + (loaders * kw_ * nc * db if db < tb else 0)
            + 2 * nc * 2048  # scales
            + 2 * kw_ * sbw * 2048  # the dequantized subblock
            + mt * nc * 4096  # the accumulator
            + 2 * sbw * 2048  # out
        )

    fits = [(d, k_) for d in (DEPTH, 2) for k_ in sorted(cands, reverse=True) if k_ <= kw]
    depth, kw = next(((d, k_) for d, k_ in fits if size(k_, d) <= budget), (2, gt))
    if size(kw, depth) > budget:
        return None
    return {
        "MT": mt,
        "KT": kt,
        "GT": gt,
        "KW": kw,
        "SBW": sbw,
        "NC": nc,
        "W_PAGES": depth * kw * nc,
        "split": split,
        "work": list(zip(cores, w["work"])),
        "l1": size(kw, depth),
    }


def t2m_linear(x, w, scale, g=128, *, memory_config=None, fidelity="hifi2", kw=None, split=True):
    """y = x @ (q * s).T for many rows: x [1, 1, M, K] bf16 TILE interleaved, already in the weights' basis
    (rotate_rows for a rotated checkpoint), w from ternary_weight, scale from group_scales. Row slices of up to MT_MAX
    tile rows (fewer when L1 is short), each one call reading the weights once. Returns [1, 1, M, N] bf16 TILE in
    memory_config (DRAM by default)."""
    device = x.device()
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
    assert x.shape[-1] == w["k"], (x.shape, w["k"])
    m = x.shape[-2]
    mts = -(-m // TILE)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, m, w["n"]]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    mt0 = 0
    while mt0 < mts:
        mt = min(MT_MAX, mts - mt0)
        while plan_m(device, w, g, mt, kw, split) is None:
            assert mt > 1, "t2m: one tile row does not fit L1"
            mt -= 1
        key = ("m", id(w["t"]), w["t"].buffer_address(), scale.buffer_address())
        key += (tuple(w["t"].padded_shape), w["k"], w["n"], w["fmt"], w["NC"], tuple(w["work"]))
        key += (str(x.memory_config()), str(mc), tuple(x.padded_shape), g, fidelity, kw, split)
        key += (mt, mt0, DEPTH, CHUNK_TILES)
        hit = _PROGRAMS.get(key)
        if hit is None:
            p = plan_m(device, w, g, mt, kw, split)
            hit = _PROGRAMS[key] = (_program_m(x, w, scale, y, p, mt0, fidelity), p["work"])
        else:
            rd, wr = hit[0].kernels[0].runtime_args, hit[0].kernels[1].runtime_args
            for c, _ in hit[1]:
                rd[c.x][c.y][0] = x.buffer_address()
                wr[c.x][c.y][0] = y.buffer_address()
        ttnn.generic_op([x, w["t"], scale, y], hit[0])
        mt0 += mt
    return y


def _program_m(x, w, scale, y, p, mt0, fidelity):
    MT, KT, GT, KW, SBW, NC, WP = (p[k] for k in ("MT", "KT", "GT", "KW", "SBW", "NC", "W_PAGES"))
    tb, db = w["tb"], w["db"]
    exp = FORMATS[w["fmt"]][1]
    sp = int(p["split"])
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in p["work"]])

    def cb(index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        cb(0, bf, 2048, 3 * MT * KW),
        cb(1, ttnn.bfloat4_b, tb, WP),
        cb(2, bf, 2048, 2 * NC),
        cb(3, f32, 4096, MT * NC),
        cb(6, bf, 2048, 2 * KW * SBW),
        cb(16, bf, 2048, 2 * SBW),
    ]
    if sp:
        cbs.append(cb(4, ttnn.bfloat4_b, tb, WP))
    if db < tb:
        cbs += [cb(14 + j, bf, KW * NC * db, 1) for j in range(1 + sp)]
    nst = -(-w["n"] // TILE)
    reader_ct = [MT, KT, KW, GT, NC, sp, tb, db, WP, exp]
    for t in (x, scale):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [MT, KT, KW, NC, SBW, sp, tb, db, WP, exp]
    writer_ct.extend(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    compute_ct = [MT, KT, KW, GT, SBW, NC, sp, int(w["fmt"].startswith("bfp2"))]
    vcs = dev.read_vcs([(c, (bnk,)) for c, (bnk, *_r) in p["work"]])
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    wa = w["t"].buffer_address()
    for (c, (bnk, off, nvalid, n0)), vc in zip(p["work"], vcs):
        rd[c.x][c.y] = [x.buffer_address(), scale.buffer_address(), wa, bnk, off, nvalid, n0, nst, vc, mt0]
        wr[c.x][c.y] = [y.buffer_address(), nst, n0, nvalid, wa, bnk, off, vc, mt0]

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            kernel("t2m_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("t2m_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "t2m_compute.cpp",
                compute_ct,
                ttnn.RuntimeArgs(),
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=dev.FIDELITY[fidelity],
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )


def linear(x, w, scale=None, g=None, memory_config=None, activation=None, rows_kernel=True, **opts):
    """t2_linear / t2m_linear for x of any rank with [..., rows, K] (bf16 TILE; a sharded x is made interleaved in L1
    first), w from ternary_weight (a dict may also carry "scale", "g" and "had", used unless given), the output in
    memory_config (DRAM by default), activation "silu" or None, the input's rank back. One row runs the one-token kernel
    (its Hadamard fused). More rows run rotate_rows (with hadamard=) and the multi-row kernel (t2m_linear), whose calls
    read the weights once per MT_MAX tile rows; rows_kernel=False instead runs the one-token kernel per row. With out=
    (and col0=, gap=: t2_linear's) one row is written into out at a column offset, which is returned."""
    scale = w["scale"] if scale is None else scale
    g = w.get("g", 128) if g is None else g
    opts.setdefault("hadamard", w.get("had"))
    shape = list(x.shape)
    rows, kd = shape[-2], shape[-1]
    if opts.get("out") is not None:
        assert rows == 1 and len(shape) == 4, f"t2.linear: out takes one row of a 4-D x, {shape}"
        assert activation is None, "t2.linear: out takes no activation"
    x4 = x if len(shape) == 4 else ttnn.reshape(x, [1, 1, rows, kd])
    tmp = None
    if x4.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        tmp = x4 = ttnn.sharded_to_interleaved(x4, ttnn.L1_MEMORY_CONFIG)
    if rows == 1:
        y = t2_linear(x4, w, scale, g, memory_config=memory_config, **opts)
    elif rows_kernel:
        xr = rotate_rows(x4, opts["hadamard"]) if opts.get("hadamard") is not None else x4
        mopts = {k_: v for k_, v in opts.items() if k_ in ("fidelity", "kw", "split")}
        y = t2m_linear(xr, w, scale, g, memory_config=memory_config, **mopts)
        if xr is not x4:
            ttnn.deallocate(xr)
    else:
        parts = []
        for r in range(rows):
            xr = ttnn.slice(x4, [0, 0, r, 0], [1, 1, r + 1, kd])
            parts.append(t2_linear(xr, w, scale, g, memory_config=ttnn.L1_MEMORY_CONFIG, **opts))
            ttnn.deallocate(xr)
        y = ttnn.concat(parts, dim=2, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
        for t in parts:
            ttnn.deallocate(t)
    if tmp is not None:
        ttnn.deallocate(tmp)
    if activation is not None:
        assert activation == "silu", f"t2.linear: activation {activation!r}"
        y = ttnn.silu(y)
    if len(shape) != 4 and opts.get("out") is None:
        y = ttnn.reshape(y, [*shape[:-1], w["n"]])
    return y
