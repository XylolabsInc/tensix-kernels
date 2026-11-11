# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""w4a16.py: an int4-weight matmul with group scales along K (G = 32) for one token, launched through
ttnn.generic_op.

The weight: symmetric int4 codes q in -7..7 (sign-magnitude, no -8) with one bf16 scale per output channel and 32
inputs, the weight being q * s exactly. int4_weight packs the codes into a uint8 ROW_MAJOR tensor DRAM width-sharded
over the banks, 512 bytes per 32 x 32 tile: one nibble per value (the sign and a 3-bit magnitude, the first element in
the low nibble), in the tile's face order. That is the mantissa section of a bfloat4_b tile (576 bytes: 64 exponent
bytes first), so the reader copies each 512-byte tile behind 64 bytes of the exponent 129 and the unpacker reads every
value as the integer q (value = m * 2^(E - 129)). The scales: [K/32, N] bf16 TILE in DRAM, one row of N per 32-row K
tile (group_scales).

The group scale varies along K, so it cannot be applied after the K sum, and dequantizing every weight tile would cost
several times the matmul's own work. Instead (kernels/w4a16_compute.cpp) the token is spread over the activation
tile's rows, so 32 K tiles accumulate in dest with row r holding exactly K tile 32p + r's partial product; that tile
times the scale tensor's own tile, then its column sum, once per 32 weight tiles.

Entry points: w4a16_linear(x, w, scale), int4_weight(device, codes), group_scales(device, scales), dequantize(w,
scale) (the weight as bfloat8_b for an ordinary matmul over many rows), reference(x, codes, scales).
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
Q4_TILE = 512
BFP4_TILE = 576
EXP = 129  # the bfloat4_b exponent that reads a 3-bit magnitude m as m (2^(129 - 129))
# Bytes of L1 a core's circular buffers may take: None, the default: the device's free L1 span per bank at the call,
# less L1_MARGIN; a number caps it.
L1_BUDGET = None
L1_MARGIN = 32 * 1024
# the dequantize packs bfloat8_b from fp32 dest with the precise rounding
PACK_PRECISE = True


def source(name):
    return src.load(KERNELS, name)


def l1_budget(device):
    """The circular buffers' budget per core: L1_BUDGET, or the device's free L1 span less L1_MARGIN."""
    if L1_BUDGET is not None:
        return L1_BUDGET
    free = dev.l1_free(device)
    assert free is not None, "w4a16: the allocator does not report free L1; set w4a16.L1_BUDGET"
    return free - L1_MARGIN


def nibbles(codes):
    """int codes [K, N] (-7..7) -> uint8 [KT, NT, 512]: per tile the 1024 values in face order, sign << 3 |
    magnitude, two per byte (element 2i in the low nibble)."""
    k, n = codes.shape
    q = codes.to(torch.int16).reshape(k // 32, 32, n // 32, 32).permute(0, 2, 1, 3)
    q = q.reshape(k // 32, n // 32, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5).reshape(k // 32, n // 32, 1024)
    v = ((q < 0).to(torch.int16) * 8 + q.abs()).to(torch.uint8)
    return v[..., 0::2] | (v[..., 1::2] << 4)


def int4_weight(device, codes, banks=None, max_banks=None):
    """codes [N_out, K] int8 in -7..7 (the [out, in] orientation of a linear layer's weight) as the kernel's weight:
    uint8 ROW_MAJOR [KT, NT_pad * 512], DRAM width-sharded over banks (default: the device's, at most max_banks), one
    block of tile columns per bank (N padded to banks * 32). Returns a dict: the tensor "t" and its layout."""
    assert int(codes.abs().max()) <= 7, "w4a16: codes are -7..7"
    q = codes.T.contiguous()  # [K, N]
    k, n = q.shape
    assert k % TILE == 0, f"w4a16: K = {k} must be a multiple of 32"
    banks = banks or dev.dram_banks(device, max_banks)
    npad = -(-n // (TILE * banks)) * TILE * banks
    if npad != n:
        q = torch.cat([q, torch.zeros(k, npad - n, dtype=q.dtype)], dim=1)
    b = nibbles(q).reshape(k // 32, -1)
    ns = npad // TILE // banks
    spec = ttnn.ShardSpec(dev.bank_grid(banks), (k // 32, ns * Q4_TILE), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    t = ttnn.from_torch(b, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    return {"t": t, "k": k, "n": n, "banks": banks, "ns": ns, "pitch": ns * Q4_TILE}


def group_scales(device, scales):
    """scales [N_out, K/32] (one per output channel and 32 inputs) as [K/32, N] bf16 TILE in DRAM."""
    return ttnn.from_torch(
        scales.T.contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def dequantized(codes, scales):
    """The float [N_out, K] weight q * s (s rounded to bf16, as the device holds it)."""
    return codes.float() * scales.to(torch.bfloat16).float().repeat_interleave(32, dim=1)


def reference(x, codes, scales):
    """x @ (q * s).T in float32 (x as the device sees it: bf16): codes [N_out, K], scales [N_out, K/32]."""
    return x.to(torch.bfloat16).float() @ dequantized(codes, scales).T


def plan(device, wq, cores_per_bank=3, kb=None, kw=None, sbw=None, full_sync=True, split=True, budget=None):
    """{KT, KB, KW, SBW, NC, W_PAGES, work: [(core, (bank, col0, nvalid, n0))], l1}."""
    k, n, banks, ns = wq["k"], wq["n"], wq["banks"], wq["ns"]
    KT, NT = k // TILE, -(-n // TILE)
    assert KT % 32 == 0, f"w4a16: K = {k} must be a multiple of 1024 (passes of 32 K tiles)"
    budget = budget or l1_budget(device)
    g = device.compute_with_storage_grid_size()
    cpb = max(1, min(cores_per_bank, (g.x * g.y) // banks))
    per = -(-ns // cpb)
    cap = 8 if full_sync else 4
    if sbw is None:
        sbw = min(range(1, cap + 1), key=lambda w_: (-(-per // w_) * w_ * (1 + 1 / w_), -w_))
    NC = -(-per // sbw) * sbw
    work = []
    for bnk in range(banks):
        valid = max(0, min(ns, NT - bnk * ns))
        for j in range(cpb):
            nvalid = max(0, min(per, valid - j * per))
            if nvalid:
                work.append((bnk, j * per, nvalid, bnk * ns + j * per))
    cores = dev.place(device, [wk[0] for wk in work], True)
    if kw is None:
        kw = max(d for d in (1, 2, 4, 8, 16, 32) if d * sbw <= 32)
    ch = kw * sbw
    w_pages = 3 * ch

    def l1(kb_):
        return (
            kb_ * 2048
            + w_pages * BFP4_TILE
            + 2 * sbw * 2048
            + NC * 4096
            + 2048
            + 2 * sbw * 4096
            + 2 * sbw * 2048
            + ch * Q4_TILE
            + 4096
            + (w_pages * BFP4_TILE + ch * Q4_TILE if split else 0)
        )

    if kb is None:
        cands = [d for d in range(32, KT + 1, 32) if KT % d == 0]
        kb = max([d for d in cands if l1(d) <= budget] or [32])
    assert KT % kb == 0 and kb % 32 == 0, (KT, kb)
    assert l1(kb) <= budget, f"w4a16: {l1(kb)} B of circular buffers (budget {budget})"
    return {
        "KT": KT,
        "NT": NT,
        "KB": kb,
        "KW": kw,
        "SBW": sbw,
        "NC": NC,
        "W_PAGES": w_pages,
        "full_sync": full_sync,
        "split": split,
        "work": list(zip(cores, work)),
        "l1": l1(kb),
    }


def w4a16_linear(x, wq, scale, *, memory_config=None, fidelity="hifi2", **opts):
    """y = x @ (q * s).T for one token: x [1, 1, 1, K] bf16 TILE interleaved (L1 or DRAM), wq from int4_weight, scale
    from group_scales. Returns [1, 1, 1, N] bf16 TILE (rows 1-31 of the tile zero) in memory_config (DRAM by default).
    opts go to plan: cores_per_bank, kb, kw, sbw, full_sync, split (the writer RISC loads every other weight chunk),
    budget."""
    device = x.device()
    assert x.shape[-2] == 1, f"w4a16: one token, got {x.shape[-2]} rows"
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
    assert x.shape[-1] == wq["k"], (x.shape, wq["k"])
    p = plan(device, wq, **opts)
    KT, NT, KB, KW, SBW, NC, WP = (p[k] for k in ("KT", "NT", "KB", "KW", "SBW", "NC", "W_PAGES"))
    n = wq["n"]
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _ in p["work"]])

    def cb(index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    bf, f32 = ttnn.bfloat16, ttnn.float32
    ch = KW * SBW
    sp = int(p["split"])
    stage_pages = -(-ch * Q4_TILE // 2048)
    cbs = [
        cb(0, bf, 2048, KB),
        cb(1, ttnn.bfloat4_b, BFP4_TILE, WP),
        cb(2, bf, 2048, 2 * SBW),
        cb(3, f32, 4096, NC),
        cb(5, bf, 2048, 1),
        cb(6, f32, 4096, SBW),
        cb(7, f32, 4096, SBW),
        cb(8, bf, 2048, stage_pages),
        cb(9, bf, 2048, 2),
        cb(16, bf, 2048, 2 * SBW),
    ]
    if sp:
        cbs += [cb(4, ttnn.bfloat4_b, BFP4_TILE, WP), cb(10, bf, 2048, stage_pages)]
    reader_ct = [KT, KB, KW, SBW, NC, WP, EXP, sp]
    for t in (x, scale):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [NC, SBW, KT, KB, KW, WP, EXP, sp, *ttnn.TensorAccessorArgs(y).get_compile_time_args()]
    compute_ct = [KT, KB, KW, SBW, NC, sp]
    vcs = dev.read_vcs([(c, (bnk,)) for c, (bnk, *_r) in p["work"]])
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    wa = wq["t"].buffer_address()
    for (c, (bnk, col0, nvalid, n0)), vc in zip(p["work"], vcs):
        rd[c.x][c.y] = [x.buffer_address(), scale.buffer_address(), wa, bnk, wq["pitch"], col0, nvalid, n0, NT, vc]
        wr[c.x][c.y] = [y.buffer_address(), NT, n0, nvalid, wa, bnk, wq["pitch"], col0, vc]

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=dev.FIDELITY[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=p["full_sync"],
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64  # one per CB index
    unpack[3] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the accumulator to dest in fp32
    cfg.unpack_to_dest_mode = unpack
    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("w4a16_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("w4a16_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel("w4a16_compute.cpp", compute_ct, ttnn.RuntimeArgs(), cfg),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([x, wq["t"], scale, y], program)
    return y


def dequantize(wq, scale, out=None, cores_per_bank=3):
    """The int4 weight times its group scales as a bfloat8_b weight [1, 1, K, N], DRAM width-sharded over the same
    banks (common/device.py sharded_memcfg): for an ordinary block-float matmul over many rows (prefill). out: a
    tensor of that shape and layout to write into (rewritten on every call)."""
    device = wq["t"].device()
    k, n, banks, ns = wq["k"], wq["n"], wq["banks"], wq["ns"]
    KT, NT = k // TILE, -(-n // TILE)
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, k, n]),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            device,
            dev.sharded_memcfg(device, k, n, banks),
        )
    g = device.compute_with_storage_grid_size()
    cpb = max(1, min(cores_per_bank, (g.x * g.y) // banks))
    per = -(-ns // cpb)
    sb = min(8, per)
    NC = -(-per // sb) * sb
    work = []
    for bnk in range(banks):
        valid = max(0, min(ns, NT - bnk * ns))
        for j in range(cpb):
            nvalid = max(0, min(per, valid - j * per))
            if nvalid:
                work.append((bnk, j * per, nvalid, bnk * ns + j * per))
    cores = dev.place(device, [wk[0] for wk in work], True)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    def cb(index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    wp = 2 * NC
    cbs = [
        cb(1, ttnn.bfloat4_b, BFP4_TILE, wp),
        cb(2, ttnn.bfloat16, 2048, 2 * NC),
        cb(8, ttnn.bfloat16, 2048, -(-NC * Q4_TILE // 2048)),
        cb(16, ttnn.bfloat8_b, 1088, 2 * NC),
    ]
    reader_ct = [KT, NC, wp, EXP, *ttnn.TensorAccessorArgs(scale).get_compile_time_args()]
    vcs = dev.read_vcs([(c, (bnk,)) for c, (bnk, *_r) in zip(cores, work)])
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (bnk, col0, nvalid, n0), vc in zip(cores, work, vcs):
        rd[c.x][c.y] = [scale.buffer_address(), wq["t"].buffer_address(), bnk, wq["pitch"], col0, nvalid, n0, NT, vc]
        wr[c.x][c.y] = [out.buffer_address(), bnk, ns, col0, nvalid]

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("w4dq_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("w4dq_writer.cpp", [KT, NC, sb], wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "w4dq_compute.cpp",
                [KT, NC, sb],
                ttnn.RuntimeArgs(),
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=True,
                    bfp8_pack_precise=PACK_PRECISE,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([wq["t"], scale, out], program)
    return out
