# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""moe8.py: a mixture-of-experts layer that reads only the chosen experts, on Tenstorrent Tensix devices, through
ttnn.generic_op. The expert ids are read on the device, so every op is traceable and nothing goes to the host.

Ops per MoE layer:
- route(logits, ids, w, k): the router's top k and their weights on one core (softmax or sigmoid scoring, an optional
  score bias, group-limited routing, a scaling factor, a fixed shared-expert slot), into the ids and weights tensors
  the other ops read;
- gate_up(x, P, ids, w, k) -> h [1, k * I]: for each chosen expert e (slot s), h_s = w_s * act(x G_e) * (x U_e), with
  the expert's gate and up tiles interleaved in the packed weight and w_s the routing weight; act "silu" (SwiGLU),
  "gelu_tanh" (GeGLU, HF's gelu_pytorch_tanh) or "swiglu_oai" (gpt-oss's clamped SwiGLU, min(g, 7) * sigmoid(1.702
  min(g, 7)) * (clamp(u, -7, 7) + 1)); an optional per-expert gate/up bias is added before the activation;
- down(h, P, ids, k) -> y [1, H]: one matmul over the k * I rows of the chosen experts' down matrices, gathered by id;
  the sum over the experts is the K accumulation.

The ids are uint16 (or uint32 with ids32=True) and the weights bf16, each a [1, 1, 1, k] tensor in TILE or ROW_MAJOR
layout: the decode kernels read the first 32 bytes of its first page, row 0, which holds the k values in either
layout (ttnn.topk's indices, or route's output).

The packed weights (pack, called per layer at load time; they replace the per-expert tensors): whole BFP4_b tiles of
W [K, N], never regrouped inside a tile, so values already on the BFP4 grid stay exact. bfp4_tiles makes the tile bytes
on the host exactly as tt-metal's host conversion does (the shared exponent of each 16 values of a face row, round to
nearest even). The weights are uint8 ROW_MAJOR tensors, DRAM width-sharded over the banks:
- gate|up: each expert's I / 32 gate/up tile pairs (gate j, up j: the GLU's order) are split over the banks, rank r =
  (bank - e) mod banks getting pairs // banks pairs and one more for r < pairs % banks, so the banks fill evenly; in a
  bank, expert e's block is its KT x 2 * pairs tiles, K row after K row, at the byte offset of the table row e
  (P["gu_table"]); more of an expert's tiles in a bank than dest holds go in column groups (gu_groups);
- down: H's column tiles are split over the banks and, in a bank, over its cores; in a bank, expert e's block is, per
  core j, its I / 32 x ncols_j tiles, contiguous.

R rows at once (prefill, batched decode; gate_up_rows and down_rows): x [1, 1, R, H], ids and w [1, 1, R, k] as TILE
tensors (R <= 32), each row with its own k experts. The work goes by expert: the union of the R x k ids, the distinct
experts in increasing order, each read once for all R rows (kernels/moe8_rows.h; every kernel builds it from the ids
tile, so nothing is read back and both ops are traceable). gate_up_rows' h [1, 1, 32, u_max * I] holds union index q's
I columns at q * I, scaled per row by that row's routing weight for the expert (0 where the row did not choose it);
down_rows sums over the union. Rows R .. 31 of x must be finite (TILE padding is 0); their outputs are 0. The ids of a
row must be distinct (top-k's are).

Two loaders: with loaders=2 the writer RISC of each core loads some of the weight chunks into a second circular buffer
(kernels/split.h's pattern, SPLIT_SHARE), a second request stream next to the reader's; the outputs are the same bit
for bit. The decode ops' column groups and gate/up bias need it.

Entry points: bfp4_tiles, bfp4_values, pack (cache=: the host result kept per layer) = pack_upload of pack_host,
prepack, pack_cached, repack_down, route, gate_up, down, gate_up_rows, down_rows, reference, reference_rows,
route_reference, union, u_max, sigmoid_table.
"""

import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device as dev  # noqa: E402
from common import source as src  # noqa: E402
from common.device import FIDELITY, TILE, bank_grid, dram_banks, place, read_vcs, split_arg  # noqa: E402

KERNELS = Path(__file__).resolve().parent / "kernels"
TB = 576  # a BFP4_b tile: 64 exponent bytes, 512 bytes of sign + 3-bit magnitudes
# gate_up's activation (moe_gu_compute.cpp ACT): silu (SwiGLU), gelu_tanh (GeGLU, HF's gelu_pytorch_tanh) or swiglu_oai
# (gpt-oss's clamped SwiGLU with alpha 1.702 and limit 7, min(g, 7) * sigmoid(1.702 min(g, 7)) * (clamp(u, -7, 7) + 1))
ACTS = {"silu": 0, "gelu_tanh": 1, "swiglu_oai": 2}
OAI_ALPHA, OAI_LIMIT = 1.702, 7.0

# --- settings (module attributes: starting points chosen from measurements on one device; retune them per device) --

# decode gate_up's SFPU stages (activation, products, routing weight) on the valid rows only (one row of x: the top
# half of the tiles, 2 SFPU iterations); False: whole tiles. The same values in row 0 either way.
ROWS_SFPU = True
# gate_up's cores per bank when the call names none (gate_up: GU_CPB; gate_up_rows: GU_ROWS_CPB); 0: the pack's. A work
# split only: each bank holds every expert's pairs whatever the cores, so any count gives the same h.
GU_CPB = 0
GU_ROWS_CPB = 0
# loaders when the call names none: 2, the writer RISC as a second loader (decode: needed for column groups and a
# bias), or 1 (one loader, the reader RISC); the same outputs bit for bit
LOADERS = 2
ROWS_LOADERS = 1
# the reader RISC's share of the weight chunks with two loaders (0.5: the alternation; 0.5 to 0.75, split.h)
SPLIT_SHARE = 0.5
# L1 kept free next to an op's circular buffers (they are checked against the device's free L1 before the launch)
L1_MARGIN = 32 * 1024
GW_MAX = 8  # gate_up's dest: 8 fp32 tiles with full sync (4 gate/up pairs)
PACK_VERSION = 1  # the host layout below; a cache of another version is made again


def source(name):
    """A kernel's source with its local headers inlined (common/source.py)."""
    return src.load(KERNELS, name)


def _fit_kw(it, kw):
    """A chunk depth (K rows a chunk) for a K of `it` tiles (down: I's; gate_up: H's): kw as asked when it divides `it`
    or `it` divides it (min(kw, it)), else the divisor of `it` nearest kw (the larger on a tie): an I of 704 (22 tiles)
    gets 11 for kw 8, an H of 2880 (90 tiles) 9."""
    if it % kw == 0 or kw % it == 0:
        return min(kw, it)
    return min((d for d in range(1, it + 1) if it % d == 0), key=lambda d: (abs(d - kw), -d))


def sigmoid_table():
    """moe_route.cpp's bounds table (kernels/moe_sigmoid_q30.h): sigmoid(i / 16 - 16), i = 0 .. 512, in Q2.30,
    rounded."""
    return [round((1.0 / (1.0 + math.exp(-(i / 16 - 16)))) * (1 << 30)) for i in range(513)]


# --- BFP4_b tiles on the host -----------------------------------------------------------------------------------------


def bfp4_tiles(w, kblock=False):
    """float [K, N] (K, N multiples of 32) -> uint8 [KT, NT, 576]: each 32 x 32 tile as BFP4_b (kblock: each tile holds
    its block transposed, so a face row of 16 is 16 consecutive K at one N: exponent blocks along K, as GPTQ along K
    exports them; the matmul takes in1 transposed), as tt-metal's host conversion makes them: per face row of 16 values
    the shared exponent is the largest biased float32 exponent; each mantissa (with its hidden bit) is shifted down by
    its exponent's distance from it, then rounded to 3 bits, ties to even, at most 7; the sign is kept only for a
    non-zero result. Exponents first (face by face, row by row), then two values per byte (the first in the low
    nibble), face by face, row-major."""
    w = np.ascontiguousarray(w.float().numpy() if isinstance(w, torch.Tensor) else w, np.float32)
    k, n = w.shape
    assert k % TILE == 0 and n % TILE == 0, w.shape
    t = w.reshape(k // 32, 32, n // 32, 32).transpose(0, 2, 1, 3)  # [KT, NT, 32, 32]
    if kblock:
        t = t.swapaxes(-1, -2)  # [KT, NT, 32 (N), 32 (K)]
    # face order: [KT, NT, face (2 x 2), 16 rows, 16 cols]
    f = t.reshape(k // 32, n // 32, 2, 16, 2, 16).transpose(0, 1, 2, 4, 3, 5)
    f = np.ascontiguousarray(f).reshape(k // 32, n // 32, 64, 16)  # 64 face rows of 16 values
    bits = f.view(np.uint32)
    exp = (bits >> 23) & 0xFF
    sign = bits >> 31
    man = (bits & 0x7FFFFF) | (1 << 23)
    shared = exp.max(axis=-1, keepdims=True)
    diff = (shared - exp).astype(np.int64)
    m = np.where(diff > 31, 0, man.astype(np.int64) >> np.minimum(diff, 31))
    # round 24 bits to 3: <2 bits> guard <21 bits>
    shift = 21
    rnd = m & ((1 << shift) - 1)
    m = m >> shift
    guard = m & 1
    tie = 1 << (shift - 1)
    m = m + ((rnd > tie) | ((rnd == tie) & (guard == 1)))
    m = np.minimum(m, 7)
    m = np.where(exp == 0, 0, m)  # zero or denormal
    s = np.where(m == 0, 0, sign)
    code = (s << 3 | m).astype(np.uint8).reshape(k // 32, n // 32, 1024)
    body = code[..., 0::2] | (code[..., 1::2] << 4)
    head = shared.reshape(k // 32, n // 32, 64).astype(np.uint8)
    return np.concatenate([head, body], axis=-1)


def bfp4_values(tiles, kblock=False):
    """uint8 [..., 576] BFP4_b tiles -> float32 [..., 32, 32] (row-major tile values; kblock: bfp4_tiles(kblock=True)'s,
    turned back to the block's [K, N] order)."""
    tiles = np.asarray(tiles)
    lead = tiles.shape[:-1]
    head = tiles[..., :64].astype(np.int32)  # [..., 64]
    body = tiles[..., 64:]
    code = np.stack([body & 15, body >> 4], axis=-1).reshape(*lead, 64, 16).astype(np.int32)
    mag = (code & 7).astype(np.float32)
    sgn = np.where(code & 8, -1.0, 1.0).astype(np.float32)
    val = sgn * mag * np.exp2((head - 129).astype(np.float32))[..., None]
    val = val.reshape(*lead, 2, 2, 16, 16).swapaxes(-3, -2).reshape(*lead, 32, 32)
    return val.swapaxes(-1, -2) if kblock else val


# --- the packed layer -------------------------------------------------------------------------------------------------


def _split(total, parts):
    """total split into parts, the first total % parts one larger: [(start, count)]."""
    q, r = divmod(total, parts)
    out, s = [], 0
    for i in range(parts):
        c = q + (1 if i < r else 0)
        out.append((s, c))
        s += c
    return out


def _sharded(device, data, banks):
    """uint8 [banks, bytes] (each a bank's shard, a multiple of 64 bytes) as a DRAM width-sharded uint8 ROW_MAJOR
    tensor: one row of the shard per 64 bytes."""
    nbytes = data.shape[1]
    assert nbytes % 64 == 0
    rows = nbytes // 64
    t = torch.from_numpy(np.ascontiguousarray(data.reshape(banks, rows, 64).transpose(1, 0, 2)))
    t = t.reshape(rows, banks * 64)
    spec = ttnn.ShardSpec(bank_grid(banks), (rows, 64), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)


def gu_pairs(e, bank, banks, pairs=16):
    """(first pair, pairs) of expert e's gate/up tile pairs held by bank (rotating by expert)."""
    return _split(pairs, banks)[(bank - e) % banks]


def gu_groups(it, banks):
    """(G, GW): gate_up's column groups per expert and bank, and their width in tiles. The widest bank's 2 * ceil(it /
    banks) tiles in one group while they fit in dest, else groups of GW_MAX tiles (an I of 2880 at 8 banks: 24 tiles
    in 8, 8 and 8). In a bank, an expert's block is its groups one after the other, each KT x (its tiles), K row
    after K row; groups past the expert's tiles in that bank are empty (the kernels zero-fill them)."""
    ncmax = 2 * -(-it // banks)
    if ncmax <= GW_MAX:
        return 1, ncmax
    return -(-ncmax // GW_MAX), GW_MAX


def _fingerprint(t):
    """A cheap identity of a weight tensor for the pack cache: its shape, dtype and the sha256 of every 4093rd value
    (as float32), so a cache made from other weights is not taken."""
    t = torch.as_tensor(t)
    sample = t.reshape(-1)[::4093].float().contiguous().numpy()
    h = hashlib.sha256(sample.tobytes()).hexdigest()[:32]
    return f"{tuple(t.shape)}:{t.dtype}:{h}"


def _bias_tiles(b, it):
    """An expert's gate/up bias [2I] as [up | gate] -> [2 * it, 64] uint8: per tile of the GLU's (gate j, up j) order its
    32 bf16 values, row 0 of the tile (faces 0 and 1: columns 0-15, then 16-31)."""
    v = torch.as_tensor(b).to(torch.bfloat16).reshape(2, it, TILE)  # (up, gate) tiles
    v = v.flip(0).transpose(0, 1).reshape(2 * it, TILE)  # (gate j, up j)
    return v.contiguous().view(torch.int16).numpy().view(np.uint8).reshape(2 * it, 64)


def _pack_host(gate_up, down, banks, cpb, gu_bias=None, kblock=False):
    """The host part of pack: (gu_data [banks, size], dn_data [banks, ...], table [E, 16]). gu_bias [E, 2I] as [up |
    gate]: in each bank right after the expert's tiles (all its groups), 64 bytes a tile of its tiles here (bf16, row 0
    of the tile), so the table's offset finds both."""
    E, H, I2 = gate_up.shape
    I = I2 // 2
    assert banks <= 16, "moe8: a table row holds 16 banks"
    kt, it, ht = H // TILE, I // TILE, H // TILE
    npairs = it  # gate/up tile pairs per expert
    # gate|up: per expert the tiles [KT, 2 * npairs], pair j = (gate j, up j); converted one expert at a time
    cols = _split(ht, banks)
    per_core = [_split(c, cpb) for _, c in cols]
    width = max(c for _, c in cols)  # an expert's columns per bank, padded to the widest bank
    G, GW = gu_groups(it, banks)
    per_tile = TB * kt + (64 if gu_bias is not None else 0)  # an expert's bytes a tile in a bank
    sizes = [sum(2 * gu_pairs(e, b, banks, npairs)[1] * per_tile for e in range(E)) for b in range(banks)]
    size = -(-max(sizes) // 64) * 64
    gu_data = np.zeros((banks, size), np.uint8)
    dn_data = np.zeros((banks, E * it * width * TB), np.uint8)
    table = np.zeros((E, 16), np.uint32)
    fill = [0] * banks
    for e in range(E):
        t = bfp4_tiles(gate_up[e], kblock=kblock).reshape(kt, 2, it, TB)  # (up, gate) tiles
        t = t[:, ::-1].transpose(0, 2, 1, 3).reshape(kt, 2 * it, TB)  # (gate j, up j)
        bt = _bias_tiles(gu_bias[e], it) if gu_bias is not None else None
        for b in range(banks):
            p0, pc = gu_pairs(e, b, banks, npairs)
            blk = t[:, 2 * p0 : 2 * (p0 + pc)]  # KT x 2pc tiles, row after row
            if G > 1:  # its column groups one after the other, each KT x (its tiles)
                blk = np.concatenate([blk[:, c : c + GW].reshape(-1) for c in range(0, 2 * pc, GW)])
            blk = blk.reshape(-1)
            if bt is not None:
                blk = np.concatenate([blk, bt[2 * p0 : 2 * (p0 + pc)].reshape(-1)])
            table[e, b] = fill[b]
            gu_data[b, fill[b] : fill[b] + blk.size] = blk
            fill[b] += blk.size
        d = bfp4_tiles(down[e], kblock=kblock).reshape(it, ht, TB)  # [IT, HT, 576]
        base = e * it * width * TB
        for b, (c0, _) in enumerate(cols):
            for j0, jc in per_core[b]:
                blk = d[:, c0 + j0 : c0 + j0 + jc].reshape(-1)  # IT x jc tiles
                off = base + it * j0 * TB
                dn_data[b, off : off + blk.size] = blk
    return gu_data, dn_data, table


class PackCacheMismatch(RuntimeError):
    """A pack cache that does not match the call (pack, pack_host, prepack, pack_cached). It is neither taken nor
    overwritten: call with the cache's own split, or pass rebuild=True to pack again from the weights and overwrite
    it."""


def _short(v):
    v = repr(v)
    return v if len(v) <= 48 else v[:45] + "..."


def _cache_read(path, meta, rebuild=False):
    """(gu_data, dn_data, table) from a pack cache made with the same meta; None when there is none (no directory, or
    an empty one), so the caller makes it. A cache that differs from meta, or that cannot be read, raises
    PackCacheMismatch, and nothing is written; with rebuild it is None, and the caller makes it again over it."""
    path = Path(path)
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return None
    try:
        old = json.loads((path / "meta.json").read_text())
    except (OSError, ValueError) as e:
        old, why = None, f"its meta.json cannot be read ({type(e).__name__})"
    if old is not None and not isinstance(old, dict):
        old, why = None, "its meta.json is not a pack's"
    split_only = False
    if old == meta:
        try:
            return tuple(np.load(path / f"{n}.npy", mmap_mode="r") for n in ("gu", "dn", "table"))
        except (OSError, ValueError) as e:
            why = f"its arrays cannot be read ({type(e).__name__})"
    elif old is not None:
        keys = sorted(set(old) | set(meta), key=str)
        diff = [k for k in keys if old.get(k) != meta.get(k)]
        why = "; ".join(f"{k}: cache {_short(old.get(k))}, call {_short(meta.get(k))}" for k in diff)
        split_only = set(diff) <= {"cpb", "banks"}
    if rebuild:
        return None
    hint = ""
    if split_only:
        hint = f" Only the split differs: call with cores_per_bank={old.get('cpb')}, banks={old.get('banks')} to use it."
    raise PackCacheMismatch(
        f"moe8 pack cache {path} does not match this call ({why}). It was neither taken nor overwritten.{hint} "
        "To pack again from these weights over it, pass rebuild=True."
    )


def _array_sums(arrays):
    """{gu, dn, table: the sha256 of the array's bytes} (C order, as np.save writes them): pack_cached's check of a
    cache's arrays against the ones it was written with (sums.json, beside meta.json)."""
    out = {}
    for n, a in zip(("gu", "dn", "table"), arrays):
        h = hashlib.sha256()
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
        out[n] = h.hexdigest()
    return out


def _cache_write(path, meta, arrays):
    """The pack cache, whole or absent: written into PATH.tmp, then renamed. sums.json holds the arrays' sha256."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for n, a in zip(("gu", "dn", "table"), arrays):
        np.save(tmp / f"{n}.npy", a)
    (tmp / "sums.json").write_text(json.dumps(_array_sums(arrays), indent=1))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def _host_cached(gate_up, down, banks, cpb, cache, gu_bias=None, rebuild=False, kblock=False):
    """_pack_host's result, from the cache when its meta matches; made and written when there is no cache; a cache
    that differs raises PackCacheMismatch unless rebuild (then it is made again and written over)."""
    if cache is None:
        return _pack_host(gate_up, down, banks, cpb, gu_bias, kblock)
    E, H, I2 = gate_up.shape
    meta = {
        "version": PACK_VERSION,
        "banks": banks,
        "cpb": cpb,
        "E": E,
        "H": H,
        "I": I2 // 2,
        "gate_up": _fingerprint(gate_up),
        "down": _fingerprint(down),
    }
    G = gu_groups((I2 // 2) // TILE, banks)[0]
    if G > 1:
        meta["gu_groups"] = G
    if gu_bias is not None:
        meta["gu_bias"] = _fingerprint(gu_bias)
    if kblock:
        meta["kblock"] = True
    host = _cache_read(cache, meta, rebuild)
    if host is None:
        host = _pack_host(gate_up, down, banks, cpb, gu_bias, kblock)
        _cache_write(cache, meta, host)
    return host


def repack_down(cache, cores_per_bank=2):
    """A pack cache moved to another split of down's columns per core, without the weights: down's blocks are whole
    tiles, so only their order in dn.npy changes (gate_up's layout and the table do not depend on the cores). The result
    equals a fresh pack at cores_per_bank bit for bit, its meta too. True when it was converted, False when the cache
    already has that split. Written beside the cache, then swapped, as _cache_write does."""
    cache = Path(cache)
    meta = json.loads((cache / "meta.json").read_text())
    assert meta["version"] == PACK_VERSION, (cache, meta["version"], PACK_VERSION)
    old, new = meta["cpb"], cores_per_bank
    if old == new:
        return False
    E, H, I, banks = meta["E"], meta["H"], meta["I"], meta["banks"]
    ht, it = H // TILE, I // TILE
    cols = _split(ht, banks)
    row = it * max(c for _, c in cols) * TB  # an expert's bytes in a bank, padded to the widest bank
    dn = np.load(cache / "dn.npy")
    assert dn.shape == (banks, E * row), (dn.shape, banks, E, row)
    out = np.zeros_like(dn)
    for b, (_, c) in enumerate(cols):
        src_, dst = dn[b].reshape(E, row), out[b].reshape(E, row)
        tiles = np.empty((E, it, c, TB), np.uint8)  # [E, K tiles, the bank's columns, a tile's bytes]
        for j0, jc in _split(c, old):
            tiles[:, :, j0 : j0 + jc] = src_[:, it * j0 * TB : it * (j0 + jc) * TB].reshape(E, it, jc, TB)
        for j0, jc in _split(c, new):
            dst[:, it * j0 * TB : it * (j0 + jc) * TB] = tiles[:, :, j0 : j0 + jc].reshape(E, it * jc * TB)
    tmp, gone = cache.with_name(cache.name + ".tmp"), cache.with_name(cache.name + ".old")
    for d in (tmp, gone):
        shutil.rmtree(d, ignore_errors=True)
    tmp.mkdir(parents=True)
    for n in ("gu.npy", "table.npy"):
        os.link(cache / n, tmp / n)
    np.save(tmp / "dn.npy", out)
    if (cache / "sums.json").exists():  # dn's sum follows its new order (gu and the table are linked unchanged)
        sums = json.loads((cache / "sums.json").read_text())
        sums["dn"] = _array_sums((np.zeros(0, np.uint8), out, np.zeros(0, np.uint8)))["dn"]
        (tmp / "sums.json").write_text(json.dumps(sums, indent=1))
    meta["cpb"] = new
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    cache.rename(gone)
    tmp.rename(cache)
    shutil.rmtree(gone)
    return True


def prepack(gate_up, down, cache, banks, cores_per_bank=3, gu_bias=None, rebuild=False, kblock=False):
    """pack()'s host part into its cache, without a device: the cache pack(..., cache=) or pack_cached then takes on a
    device with `banks` DRAM banks and cores_per_bank cores each. True when it was made, False when the cache already
    held these weights. A cache that differs (other weights, another split, an older layout) raises PackCacheMismatch
    unless rebuild=True, as pack's."""
    E, H, I2 = gate_up.shape
    assert down.shape == (E, I2 // 2, H)
    before = (Path(cache) / "meta.json").read_text() if (Path(cache) / "meta.json").exists() else None
    _host_cached(gate_up, down, banks, cores_per_bank, cache, gu_bias, rebuild, kblock)
    return (Path(cache) / "meta.json").read_text() != before


def pack_host(gate_up, down, banks, cores_per_bank=3, cache=None, gu_bias=None, rebuild=False, kblock=False):
    """pack()'s host part, no device: a dict of numpy arrays and numbers that pack_upload puts on a device with at
    least `banks` DRAM banks and room for cores_per_bank cores each. Keys: E, H, I, banks, cpb, version, gu_data
    [banks, bytes], dn_data [banks, bytes], table [E, 16] uint32, dn_cols, dn_core, dn_width, gu_bias (bool), kblock
    (bool). cache, gu_bias, rebuild, kblock: as pack()'s."""
    E, H, I2 = gate_up.shape
    I = I2 // 2
    assert down.shape == (E, I, H), (down.shape, (E, I, H))
    assert H % TILE == 0 and I % TILE == 0
    cpb = cores_per_bank
    cols = _split(H // TILE, banks)
    per_core = [_split(c, cpb) for _, c in cols]
    if gu_bias is not None:
        assert tuple(gu_bias.shape) == (E, 2 * I), (tuple(gu_bias.shape), (E, 2 * I))
    gu_data, dn_data, table = _host_cached(gate_up, down, banks, cpb, cache, gu_bias, rebuild, kblock)
    return {
        "version": PACK_VERSION,
        "E": E,
        "H": H,
        "I": I,
        "banks": banks,
        "cpb": cpb,
        "gu_data": gu_data,
        "dn_data": dn_data,
        "table": table,
        "dn_cols": cols,
        "dn_core": per_core,
        "dn_width": max(c for _, c in cols),
        "gu_bias": gu_bias is not None,
        "kblock": bool(kblock),
    }


def pack_upload(device, host):
    """pack_host's dict on the device: the dict the ops take. The device must have at least the host dict's banks
    (dram_banks, which honors common.device.MAX_BANKS) and room for its cores per bank."""
    banks, cpb = host["banks"], host["cpb"]
    assert host["version"] == PACK_VERSION, (host["version"], PACK_VERSION)
    assert banks <= dram_banks(device), (banks, dram_banks(device))
    g = device.compute_with_storage_grid_size()
    assert cpb == max(1, min(cpb, (g.x * g.y) // banks)), (cpb, g)
    gu_data, dn_data = host["gu_data"], host["dn_data"]
    tt = ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(host["table"]).view(np.int32)).contiguous(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return {
        "E": host["E"],
        "H": host["H"],
        "I": host["I"],
        "banks": banks,
        "cpb": cpb,
        "gu": _sharded(device, gu_data, banks),
        "gu_table": tt,
        "dn": _sharded(device, dn_data, banks),
        "dn_cols": host["dn_cols"],
        "dn_core": host["dn_core"],
        "dn_width": host["dn_width"],
        "gu_bias": bool(host.get("gu_bias", False)),
        "kblock": bool(host.get("kblock", False)),
        "bytes": banks * (gu_data.shape[1] + dn_data.shape[1]),
    }


def _clamp_cpb(device, banks, cores_per_bank):
    g = device.compute_with_storage_grid_size()
    return max(1, min(cores_per_bank, (g.x * g.y) // banks))


def pack(device, gate_up, down, cores_per_bank=3, banks=None, cache=None, gu_bias=None, rebuild=False, kblock=False):
    """One MoE layer's experts as moe8's packed weights: pack_upload(device, pack_host(...)) with the device's banks
    (dram_banks; or `banks`, the first ones) and cores_per_bank cores a bank (at most the grid's share). gate_up:
    [E, H, 2I] float with up in columns 0..I-1 and gate in I..2I-1 ([up | gate]); down: [E, I, H]. Returns the dict the
    ops take.

    cache: a directory for this layer's host result (gu.npy, dn.npy, table.npy, meta.json, sums.json). Taken when its
    meta matches (the layout version, the banks and cores, the shapes and a fingerprint of the weights); made and
    written when the directory is absent or empty. A cache that differs (other weights, another split, an older layout)
    or cannot be read raises PackCacheMismatch and is left as it is; rebuild=True packs again from the weights and
    writes over it.

    gu_bias: [E, 2I] as [up | gate], added to x G and x U before the activation, held in bf16 (gate_up with two
    loaders and gate_up_rows add it when the pack has it).

    kblock: each 32 x 32 block of gate_up and down held transposed in its BFP4_b tile, so the shared exponents run
    along K (the input) in blocks of 16: the layout of a GPTQ-along-K export, whose values it holds exactly; every op
    then runs its matmul with in1 transposed within the tile (KBLOCK)."""
    banks = banks or dram_banks(device)
    cpb = _clamp_cpb(device, banks, cores_per_bank)
    return pack_upload(device, pack_host(gate_up, down, banks, cpb, cache, gu_bias, rebuild, kblock))


def pack_cached(device, cache, cores_per_bank=3, banks=None, kblock=False, gu_bias=None, verify=True):
    """pack()'s result from a pack cache alone, without the weights (prepack's, for a model whose expert shards are
    packed one at a time on another machine). The call must match the cache's meta: the layout version, the banks and
    cores per bank (clamped to the device as pack() does), kblock, and the bias: gu_bias None (the cache has none), True
    (it has one), or the [E, 2I] tensor itself (its fingerprint must equal the cache's). verify: every array's sha256
    against the cache's sums.json; a cache without sums.json is refused unless verify=False. Anything that differs
    raises PackCacheMismatch, and nothing is changed."""
    path = Path(cache)
    try:
        meta = json.loads((path / "meta.json").read_text())
    except (OSError, ValueError) as e:
        raise PackCacheMismatch(f"moe8 pack cache {path}: no readable meta.json ({type(e).__name__})") from e
    banks = banks or dram_banks(device)
    cpb = _clamp_cpb(device, banks, cores_per_bank)
    want = {"version": PACK_VERSION, "banks": banks, "cpb": cpb, "kblock": bool(kblock), "a bias": gu_bias is not None}
    have = {
        "version": meta.get("version"),
        "banks": meta.get("banks"),
        "cpb": meta.get("cpb"),
        "kblock": bool(meta.get("kblock", False)),
        "a bias": "gu_bias" in meta,
    }
    diff = [f"{k}: cache {have[k]}, call {want[k]}" for k in want if want[k] != have[k]]
    bias_t = gu_bias is not None and gu_bias is not True  # (a tensor: its fingerprint is compared)
    if bias_t and "gu_bias" in meta and _fingerprint(gu_bias) != meta["gu_bias"]:
        diff.append("the bias's fingerprint")
    if diff:
        raise PackCacheMismatch(f"moe8 pack cache {path} does not match this call ({'; '.join(diff)}). It was not taken.")
    try:
        arrays = tuple(np.load(path / f"{n}.npy", mmap_mode="r") for n in ("gu", "dn", "table"))
    except (OSError, ValueError) as e:
        raise PackCacheMismatch(f"moe8 pack cache {path}: its arrays cannot be read ({type(e).__name__})") from e
    if verify:
        try:
            sums = json.loads((path / "sums.json").read_text())
        except (OSError, ValueError) as e:
            raise PackCacheMismatch(
                f"moe8 pack cache {path} has no readable sums.json; pack it again, or pass verify=False to take it"
                " unchecked"
            ) from e
        got = _array_sums(arrays)
        bad = [n for n in got if sums.get(n) != got[n]]
        if bad:
            raise PackCacheMismatch(f"moe8 pack cache {path}: {', '.join(bad)} differ from sums.json. It was not taken.")
    E, H, I = meta["E"], meta["H"], meta["I"]
    cols = _split(H // TILE, banks)
    gu_data, dn_data, table = arrays
    table = np.array(table)  # (small; a writable copy for torch.from_numpy)
    assert gu_data.shape[0] == banks and dn_data.shape[0] == banks and table.shape[0] == E, (
        gu_data.shape,
        dn_data.shape,
        table.shape,
        banks,
        E,
    )
    host = {
        "version": PACK_VERSION,
        "E": E,
        "H": H,
        "I": I,
        "banks": banks,
        "cpb": cpb,
        "gu_data": gu_data,
        "dn_data": dn_data,
        "table": table,
        "dn_cols": cols,
        "dn_core": [_split(c, cpb) for _, c in cols],
        "dn_width": max(c for _, c in cols),
        "gu_bias": "gu_bias" in meta,
        "kblock": bool(kblock),
    }
    return pack_upload(device, host)


# --- references -------------------------------------------------------------------------------------------------------


def _glu(gt, u, act):
    """The references' act(x G) * (x U): silu, gelu_tanh (HF's gelu_pytorch_tanh), or swiglu_oai (the gate clamped
    above at 7, the up half to +-7 and shifted by 1)."""
    assert act in ACTS, f"moe8: act {act!r}"
    if act == "swiglu_oai":
        g = gt.clamp(max=OAI_LIMIT)
        return g * torch.sigmoid(OAI_ALPHA * g) * (u.clamp(-OAI_LIMIT, OAI_LIMIT) + 1)
    if act == "gelu_tanh":
        return torch.nn.functional.gelu(gt, approximate="tanh") * u
    return torch.nn.functional.silu(gt) * u


def reference(x, gate_up, down, ids, w, act="silu", gu_bias=None):
    """The layer's output y [1, H] in float64, and h [1, k * I]: sum over the chosen experts of w_s * (act(x G) * (x
    U)) @ D, gate_up [E, H, 2I] as [up | gate], down [E, I, H], all as given (BFP4-grid values); gu_bias [E, 2I] as
    [up | gate] (rounded to bf16, as packed) added to x U and x G."""
    x = x.double().reshape(1, -1)
    I = gate_up.shape[-1] // 2
    y = torch.zeros(1, gate_up.shape[1], dtype=torch.float64)
    hs = []
    for s, e in enumerate(ids):
        u = x @ gate_up[e, :, :I].double()
        gt = x @ gate_up[e, :, I:].double()
        if gu_bias is not None:
            b = torch.as_tensor(gu_bias[e]).to(torch.bfloat16).double()
            u, gt = u + b[:I], gt + b[I:]
        h = _glu(gt, u, act) * float(w[s])
        hs.append(h)
        y += h @ down[e].double()
    return y, torch.cat(hs, dim=1)


def reference_rows(x, gate_up, down, ids, w, act="silu", gu_bias=None):
    """y [R, H] in float64 for R rows: row r's sum over its chosen experts ids[r] of w[r, s] * (act(x_r G) * (x_r U))
    @ D. x [R, H]; ids, w [R, k]."""
    return torch.cat(
        [reference(x[r : r + 1], gate_up, down, ids[r], w[r], act, gu_bias)[0] for r in range(len(ids))]
    )


def union(ids, E):
    """The R x k ids' distinct experts below E in increasing order (the kernels' union)."""
    return sorted({int(e) for e in torch.as_tensor(ids).reshape(-1).tolist() if 0 <= int(e) < E})


# --- the ops ----------------------------------------------------------------------------------------------------------

_PROGRAMS = {}


def _cbs(device, cores, specs, what):
    """Circular buffers [(index, dtype, page, pages)] as descriptors, after a check that they fit the L1 free on the
    device (less L1_MARGIN)."""
    total = sum(page * pages for _, _, page, pages in specs)
    free = dev.l1_free(device)
    if free is not None and total > free - L1_MARGIN:
        raise RuntimeError(f"moe8 {what}: {total} B of circular buffers a core, {free} B of L1 free")
    return [
        ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )
        for i, dt, page, pages in specs
    ]


def _kernel(cores, name, ct, rt, config):
    return ttnn.KernelDescriptor(
        kernel_source=source(name),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=config,
    )


def _work(device, P, cpb=None):
    """[(core, bank, j)]: cpb cores per bank (P's unless given), placed next to their banks (common.device.place)."""
    banks, cpb = P["banks"], cpb or P["cpb"]
    items = [(b, j) for b in range(banks) for j in range(cpb)]
    cores = place(device, [b for b, _ in items], True)
    return [(c, b, j) for c, (b, j) in zip(cores, items)]


def _gu_cpb(device, P, cpb, default=0):
    """gate_up's cores per bank: cpb, else default (capped at the grid's cores per bank), else P's (the pack's). Its
    work split only, not the packed layout (each bank holds every expert's pairs whatever the cores)."""
    g = device.compute_with_storage_grid_size()
    most = (g.x * g.y) // P["banks"]
    if not cpb:
        return max(1, min(default, most)) if default else P["cpb"]
    assert 1 <= cpb <= most, (cpb, g, P["banks"])
    return cpb


def _loaders(loaders, default):
    loaders = default if loaders is None else loaders
    assert loaders in (1, 2), loaders
    return loaders == 2


def gate_up(x, P, ids, w, k, *, ids32=False, kw=8, fidelity="hifi2", memory_config=None, cpb=None, loaders=None, act="silu"):
    """h [1, 1, 1, k * I] bf16 TILE: per slot s the routing weight w[s] times act(x G) * (x U) of expert ids[s]. x [1,
    1, 1, H] bf16 TILE interleaved; ids, w: see the module docstring. cpb: the cores per bank of this op (default
    GU_CPB; 0: the pack's). loaders: 1 or 2 (default LOADERS); column groups and a bias need 2. act: "silu" (SwiGLU),
    "gelu_tanh" (GeGLU) or "swiglu_oai" (gpt-oss's clamped SwiGLU)."""
    assert act in ACTS, f"moe8: act {act!r}"
    assert 1 <= k <= 16, k
    device = x.device()
    H, I, banks = P["H"], P["I"], P["banks"]
    cpb = _gu_cpb(device, P, cpb, GU_CPB)
    split = _loaders(loaders, LOADERS)
    sa = split_arg(split, SPLIT_SHARE)  # the chunks' pattern between the loaders (kernels/split.h)
    kt, it = H // TILE, I // TILE
    kw = _fit_kw(kt, kw)
    G, ncmax = gu_groups(it, banks)  # ncmax: a group's tiles (one group: the widest bank's per expert)
    assert G == 1 or split, "moe8: gate_up's column groups need two loaders (loaders=2)"
    bias = P.get("gu_bias", False)
    assert not bias or split, "moe8: gate_up's bias needs two loaders (loaders=2)"
    mc = memory_config or ttnn.L1_MEMORY_CONFIG
    h = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, k * I]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    work = _work(device, P, cpb)
    key = ("gu", id(P["gu"]), P["gu"].buffer_address(), bool(P.get("kblock", False)), k, ids32, kw, fidelity)
    key += (str(x.memory_config()), str(mc), cpb, split, sa, act, ROWS_SFPU, ids.buffer_address(), w.buffer_address())
    prog = _PROGRAMS.get(key)
    if prog is None:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
        bf = ttnn.bfloat16
        # with split the writer drains the products only after its loads: c_16 holds every unit's
        out_pages = max(ncmax, -(-k // cpb) * G * (ncmax // 2)) if split else ncmax
        specs = [
            (0, bf, 2048, kt),  # x, all of it
            (1, ttnn.bfloat4_b, TB, 3 * kw * ncmax),  # weight chunks
            (2, bf, 2048, 2),  # the routing weight's tile (w in row 0), per unit
            (9, bf, 2048, 1),  # scratch: ids, weights, the table row(s)
            (10, bf, 2048, 1),  # the writer's scratch: ids (split: the table rows)
            (16, bf, 2048, out_pages),  # out
        ]
        if split:
            specs.append((4, ttnn.bfloat4_b, TB, 3 * kw * ncmax))  # the writer's chunks
        if bias:
            specs.append((5, bf, 2048, 2 * ncmax))  # a unit's bias tiles (row 0)
            specs.append((6, bf, 64 * ncmax, 1))  # the reader's bias staging
        cbs = _cbs(device, cores, specs, "gate_up")
        reader_ct = [kt, kw, ncmax, k, cpb, banks, it, int(ids32)]
        if split:
            reader_ct += [sa, G, int(bias)]
        for t in (x, ids, w, P["gu_table"]):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = [ncmax, k, cpb, banks, it, int(ids32)]
        if split:
            writer_ct += [sa, kt, kw, G]
        for t in (h, ids, P["gu_table"]) if split else (h, ids):
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        # one row of x: the SFPU stages on its tiles' top half, over 2 iterations (4 rows)
        compute_ct = [kt, kw, ncmax, 2 if ROWS_SFPU else 0, 0, sa, ACTS[act], int(bias), int(P.get("kblock", False))]
        vcs = read_vcs([(c, (b,)) for c, b, _ in work])
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (c, b, j), vc in zip(work, vcs):
            units = len(range(j, k, cpb)) * G
            rd[c.x][c.y] = [
                x.buffer_address(),
                P["gu"].buffer_address(),
                P["gu_table"].buffer_address(),
                ids.buffer_address(),
                w.buffer_address(),
                b,
                j,
                vc,
            ]
            wa = [h.buffer_address(), ids.buffer_address(), b, j]
            if split:
                wa += [P["gu"].buffer_address(), P["gu_table"].buffer_address(), vc]
            wr[c.x][c.y] = wa
            cp[c.x][c.y] = [units]
        prog = ttnn.ProgramDescriptor(
            kernels=[
                _kernel(
                    cores, "moe_gu2_reader.cpp" if split else "moe_gu_reader.cpp", reader_ct, rd,
                    ttnn.ReaderConfigDescriptor(),
                ),
                _kernel(
                    cores, "moe_gu2_writer.cpp" if split else "moe_gu_writer.cpp", writer_ct, wr,
                    ttnn.WriterConfigDescriptor(),
                ),
                _kernel(
                    cores,
                    "moe_gu_compute.cpp",
                    compute_ct,
                    cp,
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=FIDELITY[fidelity],
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:
        rd, wr = prog.kernels[0].runtime_args, prog.kernels[1].runtime_args
        for c, _, _ in work:
            rd[c.x][c.y][0] = x.buffer_address()
            wr[c.x][c.y][0] = h.buffer_address()
    ttnn.generic_op([x, P["gu"], P["gu_table"], ids, w, h], prog)
    return h


def down(h, P, ids, k, *, ids32=False, kw=8, fidelity="hifi2", memory_config=None, loaders=None):
    """y [1, 1, 1, H] bf16 TILE = h @ the chosen experts' down rows (slot s's I rows: expert ids[s]'s). h [1, 1, 1, k
    * I] bf16 TILE interleaved (gate_up's). loaders: 1 or 2 (default LOADERS), the same y bit for bit."""
    assert 1 <= k <= 16, k
    device = h.device()
    split = _loaders(loaders, LOADERS)
    sa = split_arg(split, SPLIT_SHARE)  # the chunks' pattern between the loaders (kernels/split.h)
    H, I, width = P["H"], P["I"], P["dn_width"]
    it = I // TILE
    kw = _fit_kw(it, kw)
    ncmax = max(c for core in P["dn_core"] for _, c in core)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    work = _work(device, P)
    key = ("dn", id(P["dn"]), P["dn"].buffer_address(), bool(P.get("kblock", False)), k, ids32, kw, fidelity)
    key += (str(h.memory_config()), str(mc), split, sa, ids.buffer_address())
    prog = _PROGRAMS.get(key)
    if prog is None:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
        bf = ttnn.bfloat16
        specs = [
            (0, bf, 2048, 3 * kw),  # h, streamed per chunk
            (1, ttnn.bfloat4_b, TB, 3 * kw * ncmax),
            (9, bf, 2048, 1),  # scratch: the ids
            (16, bf, 2048, ncmax),
        ]
        if split:
            specs.append((4, ttnn.bfloat4_b, TB, 3 * kw * ncmax))  # the writer's chunks
            specs.append((10, bf, 2048, 1))  # the writer's scratch: the ids
        cbs = _cbs(device, cores, specs, "down")
        reader_ct = [k, it, kw, ncmax, int(ids32)]
        if split:
            reader_ct += [sa]
        for t in (h, ids):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = [ncmax]
        if split:
            writer_ct += [sa, k, it, kw, int(ids32)]
        for t in (y, ids) if split else (y,):
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        # full dest sync (8 fp32 tiles) only for more than 4 columns a core; up to 4 the half-sync program
        full = ncmax > 4
        compute_ct = [k * it, kw, ncmax, 0, sa, int(full), int(P.get("kblock", False))]
        vcs = read_vcs([(c, (b,)) for c, b, _ in work])
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (c, b, j), vc in zip(work, vcs):
            c0 = P["dn_cols"][b][0]
            j0, jc = P["dn_core"][b][j]
            coff, estride = it * j0 * TB, it * width * TB  # the core's block, an expert's
            rd[c.x][c.y] = [h.buffer_address(), P["dn"].buffer_address(), ids.buffer_address(), b, coff, jc, estride, vc]
            wa = [y.buffer_address(), c0 + j0, jc]
            if split:
                wa += [P["dn"].buffer_address(), ids.buffer_address(), b, coff, estride, vc]
            wr[c.x][c.y] = wa
        prog = ttnn.ProgramDescriptor(
            kernels=[
                _kernel(
                    cores, "moe_dn2_reader.cpp" if split else "moe_dn_reader.cpp", reader_ct, rd,
                    ttnn.ReaderConfigDescriptor(),
                ),
                _kernel(
                    cores, "moe_dn2_writer.cpp" if split else "moe_dn_writer.cpp", writer_ct, wr,
                    ttnn.WriterConfigDescriptor(),
                ),
                _kernel(
                    cores,
                    "moe_dn_compute.cpp",
                    compute_ct,
                    ttnn.RuntimeArgs(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=FIDELITY[fidelity],
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=full,
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:
        rd, wr = prog.kernels[0].runtime_args, prog.kernels[1].runtime_args
        for c, _, _ in work:
            rd[c.x][c.y][0] = h.buffer_address()
            wr[c.x][c.y][0] = y.buffer_address()
    ttnn.generic_op([h, P["dn"], ids, y], prog)
    return y


# --- R rows at once (prefill, batched decode) -------------------------------------------------------------------------


def _rows_args(x, ids, w, rows, k, ids32):
    assert 1 <= rows <= TILE and 1 <= k <= 16, (rows, k)
    for t, what in ((ids, "ids"), (w, "w")):
        assert t.layout == ttnn.TILE_LAYOUT, f"moe8 rows: {what} must be a TILE tensor"
        assert t.padded_shape[-2] == TILE and t.padded_shape[-1] == TILE, (what, t.padded_shape)
    want = ttnn.uint32 if ids32 else ttnn.uint16
    assert ids.dtype == want, f"moe8 rows: ids are {ids.dtype}, ids32={ids32} wants {want}"
    assert x.padded_shape[-2] == TILE, x.padded_shape


def u_max(P, rows, k):
    """The most experts the union of rows x k ids can hold: h's width is u_max(...) * I."""
    return min(P["E"], rows * k)


def gate_up_rows(
    x, P, ids, w, k, rows, *, ids32=False, kw=8, fidelity="hifi2", memory_config=None, cpb=None, act="silu", loaders=None
):
    """h [1, 1, 32, u_max * I] bf16 TILE for x's first `rows` rows (x [1, 1, rows, H], one tile of rows): union index
    q's I columns hold row r's routing weight for expert union(ids)[q] times act(x_r G) * (x_r U) (0 where row r did
    not choose it). ids [1, 1, rows, k] uint16 (uint32 with ids32) and w [1, 1, rows, k] bf16, TILE. Columns past the
    union are not written. act: as gate_up's. cpb: cores per bank (default GU_ROWS_CPB; 0: the pack's). loaders: 1 or
    2 (default ROWS_LOADERS), the same h bit for bit."""
    assert act in ACTS, f"moe8: act {act!r}"
    _rows_args(x, ids, w, rows, k, ids32)
    split = _loaders(loaders, ROWS_LOADERS)
    sa = split_arg(split, SPLIT_SHARE)  # the chunks' pattern between the loaders (kernels/split.h)
    device = x.device()
    E, H, I, banks = P["E"], P["H"], P["I"], P["banks"]
    assert E <= 1024, "moe8 rows: at most 1024 experts"
    cpb = _gu_cpb(device, P, cpb, GU_ROWS_CPB)
    kt, it = H // TILE, I // TILE
    kw = _fit_kw(kt, kw)
    G, ncmax = gu_groups(it, banks)
    bias = P.get("gu_bias", False)
    um = u_max(P, rows, k)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    h = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE, um * I]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    work = _work(device, P, cpb)
    key = ("gur", id(P["gu"]), P["gu"].buffer_address(), bool(P.get("kblock", False)), k, rows, ids32, kw, fidelity)
    key += (cpb, act, split, sa, str(x.memory_config()), str(mc), ids.buffer_address(), w.buffer_address())
    prog = _PROGRAMS.get(key)
    if prog is None:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
        bf = ttnn.bfloat16
        # the reader's scratch: ids and w tiles, the table row, the union's bits, list and index, and the coefficients
        # C [u_max][32] (moe_gur_reader.cpp's offsets)
        off_c = -(-(6208 + 128 + 4 * E) // 16) * 16
        scr = -(-(off_c + 64 * um) // 64) * 64
        # the writer's: the ids tile, the union's bits and list (moe_gur_writer.cpp); with split also the table row of
        # the unit it loads (moe_gur2_writer.cpp's OFF_T)
        wscr = -(-(4096 + 128 + 2 * E) // 64) * 64 + (64 if split else 0)
        specs = [
            (0, bf, 2048, kt),  # x, all of it
            (1, ttnn.bfloat4_b, TB, 3 * kw * ncmax),  # weight chunks
            (2, bf, 2048, 2),  # the coefficient tile, per unit
            (3, bf, 2048, 1),  # the unit count (word 0), for the compute
            (9, bf, scr, 1),  # the reader's scratch
            (10, bf, wscr, 1),  # the writer's
            # out: two units' products, also with split (moe_gur2_writer.cpp writes unit n after unit n + 1's loads)
            (16, bf, 2048, ncmax),
        ]
        if split:
            specs.append((4, ttnn.bfloat4_b, TB, 3 * kw * ncmax))  # the writer's chunks
        if bias:
            specs.append((5, bf, 2048, 2 * ncmax))  # a unit's bias tiles (every row)
            specs.append((6, bf, 64 * ncmax, 1))  # the reader's bias staging
        cbs = _cbs(device, cores, specs, "gate_up_rows")
        reader_ct = [kt, kw, ncmax, k, cpb, banks, it, int(ids32), rows, E, G, int(bias), sa]
        for t in (x, ids, w, P["gu_table"]):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = [ncmax, k, cpb, banks, it, int(ids32), rows, E, G]
        if split:
            writer_ct += [kt, kw, sa]
        for t in (h, ids, P["gu_table"]) if split else (h, ids):
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        # whole tiles (ITER 0), the unit count from the reader (DYN)
        compute_ct = [kt, kw, ncmax, 0, 1, sa, ACTS[act], int(bias), int(P.get("kblock", False))]
        vcs = read_vcs([(c, (b,)) for c, b, _ in work])
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (c, b, j), vc in zip(work, vcs):
            rd[c.x][c.y] = [
                x.buffer_address(),
                P["gu"].buffer_address(),
                P["gu_table"].buffer_address(),
                ids.buffer_address(),
                w.buffer_address(),
                b,
                j,
                vc,
            ]
            wa = [h.buffer_address(), ids.buffer_address(), b, j]
            if split:
                wa += [P["gu"].buffer_address(), P["gu_table"].buffer_address(), vc]
            wr[c.x][c.y] = wa
            cp[c.x][c.y] = [0]
        writer = "moe_gur2_writer.cpp" if split else "moe_gur_writer.cpp"
        prog = ttnn.ProgramDescriptor(
            kernels=[
                _kernel(cores, "moe_gur_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
                _kernel(cores, writer, writer_ct, wr, ttnn.WriterConfigDescriptor()),
                _kernel(
                    cores,
                    "moe_gu_compute.cpp",
                    compute_ct,
                    cp,
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=FIDELITY[fidelity],
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:
        rd, wr = prog.kernels[0].runtime_args, prog.kernels[1].runtime_args
        for c, _, _ in work:
            rd[c.x][c.y][0] = x.buffer_address()
            wr[c.x][c.y][0] = h.buffer_address()
    ttnn.generic_op([x, P["gu"], P["gu_table"], ids, w, h], prog)
    return h


def down_rows(h, P, ids, k, rows, *, ids32=False, kw=8, fidelity="hifi2", memory_config=None, loaders=None):
    """y [1, 1, 32, H] bf16 TILE: row r = sum over the union of the R x k ids of h's row r (union index q's I columns)
    times expert union(ids)[q]'s down rows. h is gate_up_rows' (the same ids, rows and k); rows R .. 31 of y are 0.
    loaders: 1 or 2 (default ROWS_LOADERS), the same y bit for bit."""
    assert 1 <= rows <= TILE and 1 <= k <= 16, (rows, k)
    split = _loaders(loaders, ROWS_LOADERS)
    sa = split_arg(split, SPLIT_SHARE)  # the chunks' pattern between the loaders (kernels/split.h)
    device = h.device()
    E, H, I, width = P["E"], P["H"], P["I"], P["dn_width"]
    assert E <= 1024, "moe8 rows: at most 1024 experts"
    it = I // TILE
    kw = _fit_kw(it, kw)
    assert h.padded_shape[-1] == u_max(P, rows, k) * I, (h.padded_shape, rows, k)
    ncmax = max(c for core in P["dn_core"] for _, c in core)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    work = _work(device, P)
    key = ("dnr", id(P["dn"]), P["dn"].buffer_address(), bool(P.get("kblock", False)), k, rows, ids32, kw, fidelity)
    key += (sa, str(h.memory_config()), str(mc), ids.buffer_address(), split)
    prog = _PROGRAMS.get(key)
    if prog is None:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
        bf = ttnn.bfloat16
        scr = -(-(4096 + 128 + 2 * E) // 64) * 64  # ids, the union (moe_dnr_reader.cpp's offsets)
        specs = [
            (0, bf, 2048, 3 * kw),  # h, streamed per chunk
            (1, ttnn.bfloat4_b, TB, 3 * kw * ncmax),
            (3, bf, 2048, 1),  # the chunk count (word 0), for the compute
            (9, bf, scr, 1),  # ids, the union
            (16, bf, 2048, ncmax),
        ]
        if split:
            specs.append((4, ttnn.bfloat4_b, TB, 3 * kw * ncmax))  # the writer's chunks
            specs.append((10, bf, scr, 1))  # the writer's: ids, the union
        cbs = _cbs(device, cores, specs, "down_rows")
        reader_ct = [k, it, kw, ncmax, int(ids32), rows, E, sa]
        for t in (h, ids):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if split:
            writer_ct = [ncmax, k, it, kw, int(ids32), rows, E, sa]
            for t in (y, ids):
                writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        else:
            writer_ct = [ncmax, *ttnn.TensorAccessorArgs(y).get_compile_time_args()]
        full = ncmax > 4  # (down's: full dest sync only past 4 columns a core)
        compute_ct = [it, kw, ncmax, 1, sa, int(full), int(P.get("kblock", False))]  # the chunk count from the reader
        vcs = read_vcs([(c, (b,)) for c, b, _ in work])
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (c, b, j), vc in zip(work, vcs):
            c0 = P["dn_cols"][b][0]
            j0, jc = P["dn_core"][b][j]
            rd[c.x][c.y] = [
                h.buffer_address(),
                P["dn"].buffer_address(),
                ids.buffer_address(),
                b,
                it * j0 * TB,
                jc,
                it * width * TB,
                vc,
            ]
            wa = [y.buffer_address(), c0 + j0, jc]
            if split:
                wa += [P["dn"].buffer_address(), ids.buffer_address(), b, it * j0 * TB, it * width * TB, vc]
            wr[c.x][c.y] = wa
        writer = "moe_dnr2_writer.cpp" if split else "moe_dn_writer.cpp"
        prog = ttnn.ProgramDescriptor(
            kernels=[
                _kernel(cores, "moe_dnr_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
                _kernel(cores, writer, writer_ct, wr, ttnn.WriterConfigDescriptor()),
                _kernel(
                    cores,
                    "moe_dn_compute.cpp",
                    compute_ct,
                    ttnn.RuntimeArgs(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=FIDELITY[fidelity],
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=full,
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:
        rd, wr = prog.kernels[0].runtime_args, prog.kernels[1].runtime_args
        for c, _, _ in work:
            rd[c.x][c.y][0] = h.buffer_address()
            wr[c.x][c.y][0] = y.buffer_address()
    ttnn.generic_op([h, P["dn"], ids, y], prog)
    return y


# --- the router ---------------------------------------------------------------------------------------------------------


def route(
    logits,
    ids,
    w,
    k,
    *,
    kind="softmax",
    bias=None,
    ids32=False,
    experts=None,
    shared=None,
    scale=1.0,
    norm=True,
    groups=None,
    slot=None,
):
    """The router's top k and their weights into moe8's buffers, on one core (moe_route.cpp), in place of ttnn.topk and
    the weights' ops. logits [1, 1, R, E] float32 TILE (the router's linear with dtype float32), R <= 32, E a multiple of
    32; ids [1, 1, R, k] uint16 (uint32 with ids32) and w [1, 1, R, k] bf16, TILE, each written whole (zero past the
    picks). Slots in decreasing order of the pick's key; among equal keys the lower id.
    - kind "softmax" (the softmax over the experts, top k, renormalized): the top k by logit, compared in fp32; the
      weights the softmax over the k.
    - kind "sigmoid": the top k by sigmoid(logit) + bias (bias [1, 1, 1, E] float32 TILE, or None: by logit, exactly as
      the softmax kind picks); the weights the picks' sigmoids over their sum.
    experts: E, the experts' columns 0 .. E - 1 of logits (default all; the router's matmul may carry more columns).
    shared: a [1, 1, R, 1] bf16 TILE that gets sigmoid(logits[:, E]) (a shared expert's gate folded into the router's
    matmul as column E). scale: every weight times it (0 < scale < 4: a routed scaling factor). norm=False (the sigmoid
    kind): each weight is the pick's own sigmoid, not divided by their sum. groups=(n_group, topk_group) (the sigmoid
    kind with a bias; DeepSeek-V3's group-limited routing): the experts in n_group groups of E / n_group in order, a
    group's score the sum of its two largest keys (score + bias), the topk_group best groups kept (the lower group among
    equal scores), the top k of their experts by key. slot: an id written as slot k of every row at weight 1.0 (a shared
    expert without a gate as moe8's slot k + 1; ids and w then hold k + 1 slots, and moe8 runs with k + 1). Fixed point
    on the core's RISC-V, then bf16. Returns w."""
    assert kind in ("softmax", "sigmoid"), f"moe8.route: kind {kind!r}"
    R, W = logits.shape[-2], logits.shape[-1]
    E = experts or W
    assert E <= W and (shared is None or E + 1 <= W), (E, W)
    assert logits.dtype == ttnn.float32 and logits.layout == ttnn.TILE_LAYOUT, logits.dtype
    assert R <= 32 and E % TILE == 0 and 1 <= k <= 16, (R, E, k)
    assert 0 < scale < 4, scale
    assert norm or kind == "sigmoid", "moe8.route: norm=False is for the sigmoid kind"
    assert slot is None or (k + 1 <= 16 and 0 <= int(slot) < (1 << 32 if ids32 else 1 << 16)), (k, slot)
    # (no groups: 1, 1, so the kernel's discarded GROUPS branch still compiles: E / NG, arrays of NG)
    ng, tg = (1, 1) if groups is None else (int(groups[0]), int(groups[1]))
    if groups is not None:
        assert kind == "sigmoid" and bias is not None, "moe8.route: groups need the sigmoid kind with a bias"
        assert ng <= 32 and E % ng == 0 and E // ng >= 2 and 1 <= tg <= ng and tg * (E // ng) >= k, (E, groups, k)
    flags = int(bool(norm)) | (2 if slot is not None else 0) | (4 if groups is not None else 0)
    scale_q30 = round(scale * (1 << 30))
    assert ids.dtype == (ttnn.uint32 if ids32 else ttnn.uint16) and ids.layout == ttnn.TILE_LAYOUT
    assert w.dtype == ttnn.bfloat16 and w.layout == ttnn.TILE_LAYOUT
    # the kernel's kinds: 0 softmax, 1 sigmoid with a bias, 2 sigmoid without (the pick by logit)
    sig = kind == "sigmoid" and bias is not None
    kcode = 0 if kind == "softmax" else (1 if sig else 2)
    if sig:
        assert bias.dtype == ttnn.float32 and bias.layout == ttnn.TILE_LAYOUT and bias.shape[-1] == E
    if shared is not None:
        assert shared.dtype == ttnn.bfloat16 and shared.layout == ttnn.TILE_LAYOUT, shared.dtype
    b = bias if sig else logits
    gt = shared if shared is not None else w
    key = (
        "route",
        R,
        E,
        k,
        ids32,
        kind,
        sig,
        str(logits.memory_config()),
        str(ids.memory_config()),
        str(w.memory_config()),
        str(b.memory_config()),
        shared is not None,
        str(gt.memory_config()),
        flags,
        scale_q30,
        slot,
        ng,
        tg,
    )
    addrs = [logits.buffer_address(), ids.buffer_address(), w.buffer_address(), b.buffer_address(), gt.buffer_address()]
    prog = _PROGRAMS.get(key)
    core = ttnn.CoreCoord(0, 0)
    if prog is None:
        device = logits.device()
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        bf = ttnn.bfloat16
        specs = [
            # the logits' rows and the bias (fp32 bits), then the sigmoid kind's scratch: the bias and a row's x in
            # Q32.32, the upper bounds
            (0, bf, (R + 1) * E * 4 + (8 + 8 + 4) * E + R * 64, 1),
            (1, bf, 1024 * (4 if ids32 else 2), 1),  # the ids' page
            (2, bf, 2048, 1),  # the weights' page
            (3, bf, 2048, 1),  # the shared expert's gate page
        ]
        cbs = _cbs(device, cores, specs, "route")
        ct = [E, k, R, int(ids32), kcode, int(shared is not None), flags, scale_q30, int(slot or 0), ng, tg]
        for t in (logits, ids, w, b, gt):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        rt[core.x][core.y] = list(addrs)
        prog = ttnn.ProgramDescriptor(
            kernels=[_kernel(cores, "moe_route.cpp", ct, rt, ttnn.ReaderConfigDescriptor())],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:
        rt = prog.kernels[0].runtime_args
        for i, v in enumerate(addrs):
            rt[core.x][core.y][i] = v
    io = [logits, bias] if sig else [logits]
    io += [ids, shared, w] if shared is not None else [ids, w]
    ttnn.generic_op(io, prog)
    return w


def route_reference(logits, k, kind="softmax", bias=None, scale=1.0, norm=True, groups=None, slot=None):
    """moe8.route's result on the host: (ids [R, k] long, w [R, k] float64) for float32 logits [R, E] (float64 math):
    softmax, the top k by logit and the softmax over the k; sigmoid, the top k by sigmoid + bias (bias [E]; without a
    bias by logit) and the picks' sigmoids over their sum (norm=False: not divided). The lower id first among equal
    keys. groups=(n_group, topk_group): a group's score the sum of its two largest keys, the topk_group best groups
    kept (the lower group first among equal scores), the rest masked to -inf. Weights times scale. slot: an extra
    column of that id at weight 1.0 (ids and w [R, k + 1])."""
    lg = logits.float().double()
    if kind == "softmax":
        order = torch.argsort(-lg, dim=-1, stable=True)[:, :k]
        w = torch.softmax(lg.gather(1, order), dim=-1) * scale
    else:
        s = torch.sigmoid(lg)
        # without a bias the pick is by logit: sigmoid is increasing, and float64's saturates (1.0 exactly above
        # about 37), which would tie logits that differ
        pick = lg if bias is None else s + bias.float().double().reshape(1, -1)
        if groups is not None:
            ng, tg = groups
            R, E = pick.shape
            gsc = pick.reshape(R, ng, E // ng).topk(2, dim=-1).values.sum(-1)
            kept = torch.argsort(-gsc, dim=-1, stable=True)[:, :tg]
            mask = torch.zeros(R, ng, dtype=torch.bool).scatter_(1, kept, True)
            mask = mask.repeat_interleave(E // ng, dim=1)
            pick = pick.masked_fill(~mask, float("-inf"))
        order = torch.argsort(-pick, dim=-1, stable=True)[:, :k]
        v = s.gather(1, order)
        w = (v / v.sum(-1, keepdim=True) if norm else v) * scale
    if slot is not None:
        order = torch.cat([order, torch.full_like(order[:, :1], int(slot))], dim=1)
        w = torch.cat([w, torch.ones_like(w[:, :1])], dim=1)
    return order, w
