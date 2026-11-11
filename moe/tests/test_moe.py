# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Correctness of the MoE ops (moe/moe8.py) against torch, on a Tensix device or ttsim.

  python -s moe/tests/test_moe.py [test names]     # or, with pytest: pytest -s moe/tests

Experts are random matrices on the BFP4_b grid (so the packed weights hold them exactly), x has a few outlier channels.
- bfp4_tiles against ttnn's own BFP4_b host conversion, bit for bit; the route kernel's sigmoid table against
  sigmoid_table();
- gate_up and down (decode, one row) at HiFi4 against a float64 reference: relative Frobenius error below REL (1 %)
  and PCC above 0.9999 for h and y, for each activation; two loaders against one, other cores per bank, a split share
  of 0.6 and uint32 ids against the default, bit for bit;
- a gate/up bias, column groups (two banks) and the K-block layout (kblock) at HiFi4 against the reference;
- gate_up_rows and down_rows (R rows, each with its own experts) at HiFi4 against the reference, rows past R zero; two
  loaders against one, bit for bit;
- the same ops at HiFi2 (the ops' default fidelity): relative error below REL_HIFI2 (1.6 %; measured up to 1.40 %, on
  down's y with gelu_tanh). HiFi2 makes two of the matrix unit's four multiply passes over the operands' mantissas,
  so its products leave out low-order bits that HiFi4 keeps: a property of the fidelity, the same on any device;
- route: the picks equal the float64 reference's (the lower id among equal keys), each weight within 1 bf16 ulp of the
  reference or within 2^-29 (times the scale) of it, over random rows with ties and fixed edge rows;
- the pack cache: prepack, then pack_cached, gives the same outputs as pack bit for bit; a mismatched call is refused;
  repack_down equals a fresh pack at the new split.
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.testing import check, main, raises, same  # noqa: E402
from moe import moe8  # noqa: E402

E, H, I, K = 16, 1024, 512, 4
BF = torch.bfloat16
REL = 0.01  # the relative bar against the float64 reference at HiFi4 (see the docstring)
REL_HIFI2 = 0.016  # the same at HiFi2: its measured error (up to 1.40 %) with a margin
FID = "hifi4"  # the fidelity of the checks against the reference, unless a test says otherwise


def on_grid(w, kblock=False):
    """w's values rounded to BFP4_b as tt-metal rounds them (moe8.bfp4_tiles), as float32 [K, N]."""
    k, n = w.shape
    v = moe8.bfp4_values(moe8.bfp4_tiles(w, kblock=kblock), kblock=kblock)  # [KT, NT, 32, 32]
    return torch.from_numpy(v.transpose(0, 2, 1, 3).reshape(k, n).copy())


def experts(seed, gu_scale=0.02, kblock=False):
    g = torch.Generator().manual_seed(seed)
    gu = torch.stack([on_grid(torch.randn(H, 2 * I, generator=g) * gu_scale, kblock).to(BF) for _ in range(E)])
    dn = torch.stack([on_grid(torch.randn(I, H, generator=g) * 0.02, kblock).to(BF) for _ in range(E)])
    return gu, dn


def make_x(rows, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, H, generator=g)
    x[:, torch.randperm(H, generator=g)[:8]] *= 20
    return x.to(BF)


def dev(device, t, dtype, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mc)


def decode_inputs(device, seed, ids32=False):
    g = torch.Generator().manual_seed(100 + seed)
    ids = torch.randperm(E, generator=g)[:K]
    wts = torch.softmax(torch.randn(K, generator=g), 0).to(BF)
    x = make_x(1, seed)
    idt, dt = (torch.int32, ttnn.uint32) if ids32 else (torch.int16, ttnn.uint16)
    tid = dev(device, ids.to(idt).reshape(1, 1, 1, K), dt, ttnn.ROW_MAJOR_LAYOUT)
    twt = dev(device, wts.reshape(1, 1, 1, K), ttnn.bfloat16)
    tx = dev(device, x.reshape(1, 1, 1, H), ttnn.bfloat16, mc=ttnn.L1_MEMORY_CONFIG)
    return ids, wts, x, tid, twt, tx


def host(t, cols):
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])[:, :cols]


def test_conversion(device):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 512, generator=g) * 0.02
    w[3, 5], w[7, 9] = 0.0, -1e-30  # a zero and a denormal
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    same("bfp4_tiles against ttnn's BFP4_b conversion", on_grid(w), ttnn.to_torch(tw).float())
    hdr = (Path(moe8.KERNELS) / "moe_sigmoid_q30.h").read_text()
    table = [int(v) for v in re.findall(r"\d+", hdr[hdr.index("{") + 1 : hdr.index("}")])]
    same("the route kernel's sigmoid table against sigmoid_table()", torch.tensor(table), torch.tensor(moe8.sigmoid_table()))


def test_decode(device):
    decode_case(device, FID, REL)


def decode_case(device, fid, rel):
    for act, scale in (("silu", 0.02), ("gelu_tanh", 0.02), ("swiglu_oai", 0.1)):
        gu, dn = experts(1, scale)
        P = moe8.pack(device, gu, dn)
        ids, wts, x, tid, twt, tx = decode_inputs(device, 2)
        ry, rh = moe8.reference(x.float(), gu, dn, ids.tolist(), wts.float(), act)
        outs = {}
        for loaders in (1, 2):
            th = moe8.gate_up(tx, P, tid, twt, K, act=act, loaders=loaders, fidelity=fid)
            ty = moe8.down(th, P, tid, K, loaders=loaders, fidelity=fid)
            outs[loaders] = (host(th, K * I)[:1], host(ty, H)[:1])
            check(f"{act} gate_up h, {fid}, loaders {loaders}", outs[loaders][0], rh, rel=rel)
            check(f"{act} down y, {fid}, loaders {loaders}", outs[loaders][1], ry, rel=rel)
        same(f"{act} h: two loaders against one", outs[2][0], outs[1][0])
        same(f"{act} y: two loaders against one", outs[2][1], outs[1][1])
        if act == "silu" and fid == FID:
            cpb = 2 if P["cpb"] != 2 else 1
            th = moe8.gate_up(tx, P, tid, twt, K, act=act, cpb=cpb, fidelity=fid)
            same(f"h with {cpb} cores a bank against the pack's {P['cpb']}", host(th, K * I)[:1], outs[2][0])
            try:
                moe8.SPLIT_SHARE = 0.6
                th = moe8.gate_up(tx, P, tid, twt, K, act=act, fidelity=fid)
                ty = moe8.down(th, P, tid, K, fidelity=fid)
                same("h with a split share of 0.6", host(th, K * I)[:1], outs[2][0])
                same("y with a split share of 0.6", host(ty, H)[:1], outs[2][1])
            finally:
                moe8.SPLIT_SHARE = 0.5
            _, _, _, tid32, twt32, tx32 = decode_inputs(device, 2, ids32=True)
            th = moe8.gate_up(tx32, P, tid32, twt32, K, act=act, ids32=True, fidelity=fid)
            ty = moe8.down(th, P, tid32, K, ids32=True, fidelity=fid)
            same("h with uint32 ids", host(th, K * I)[:1], outs[2][0])
            same("y with uint32 ids", host(ty, H)[:1], outs[2][1])


def test_bias_groups_kblock(device):
    gu, dn = experts(3, 0.1, kblock=True)
    g = torch.Generator().manual_seed(4)
    gb = torch.randn(E, 2 * I, generator=g) * 0.5
    # two banks: 2 * ceil(16 / 2) = 16 tiles of an expert a bank, more than dest holds: two column groups
    P = moe8.pack(device, gu, dn, banks=2, gu_bias=gb, kblock=True)
    assert moe8.gu_groups(I // 32, 2)[0] == 2
    for act in ("swiglu_oai", "silu"):
        ids, wts, x, tid, twt, tx = decode_inputs(device, 5)
        ry, rh = moe8.reference(x.float(), gu, dn, ids.tolist(), wts.float(), act, gb)
        th = moe8.gate_up(tx, P, tid, twt, K, act=act, loaders=2, fidelity=FID)
        ty = moe8.down(th, P, tid, K, loaders=2, fidelity=FID)
        check(f"{act} with a bias, column groups and K blocks, {FID}: h", host(th, K * I)[:1], rh, rel=REL)
        check(f"{act} with a bias, column groups and K blocks, {FID}: y", host(ty, H)[:1], ry, rel=REL)
    raises("column groups with one loader", AssertionError, lambda: moe8.gate_up(tx, P, tid, twt, K, loaders=1))
    rows_case(device, gu, dn, P, 6, act="swiglu_oai", gb=gb, loaders=(1, 2), seed=6)


def rows_inputs(device, R, seed, ids32=False):
    g = torch.Generator().manual_seed(200 + seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(R)])
    wts = torch.softmax(torch.randn(R, K, generator=g), -1).to(BF)
    x = make_x(R, seed)
    idt, dt = (torch.int32, ttnn.uint32) if ids32 else (torch.int16, ttnn.uint16)
    tid = dev(device, ids.to(idt).reshape(1, 1, R, K), dt)
    twt = dev(device, wts.reshape(1, 1, R, K), ttnn.bfloat16)
    tx = dev(device, x.reshape(1, 1, R, H), ttnn.bfloat16)
    return ids, wts, x, tid, twt, tx


def rows_case(device, gu, dn, P, R, act="silu", gb=None, loaders=(1,), seed=0, ids32=False, fid=FID):
    ids, wts, x, tid, twt, tx = rows_inputs(device, R, seed, ids32)
    ref = moe8.reference_rows(x.float(), gu, dn, ids.tolist(), wts.float(), act, gb)
    u = moe8.union(ids, E)
    href = torch.zeros(R, len(u) * I, dtype=torch.float64)
    for r in range(R):
        _, hr = moe8.reference(x[r : r + 1].float(), gu, dn, ids[r].tolist(), wts[r].float(), act, gb)
        for s, e in enumerate(ids[r].tolist()):
            q = u.index(e)
            href[r, q * I : (q + 1) * I] = hr[0, s * I : (s + 1) * I]
    outs = []
    for ld in loaders:
        th = moe8.gate_up_rows(tx, P, tid, twt, K, R, act=act, loaders=ld, ids32=ids32, fidelity=fid)
        ty = moe8.down_rows(th, P, tid, K, R, loaders=ld, ids32=ids32, fidelity=fid)
        hd = host(th, len(u) * I)[:R]
        yall = host(ty, H)
        name = f"rows {R} ({act}{', bias' if gb is not None else ''}{', uint32 ids' if ids32 else ''}), {fid}, loaders {ld}"
        bar = REL if fid == "hifi4" else REL_HIFI2
        check(f"{name}: h", hd, href, rel=bar)
        check(f"{name}: y", yall[:R], ref, rel=bar)
        same(f"{name}: rows past R zero", yall[R:32], torch.zeros(32 - R, H))
        outs.append((hd, yall))
    for (h1, y1), (h2, y2), ld in zip(outs, outs[1:], loaders[1:]):
        same(f"rows {R}: h with {ld} loaders against {loaders[0]}", h2, h1)
        same(f"rows {R}: y with {ld} loaders against {loaders[0]}", y2, y1)
    return outs


def test_rows(device):
    gu, dn = experts(7)
    P = moe8.pack(device, gu, dn)
    rows_case(device, gu, dn, P, 1, seed=1)
    rows_case(device, gu, dn, P, 5, loaders=(1, 2), seed=2)
    rows_case(device, gu, dn, P, 32, act="gelu_tanh", seed=3)
    rows_case(device, gu, dn, P, 5, seed=2, ids32=True)


def test_hifi2(device):
    """The ops at HiFi2, their default fidelity: REL_HIFI2, HiFi2's measured accuracy (see the module docstring)."""
    decode_case(device, "hifi2", REL_HIFI2)
    gu, dn = experts(7)
    P = moe8.pack(device, gu, dn)
    rows_case(device, gu, dn, P, 5, loaders=(1, 2), seed=2, fid="hifi2")
    rows_case(device, gu, dn, P, 32, act="gelu_tanh", seed=3, fid="hifi2")


def ulps(a, b):
    """|a - b| in bf16 ulps of b (bf16 tensors)."""
    return (a.view(torch.int16).to(torch.int32) - b.view(torch.int16).to(torch.int32)).abs()


def edge_rows(e, k, g):
    """32 rows of logits [32, e] that a range or sign error shows in: all equal, near 0, tiny, large, huge, ties at the
    k-th place, two values, signs mixed, the rest random."""
    rows = [
        torch.zeros(e),
        torch.full((e,), 1.5),
        torch.full((e,), -2.0),
        torch.randn(e, generator=g) * 1e-3,
        torch.randn(e, generator=g) * 1e-6,
        torch.randn(e, generator=g) * 40,
        torch.randn(e, generator=g) * 1e3,
    ]
    t = torch.randn(e, generator=g)
    t[: 2 * k] = 2.5  # 2k equal at the top: the pick is the lowest k ids of them
    rows.append(t)
    t = torch.randn(e, generator=g)
    t[torch.randperm(e, generator=g)[: k + 3]] = 1.25  # ties across the k-th place
    rows.append(t)
    rows.append(torch.where(torch.rand(e, generator=g) < 0.5, torch.tensor(-1.0), torch.tensor(1.0)))
    rows.append(torch.randn(e, generator=g).abs())
    rows.append(-torch.randn(e, generator=g).abs())
    while len(rows) < 32:
        rows.append(torch.randn(e, generator=g) * float(2 ** (len(rows) % 7 - 3)))
    return torch.stack(rows)


def route_case(device, g, name, kind, R, k, ids32=False, bias=False, edge=False, **opts):
    e = 128
    lg = edge_rows(e, k, g) if edge else torch.randn(R, e, generator=g) * 3
    if not edge:
        lg[::2] = lg[::2].to(BF).float()  # ties in the even rows
    R = lg.shape[0]
    tl = dev(device, lg.reshape(1, 1, R, e), ttnn.float32)
    kk = k + (opts.get("slot") is not None)
    idt, tdt = (torch.int32, ttnn.uint32) if ids32 else (torch.int16, ttnn.uint16)
    junk = torch.full((1, 1, R, kk), 7)
    ti = dev(device, junk.to(idt), tdt)
    tw = dev(device, junk.to(BF), ttnn.bfloat16)
    b, tb = None, None
    if bias:
        b = torch.randn(e, generator=g) * 0.1
        tb = dev(device, b.reshape(1, 1, 1, e), ttnn.float32)
    moe8.route(tl, ti, tw, k, kind=kind, bias=tb, ids32=ids32, **opts)
    gi = ttnn.to_torch(ti).reshape(R, kk).long() & (0xFFFFFFFF if ids32 else 0xFFFF)
    gw = ttnn.to_torch(tw).reshape(R, kk).to(BF)
    ri, rw = moe8.route_reference(lg, k, kind, b, **opts)
    same(f"route {name}: picks", gi, ri)
    tol = 2.0**-29 * max(1.0, opts.get("scale", 1.0))
    bad = (ulps(gw, rw.to(BF)) > 1) & ((gw.double() - rw).abs() > tol)
    ok = int(bad.sum()) == 0
    print(f"{'ok  ' if ok else 'FAIL'} route {name}: weights within 1 bf16 ulp or {tol:.0e}: {int(bad.sum())} off")
    assert ok, name


def test_route(device):
    g = torch.Generator().manual_seed(3)
    route_case(device, g, "softmax R 1 k 8", "softmax", 1, 8)
    route_case(device, g, "softmax R 4 k 6 uint32", "softmax", 4, 6, ids32=True)
    route_case(device, g, "softmax edge rows k 8", "softmax", 32, 8, edge=True)
    route_case(device, g, "sigmoid + bias R 4 k 8", "sigmoid", 4, 8, bias=True)
    route_case(device, g, "sigmoid + bias edge rows k 6 uint32", "sigmoid", 32, 6, ids32=True, bias=True, edge=True)
    route_case(device, g, "sigmoid, no bias, R 4 k 8", "sigmoid", 4, 8)
    route_case(device, g, "sigmoid, no bias, not normalized, edge rows", "sigmoid", 32, 8, edge=True, norm=False)
    route_case(
        device, g, "sigmoid + bias, groups 8/4, scale 2.5, slot", "sigmoid", 8, 8, bias=True,
        groups=(8, 4), scale=2.5, slot=128,
    )


def test_pack_cache(device):
    gu, dn = experts(9)
    banks = moe8.dram_banks(device)
    tmp = Path(tempfile.mkdtemp(prefix="moe8-cache-"))
    try:
        cache = tmp / "layer0"
        cpb = moe8._clamp_cpb(device, banks, 3)  # (pack's cores a bank on this grid)
        assert moe8.prepack(gu, dn, cache, banks, cpb) is True
        assert moe8.prepack(gu, dn, cache, banks, cpb) is False  # already there
        ids, wts, x, tid, twt, tx = decode_inputs(device, 8)
        P0 = moe8.pack(device, gu, dn)
        P1 = moe8.pack_cached(device, cache)
        h0, h1 = moe8.gate_up(tx, P0, tid, twt, K), moe8.gate_up(tx, P1, tid, twt, K)
        same("pack_cached against pack: h", host(h1, K * I), host(h0, K * I))
        same("pack_cached against pack: y", host(moe8.down(h1, P1, tid, K), H), host(moe8.down(h0, P0, tid, K), H))
        other = experts(10)
        raises("a cache of other weights", moe8.PackCacheMismatch, lambda: moe8.prepack(*other, cache, banks, cpb))
        raises("a cache of another split", moe8.PackCacheMismatch, lambda: moe8.pack_cached(device, cache, cores_per_bank=1))
        assert moe8.repack_down(cache, cores_per_bank=1) is True
        P2 = moe8.pack_cached(device, cache, cores_per_bank=1)
        P3 = moe8.pack(device, gu, dn, cores_per_bank=1)
        same("repack_down against a fresh pack: y", host(moe8.down(h0, P2, tid, K), H), host(moe8.down(h0, P3, tid, K), H))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(globals()))
