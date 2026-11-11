# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Correctness of the dense reader (dense_reader/linear.py) against torch, on a Tensix device or ttsim.

  python -s dense_reader/tests/test_dense_reader.py [test names]     # or, with pytest: pytest -s dense_reader/tests

Reference: x rounded to bf16 times the weight's own values (read back from the device: the matrix on the bfloat8_b,
bfloat4_b or bfloat16 grid), in float64. Bars: relative Frobenius error below 1 % and PCC above 0.9999 for bf16 outputs
(the output's bf16 rounding and the fp32 sum order are the only differences), 0.1 % for the fp32 output. Where two
configurations of the op must agree (split shares, a block list against one call per block, the top-k records against
the op's own output), the check is bit for bit.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import device as cdev  # noqa: E402
from common.testing import check, main, raises, same  # noqa: E402
from dense_reader import linear  # noqa: E402

DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}


def x_input(device, m, k, seed, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    x[:, :4] *= 8  # a few outlier channels
    tx = ttnn.from_torch(
        x.reshape(1, 1, m, k).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return x, tx


def matrix(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g) * 0.02


def held(tw, n):
    """[K, N] values the device weight tw holds (read back), its first n columns."""
    return ttnn.to_torch(tw).float().reshape(tw.shape[-2], -1)[:, :n]


def host(y, m, n):
    return ttnn.to_torch(y).float().reshape(-1, y.shape[-1])[:m, :n]


def ref(x, wk, scale=None):
    y = x.to(torch.bfloat16).double() @ wk.double()
    return (y * scale.double()[None, :] if scale is not None else y).float()


def test_formats(device):
    k, n = 1024, 2048
    wt = matrix(n, k, 2)
    for fmt, dtype in DTYPES.items():
        tw, _ = linear.native_weight(device, wt, dtype)
        wk = held(tw, n)
        for m in (1, 7, 32):
            x, tx = x_input(device, m, k, 1 + m)
            check(f"{fmt} rows {m} K {k} N {n}", host(linear.matmul(tx, tw, None), m, n), ref(x, wk))


def test_scales_interleaved_prefill(device):
    k, n = 1024, 1536
    wt = matrix(n, k, 3)
    wt[: n // 2] *= 40  # columns of very different size
    tw, ts = linear.native_weight(device, wt, ttnn.bfloat8_b, col_scale=True)
    c = ttnn.to_torch(ts).float().reshape(-1)[:n]
    wk = held(tw, n)
    for m in (1, 100, 200):  # decode; prefill in one and in two row slices
        x, tx = x_input(device, m, k, 4 + m)
        check(f"bfp8 column scales rows {m}", host(linear.matmul(tx, tw, ts), m, n), ref(x, wk, c))
    ti, _ = linear.native_weight(device, wt, ttnn.bfloat8_b, sharded=False)
    x, tx = x_input(device, 3, k, 5)
    check("bfp8 interleaved weight rows 3", host(linear.matmul(tx, ti, None), 3, n), ref(x, held(ti, n)))


def test_split_shares_bitwise(device):
    k, n, m = 2048, 2048, 3
    x, tx = x_input(device, m, k, 6, ttnn.L1_MEMORY_CONFIG)
    tw, _ = linear.native_weight(device, matrix(n, k, 7), ttnn.bfloat4_b)
    base = host(linear.matmul(tx, tw, None), m, n)
    try:
        for share in (0.6, 2 / 3, 0.75):
            linear.SPLIT_SHARE = share
            same(f"split share {share:.3f} against 0.5", host(linear.matmul(tx, tw, None), m, n), base)
        linear.SPLIT_SHARE = 0.5
        same("one loader against two", host(linear.matmul(tx, tw, None, split=False), m, n), base)
        linear.REQ_TILES = 3
        same("requests of 3 tiles against one a row", host(linear.matmul(tx, tw, None), m, n), base)
    finally:
        linear.SPLIT_SHARE, linear.REQ_TILES = 0.5, 0


def test_glu(device):
    k, n = 1024, 1024
    gate, up = matrix(n, k, 8), matrix(n, k, 9)
    tw, _ = linear.native_weight(device, linear.interleave_glu(gate, up), ttnn.bfloat8_b)
    tg, _ = linear.native_weight(device, gate, ttnn.bfloat8_b)
    tu, _ = linear.native_weight(device, up, ttnn.bfloat8_b)
    gk, uk = held(tg, n), held(tu, n)
    for m in (3, 20):  # the SFPU stage on the valid rows only; on whole tiles
        x, tx = x_input(device, m, k, 10 + m)
        g, u = ref(x, gk).double(), ref(x, uk).double()
        swi = (torch.nn.functional.silu(g) * u).float()
        gel = (torch.nn.functional.gelu(g, approximate="tanh") * u).float()
        check(f"swiglu interleaved rows {m}", host(linear.linear(tx, tw, None, activation="swiglu"), m, n), swi)
        check(f"swiglu two weights rows {m}", host(linear.linear(tx, (tg, tu), None, activation="swiglu"), m, n), swi)
        check(f"geglu interleaved rows {m}", host(linear.linear(tx, tw, None, activation="geglu"), m, n), gel)
        check(f"geglu two weights rows {m}", host(linear.linear(tx, (tg, tu), None, activation="geglu"), m, n), gel)


def test_residual_out32(device):
    k, n, m = 1024, 2048, 5
    x, tx = x_input(device, m, k, 12)
    wt = matrix(n, k, 13)
    tw, _ = linear.native_weight(device, wt, ttnn.bfloat8_b)
    wk = held(tw, n)
    res = torch.randn(32, n, generator=torch.Generator().manual_seed(27))
    resb = res.to(torch.bfloat16)
    tr = ttnn.from_torch(resb.reshape(1, 1, 32, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    want = ref(x, wk) + resb[:m].float()
    for mode in ("epilogue", "preload"):
        y = linear.matmul(tx, tw, None, residual=tr, residual_mode=mode)
        check(f"residual {mode} rows {m}", host(y, m, n), want)
    y = linear.linear(tx, tw, None, out_dtype=ttnn.float32, fidelity="hifi4")
    check(f"fp32 output rows {m}", host(y, m, n), ref(x, wk), rel=1e-3, pcc=0.999999)


def test_topk_records(device):
    k, n, rows, kk = 1024, 2048, 3, 8
    x, tx = x_input(device, 32, k, 14, ttnn.L1_MEMORY_CONFIG)
    tw, _ = linear.native_weight(device, matrix(n, k, 15), ttnn.bfloat4_b)
    grid = device.compute_with_storage_grid_size()
    for n_valid, base in ((n, 0), (n - 37, 1000)):
        r = linear.topk_result(device, grid.x * grid.y, rows, kk)
        spec = {"r": r, "k": kk, "rows": rows, "slot0": 0, "base": base, "n": n_valid, "write_y": True}
        y = host(linear.matmul(tx, tw, None, topk=spec), rows, n)
        vals, idx = linear.topk_records(r, rows, kk)
        for t in range(rows):
            ok = idx[t] >= 0
            got = vals[t][ok]
            col = idx[t][ok] - base
            assert int(col.min()) >= 0 and int(col.max()) < n_valid, (col.min(), col.max())
            same(f"top-k records row {t} (n {n_valid}, base {base}): values at their ids", y[t][col], got)
            top = torch.sort(got, descending=True).values[:kk]
            same(f"top-k records row {t}: the best {kk} of the row", top, torch.topk(y[t][:n_valid], kk).values)
        ttnn.deallocate(r)


def test_block_list_bitwise(device):
    k, m = 1024, 3
    x, tx = x_input(device, m, k, 16, ttnn.L1_MEMORY_CONFIG)
    a, _ = linear.native_weight(device, matrix(1024, k, 17), ttnn.bfloat8_b)
    b, _ = linear.native_weight(device, matrix(1024, k, 18), ttnn.bfloat8_b)
    width = 2048 + 64  # a gap of two tiles between the blocks
    one = linear.matmul(tx, [(a, 0), (b, 34)], None)
    two = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, m, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    linear.matmul(tx, a, None, out=two, col0=0)
    linear.matmul(tx, b, None, out=two, col0=34)
    ho, ht = host(one, m, width), host(two, m, width)
    same("block list against one call per block", torch.cat([ho[:, :1024], ho[:, 1088:]], 1),
         torch.cat([ht[:, :1024], ht[:, 1088:]], 1))
    g1, _ = linear.native_weight(device, matrix(512, k, 19), ttnn.bfloat8_b)
    u1, _ = linear.native_weight(device, matrix(512, k, 20), ttnn.bfloat8_b)
    g2, _ = linear.native_weight(device, matrix(512, k, 21), ttnn.bfloat8_b)
    u2, _ = linear.native_weight(device, matrix(512, k, 22), ttnn.bfloat8_b)
    for act in ("swiglu", "geglu"):
        one = linear.linear(tx, [((g1, u1), 0), ((g2, u2), 16)], None, activation=act)
        p1 = linear.linear(tx, (g1, u1), None, activation=act)
        p2 = linear.linear(tx, (g2, u2), None, activation=act)
        same(f"{act} block list against one call per pair", host(one, m, 1024),
             torch.cat([host(p1, m, 512), host(p2, m, 512)], 1))


def test_max_banks(device):
    k, n, m = 1024, 1024, 2
    banks = cdev.dram_banks(device)
    fewer = max(1, banks // 2)
    x, tx = x_input(device, m, k, 23)
    wt = matrix(n, k, 24)
    tw, _ = linear.native_weight(device, wt, ttnn.bfloat8_b, max_banks=fewer)
    assert tw.memory_config().shard_spec.num_cores() == fewer
    check(f"weight over {fewer} of {banks} banks", host(linear.matmul(tx, tw, None), m, n),
          ref(x, held(tw, n)))
    full, _ = linear.native_weight(device, wt, ttnn.bfloat8_b)
    if banks > fewer:
        raises(f"a weight over {banks} banks with max_banks {fewer}", ValueError,
               lambda: linear.matmul(tx, full, None, max_banks=fewer))


def test_linear_rank3(device):
    k, n, m = 1024, 1024, 4
    x, tx = x_input(device, m, k, 25)
    wt = matrix(n, k, 26)
    tw, _ = linear.native_weight(device, wt, ttnn.bfloat8_b)
    x3 = ttnn.reshape(tx, [1, m, k])
    y = linear.linear(x3, tw, None, activation="silu")
    assert list(y.shape) == [1, m, n], y.shape
    want = torch.nn.functional.silu(ref(x, held(tw, n)).double()).float()
    check(f"linear rank 3 with silu rows {m}", ttnn.to_torch(y).float().reshape(m, n), want)


if __name__ == "__main__":
    sys.exit(main(globals()))
