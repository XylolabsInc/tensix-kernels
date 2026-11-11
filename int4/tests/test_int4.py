# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Correctness of the int4 group-scale matmul (int4/w4a16.py) against torch, on a Tensix device or ttsim.

  python -s int4/tests/test_int4.py [test names]     # or, with pytest: pytest -s int4/tests

Random codes in -7..7 and bf16 group scales (G = 32); x with a few outlier channels. Reference: x rounded to bf16 times
q * s in float32 (the device's weight exactly). Bars, relative Frobenius error and PCC above 0.9999:
- the matmul at HiFi4: below 1 %;
- the matmul at HiFi2 (its default fidelity): below 1 % as well;
- the dequantized bfloat8_b weight: below 2 % (its block-float rounding).
One loader against two (the same chunks in the same order): bit for bit.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import device as cdev  # noqa: E402
from common.testing import check, main, same  # noqa: E402
from int4 import w4a16  # noqa: E402


def make_case(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-7, 8, (n, k), generator=g).to(torch.int8)
    scales = (torch.rand(n, k // 32, generator=g) * 0.01 + 0.001).to(torch.bfloat16)
    return codes, scales


def make_x(device, k, seed, memory_config):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, k, generator=g)
    x[:, torch.randperm(k, generator=g)[:8]] *= 20  # a few outlier channels
    x = x.to(torch.bfloat16)
    tx = ttnn.from_torch(
        x.reshape(1, 1, 1, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    return x, tx


def host(y, n):
    return ttnn.to_torch(y).float().reshape(-1, y.shape[-1])[:1, :n]


def test_matmul(device):
    matmul_case(device, "hifi4")


def test_hifi2(device):
    matmul_case(device, "hifi2")


def matmul_case(device, fid):
    for k, n in ((1024, 1152), (3072, 1024)):
        codes, scales = make_case(n, k, k + n)
        wq = w4a16.int4_weight(device, codes)
        ts = w4a16.group_scales(device, scales)
        for mc, where in ((ttnn.L1_MEMORY_CONFIG, "L1"), (ttnn.DRAM_MEMORY_CONFIG, "DRAM")):
            x, tx = make_x(device, k, n, mc)
            ref = w4a16.reference(x, codes, scales)
            y = host(w4a16.w4a16_linear(tx, wq, ts, fidelity=fid), n)
            check(f"int4 {fid} K {k} N {n} x in {where}", y, ref)
            same(f"int4 {fid} K {k} N {n} x in {where}: one loader against two",
                 host(w4a16.w4a16_linear(tx, wq, ts, split=False, fidelity=fid), n), y)
            ttnn.deallocate(tx)
        ttnn.deallocate(wq["t"])
        ttnn.deallocate(ts)


def test_fewer_banks(device):
    k, n = 1024, 1024
    banks = cdev.dram_banks(device)
    fewer = max(1, banks // 2)
    codes, scales = make_case(n, k, 7)
    wq = w4a16.int4_weight(device, codes, max_banks=fewer)
    assert wq["banks"] == fewer and wq["t"].memory_config().shard_spec.num_cores() == fewer
    ts = w4a16.group_scales(device, scales)
    x, tx = make_x(device, k, 8, ttnn.L1_MEMORY_CONFIG)
    check(f"int4 hifi4 weight over {fewer} of {banks} banks", host(w4a16.w4a16_linear(tx, wq, ts, fidelity="hifi4"), n),
          w4a16.reference(x, codes, scales))


def test_dequantize(device):
    k, n = 2048, 1152
    codes, scales = make_case(n, k, 9)
    wq = w4a16.int4_weight(device, codes)
    ts = w4a16.group_scales(device, scales)
    wd = w4a16.dequantize(wq, ts)
    got = ttnn.to_torch(wd).float().reshape(k, -1)[:, :n]
    check("dequantized bfloat8_b weight against q * s", got, w4a16.dequantized(codes, scales).T, rel=0.02)
    again = w4a16.dequantize(wq, ts, out=wd)
    same("dequantize into a given output: the same tensor and values",
         ttnn.to_torch(again).float().reshape(k, -1)[:, :n], got)


if __name__ == "__main__":
    sys.exit(main(globals()))
