# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Correctness of the ternary group-scale matmul (ternary_matmul/t2.py) against torch, on a Tensix device or ttsim.

  python -s ternary_matmul/tests/test_ternary.py [test names]     # or, with pytest: pytest -s ternary_matmul/tests

Random trits (zero with probability 0.3, else +-1) and fp16 group scales; x with a few outlier channels. Reference:
x rounded to bf16 times q * bf16(s) in float32 (the device's weight exactly); with the Hadamard, x rotated in float64
(t2.rotate), so the kernel's bf16 x' is part of its error. Bars (PCC above 0.9999 throughout; relative Frobenius
error):
- at HiFi4, every kernel and the rotation: below BAR (0.5 %);
- at HiFi2 (the kernels' default fidelity): the one-token kernel below BAR_HIFI2_ONE (1.5 %; measured 0.78 to 1.26 %,
  the most with the fused rotation), the multi-row kernel below BAR_HIFI2_ROWS (0.8 %; measured 0.69 %). HiFi2 makes
  two of the matrix unit's four multiply passes over the operands' mantissas, so its products leave out low-order bits
  that HiFi4 keeps; the one-token kernel feels it most, as its scale stage and fused rotation multiply fp32 partial
  sums at the call's fidelity. A property of the fidelity, the same on any device.
Where two configurations must agree (the three weight formats, one loader against two, the fused Hadamard against the
Hadamard op (both at HiFi4), an offset output against the plain one, the top-k records against the op's own output),
the check is bit for bit.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.testing import check, main, same  # noqa: E402
from dense_reader import linear as dense_linear  # noqa: E402  (the top-k result tensor and its decode)
from ternary_matmul import t2  # noqa: E402

BAR = 0.005  # every check against the reference at HiFi4
BAR_HIFI2_ONE, BAR_HIFI2_ROWS = 0.015, 0.008  # at HiFi2: the measured errors with a margin (see the docstring)
FID = "hifi4"


def make_weight(n, k, g, seed, zeros=0.3):
    gen = torch.Generator().manual_seed(seed)
    u = torch.randint(0, 2000, (n, k), generator=gen, dtype=torch.int16)
    trits = torch.where(u < 2000 * zeros, 0, torch.where(u % 2 == 0, 1, -1)).to(torch.int8)
    scales = (0.01 + 0.02 * torch.rand(n, k // g, generator=gen)).to(torch.float16)
    return trits, scales


def make_x(m, k, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=gen)
    x[:, torch.randperm(k, generator=gen)[:8]] *= 20  # a few outlier channels
    return x.to(torch.bfloat16)


def to_dev(device, x, memory_config=ttnn.L1_MEMORY_CONFIG):
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        memory_config=memory_config,
    )


def host(y, m, n):
    return ttnn.to_torch(y).float().reshape(-1, y.shape[-1])[:m, :n]


def signs_for(k, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(k, generator=gen) < 0.5, -1.0, 1.0)


def test_formats_and_loaders(device):
    k, n, g = 1024, 1024, 128
    trits, scales = make_weight(n, k, g, 1)
    x = make_x(1, k, 2)
    ref = t2.reference(x, trits, scales, g)
    ts = t2.group_scales(device, scales)
    tx = to_dev(device, x)
    base = None
    for fmt in ("bfp2", "bfp2c", "bfp4"):
        w = t2.ternary_weight(device, trits, fmt=fmt)
        y = host(t2.t2_linear(tx, w, ts, g, fidelity=FID), 1, n)
        check(f"{fmt} {FID} K {k} N {n} G {g}", y, ref, rel=BAR)
        if base is None:
            base = y
        else:
            same(f"{fmt} against bfp2", y, base)
        same(f"{fmt}: one loader against two", host(t2.t2_linear(tx, w, ts, g, split=False, fidelity=FID), 1, n), y)
        ttnn.deallocate(w["t"])


def test_group32_x_in_dram(device):
    k, n, g = 2048, 512, 32
    trits, scales = make_weight(n, k, g, 3)
    x = make_x(1, k, 4)
    w = t2.ternary_weight(device, trits)
    ts = t2.group_scales(device, scales)
    for mc, where in ((ttnn.DRAM_MEMORY_CONFIG, "DRAM"), (ttnn.L1_MEMORY_CONFIG, "L1")):
        y = host(t2.t2_linear(to_dev(device, x, mc), w, ts, g, fidelity=FID), 1, n)
        check(f"bfp2 {FID} K {k} N {n} G {g} x in {where}", y, t2.reference(x, trits, scales, g), rel=BAR)


def test_hadamard(device):
    k, n, g = 2048, 1024, 128
    trits, scales = make_weight(n, k, g, 5)
    signs = signs_for(k, 6)
    x = make_x(1, k, 7)
    xr = t2.rotate(x.double(), signs).float()
    ref = xr @ t2.dequantize(trits, scales.to(torch.bfloat16), g).T
    w = t2.ternary_weight(device, trits)
    ts = t2.group_scales(device, scales, t2.HADAMARD_SCALE)
    mask = t2.hadamard_signs(device, signs)
    tx = to_dev(device, x)
    f4 = host(t2.t2_linear(tx, w, ts, g, hadamard=mask, fidelity=FID), 1, n)
    check(f"fused Hadamard {FID} K {k} N {n}", f4, ref, rel=BAR)
    xh = t2.hadamard(tx, mask)  # (the op runs at HiFi4)
    check("the Hadamard op against x rotated in float64", host(xh, 1, k), xr * 32, rel=BAR)
    apart = host(t2.t2_linear(xh, w, ts, g, fidelity=FID), 1, n)
    check(f"Hadamard op, then the matmul {FID} K {k} N {n}", apart, ref, rel=BAR)
    # the fused rotation runs at the call's fidelity, the op at HiFi4: at HiFi4 both, the same bits
    same("fused Hadamard against the Hadamard op, both at HiFi4", f4, apart)


def test_rows(device):
    k, n, g = 2048, 512, 128
    trits, scales = make_weight(n, k, g, 8)
    signs = signs_for(k, 9)
    w = t2.ternary_weight(device, trits)
    ts = t2.group_scales(device, scales, t2.HADAMARD_SCALE)
    mask = t2.hadamard_signs(device, signs)
    wd = t2.dequantize(trits, scales.to(torch.bfloat16), g).T
    for m in (5, 160):  # one call; two (4 + 1 tile rows)
        x = make_x(m, k, 10 + m)
        ref = t2.rotate(x.double(), signs).float() @ wd
        y = t2.linear(to_dev(device, x), w, ts, g, hadamard=mask, fidelity=FID)
        check(f"linear {m} rows with the Hadamard (rotate_rows, t2m_linear), {FID}", host(y, m, n), ref, rel=BAR)
    x = make_x(3, k, 20)
    y = t2.linear(to_dev(device, x), w, ts, g, hadamard=mask, rows_kernel=False, fidelity=FID)
    check(f"linear 3 rows, the one-token kernel per row, {FID}", host(y, 3, n),
          t2.rotate(x.double(), signs).float() @ wd, rel=BAR)


def test_topk_and_offset_output(device):
    k, n, g, kk = 1024, 1024, 128, 8
    trits, scales = make_weight(n, k, g, 21)
    x = make_x(1, k, 22)
    w = t2.ternary_weight(device, trits)
    ts = t2.group_scales(device, scales)
    tx = to_dev(device, x)
    grid = device.compute_with_storage_grid_size()
    r = dense_linear.topk_result(device, grid.x * grid.y, 1, kk)
    spec = {"r": r, "k": kk, "slot0": 0, "base": 0, "n": n, "write_y": True}
    y = host(t2.t2_linear(tx, w, ts, g, topk=spec, fidelity=FID), 1, n)
    check(f"top-k call's output {FID} K {k} N {n}", y, t2.reference(x, trits, scales, g), rel=BAR)
    vals, ids = dense_linear.topk_records(r, 1, kk)
    ok = ids[0] >= 0
    same("top-k records: values at their ids", y[0][ids[0][ok]], vals[0][ok])
    same(f"top-k records: the best {kk} of the row", torch.sort(vals[0][ok], descending=True).values[:kk],
         torch.topk(y[0], kk).values)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, 1152]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    t2.t2_linear(tx, w, ts, g, out=out, col0=1, gap=(16, 2), fidelity=FID)  # tiles 0-15 at 1-16, 16-31 at 19-34
    o = host(out, 1, 1152)
    same("offset output (col0 1, gap at tile 16 of 2 tiles) against the plain output",
         torch.cat([o[:, 32:544], o[:, 608:1120]], 1), y)


def test_hifi2(device):
    """The kernels at HiFi2, their default fidelity, against HiFi2's measured accuracy (see the module docstring)."""
    k, n, g = 1024, 1024, 128
    trits, scales = make_weight(n, k, g, 1)
    x = make_x(1, k, 2)
    w = t2.ternary_weight(device, trits)
    y = host(t2.t2_linear(to_dev(device, x), w, t2.group_scales(device, scales), g, fidelity="hifi2"), 1, n)
    check(f"bfp2 hifi2 K {k} N {n} G {g}", y, t2.reference(x, trits, scales, g), rel=BAR_HIFI2_ONE)
    ttnn.deallocate(w["t"])
    k, n = 2048, 1024
    trits, scales = make_weight(n, k, g, 5)
    signs = signs_for(k, 6)
    x = make_x(1, k, 7)
    wd = t2.dequantize(trits, scales.to(torch.bfloat16), g).T
    w = t2.ternary_weight(device, trits)
    ts = t2.group_scales(device, scales, t2.HADAMARD_SCALE)
    mask = t2.hadamard_signs(device, signs)
    fused = host(t2.t2_linear(to_dev(device, x), w, ts, g, hadamard=mask, fidelity="hifi2"), 1, n)
    check(f"fused Hadamard hifi2 K {k} N {n}", fused, t2.rotate(x.double(), signs).float() @ wd, rel=BAR_HIFI2_ONE)
    ttnn.deallocate(w["t"])
    n = 512  # (test_rows' case)
    trits, scales = make_weight(n, k, g, 8)
    signs = signs_for(k, 9)
    w = t2.ternary_weight(device, trits)
    ts = t2.group_scales(device, scales, t2.HADAMARD_SCALE)
    mask = t2.hadamard_signs(device, signs)
    x = make_x(5, k, 15)
    y = t2.linear(to_dev(device, x), w, ts, g, hadamard=mask, fidelity="hifi2")
    check("linear 5 rows with the Hadamard (rotate_rows, t2m_linear), hifi2", host(y, 5, n),
          t2.rotate(x.double(), signs).float() @ t2.dequantize(trits, scales.to(torch.bfloat16), g).T,
          rel=BAR_HIFI2_ROWS)


if __name__ == "__main__":
    sys.exit(main(globals()))
