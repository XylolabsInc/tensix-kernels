# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Check prompt gates against stock ttnn ops, bit for bit over logical columns.

  python check_pgates.py [--shape small|qwen] [--rows 2,31,32,33,100,256] [--heads 48] [--bias-shape 3d|1d]

Compare gdn_pconv.prefill_gates with gdn_pconv.stock_gates on ttsim or a device. The fused
projection output x is [1, T, W] bf16, with a after the convolution channels and b after a.
dt_bias and A_log are random. Compute A_neg = -exp(A_log) on the device, as the stock path does.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_pconv
from check_pconv import SHAPES, same


def put(t, device):
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--rows", default="2,31,32,33,100,256")
    ap.add_argument("--heads", type=int, default=48)
    ap.add_argument("--bias-shape", default="3d", choices=("3d", "1d"))
    a = ap.parse_args()
    C, _, _, W = SHAPES[a.shape]
    HV = a.heads
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T')}", flush=True)
    fails = 0
    try:
        g0 = torch.Generator().manual_seed(17)
        shp = [1, 1, HV] if a.bias_shape == "3d" else [HV]
        dtb = put(torch.randn(*shp, generator=g0), device)
        a_log = put(torch.randn(*shp, generator=g0) * 0.5, device)
        a_neg = ttnn.neg(ttnn.exp(a_log))
        kw = {"a_col": C, "b_col": C + HV, "HV": HV, "dt_bias": dtb, "A_neg": a_neg}
        for T in (int(v) for v in a.rows.split(",")):
            g = torch.Generator().manual_seed(T)
            x = torch.randn(1, T, W, generator=g) * 3
            x[torch.rand(1, T, W, generator=g) < 0.05] = 0
            tx = put(x, device)
            print(f"[{a.shape} T {T} HV {HV} dt_bias {a.bias_shape}]", flush=True)
            n0 = gdn_pconv.GCALLS[0]
            got = gdn_pconv.prefill_gates(tx, **kw)
            ttnn.synchronize_device(device)
            print("  kernel done", flush=True)
            assert gdn_pconv.GCALLS[0] == n0 + 1
            ref = gdn_pconv.stock_gates(tx, dram=a.shape == "qwen", **kw)
            ttnn.synchronize_device(device)
            print("  stock done", flush=True)
            for name, g_, r_ in zip(("g", "beta"), got, ref, strict=True):
                fails += not same(name, g_, r_)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T')}", flush=True)
    print(f"check_pgates: {fails} mismatch(es)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
