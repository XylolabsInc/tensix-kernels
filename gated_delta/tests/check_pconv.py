# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Check prompt convolution and state outputs against upstream ops, bit for bit.

  python check_pconv.py [--shape small|qwen] [--rows 1,2,3,31,32,33,64,100,256] [--chain 256,37] [--time N]

Compare gdn_pconv.prefill_conv with _causal_conv1d_fir and its surrounding slices through
gdn_pconv.stock. Check T output rows and the new state, both fused and split into q, k, and v.
Run on ttsim or a device. x is [1, T, W] bf16 with C convolution channels first; state is
[1, 3, C] or None for zeros; four taps are [1, 1, C]. Inputs are random, with a tenth of x's
values zero. --chain processes calls in order, carrying state between them in both paths.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
UPSTREAM = Path(os.environ.get("TT_METAL_MODELS", os.environ.get("TT_METAL_HOME", ".")))  # a tt-metal checkout: its models/ has the stock path
if (UPSTREAM / "models").is_dir():
    sys.path.append(str(UPSTREAM))
import gdn_pconv

SHAPES = {"small": (1024, 256, 256, 1600), "qwen": (10240, 2048, 2048, 16480)}  # C, q_dim, k_dim, W


def put(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def bits(t):
    return ttnn.to_torch(t).contiguous().view(torch.int16)


def same(name, a, b):
    """Bit for bit over the logical shape; else where and how much."""
    ta, tb = bits(a), bits(b)
    if ta.shape != tb.shape:
        print(f"  {name}: shapes {list(ta.shape)} and {list(tb.shape)}", flush=True)
        return False
    d = ta != tb
    if not d.any():
        print(f"  {name}: bit-identical ({ta.numel()} values)", flush=True)
        return True
    fa, fb = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
    idx = d.nonzero()[0].tolist()
    print(
        f"  {name}: {int(d.sum())} of {ta.numel()} differ, max |diff| {(fa - fb).abs().max().item():.3e}, "
        f"first at {idx}: {fa[tuple(idx)].item()!r} against {fb[tuple(idx)].item()!r}",
        flush=True,
    )
    return False


def inputs(C, W, T, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, W, generator=g)
    x[torch.rand(1, T, W, generator=g) < 0.1] = 0
    s = torch.randn(1, 3, C, generator=g)
    w = [torch.randn(1, 1, C, generator=g) * 0.5 for _ in range(4)]
    return x.bfloat16(), s.bfloat16(), [t.bfloat16() for t in w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--rows", default="2,3,31,32,33,64,100")  # T 1 is the decode conv
    ap.add_argument("--chain", default="")
    ap.add_argument(
        "--no-state", action="store_true", help="also each row count without a state (zeros)"
    )
    ap.add_argument(
        "--time", type=int, default=0, help="N timed calls of each path at the first row count"
    )
    ap.add_argument(
        "--stock-l1",
        action="store_true",
        help="the stock FIR's intermediates in L1 as the model's "
        "(default: DRAM, the same ops and results; L1 runs out at the 27B's width)",
    )
    a = ap.parse_args()
    C, qd, kd, W = SHAPES[a.shape]
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T')}", flush=True)
    fails = 0
    try:
        torch.manual_seed(0)
        _, _, w0 = inputs(C, W, 1, 99)
        taps = [put(t, device) for t in w0]
        kw = {"channels": C, "q_dim": qd, "k_dim": kd}
        runs = [[int(t)] for t in a.rows.split(",") if t] + (
            [[int(t) for t in a.chain.split(",")]] if a.chain else []
        )
        for seq in runs:
            for with_state in (True, False) if a.no_state else (True,):
                _, s0, _ = inputs(C, W, max(seq), sum(seq) + with_state)
                st_k = st_s = put(s0, device) if with_state else None
                for i, T in enumerate(seq):
                    x = put(inputs(C, W, T, 1000 * i + T)[0], device)
                    tag = f"[{a.shape} T {T}{' (chain ' + str(i) + ')' if len(seq) > 1 else ''}{'' if with_state else ', no state'}]"
                    print(tag, flush=True)
                    got = gdn_pconv.prefill_conv(x, st_k, taps, **kw)
                    ttnn.synchronize_device(device)
                    print("  kernel done", flush=True)  # a hang before this line is the kernel's
                    ref = gdn_pconv.stock(x, st_s, taps, dram=not a.stock_l1, **kw)
                    ttnn.synchronize_device(device)
                    print("  stock done", flush=True)
                    for name, g_, r_ in zip(
                        ("out", "state", "conv_q", "conv_k", "conv_v"), got, ref, strict=True
                    ):
                        fails += not same(name, g_, r_)
                    st_k, st_s = got[1], ref[1]
        if a.time:
            T = runs[0][0]
            x = put(inputs(C, W, T, 7)[0], device)
            s = put(inputs(C, W, 1, 8)[1], device)
            for name, fn in (("kernel", gdn_pconv.prefill_conv), ("stock", gdn_pconv.stock)):
                fn(x, s, taps, **kw)
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                for _ in range(a.time):
                    fn(x, s, taps, **kw)
                ttnn.synchronize_device(device)
                print(
                    f"time {name} T {T}: {(time.perf_counter() - t0) / a.time * 1e3:.3f} ms a call (eager, wall)",
                    flush=True,
                )
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T')}", flush=True)
    print(f"check_pconv: {fails} mismatch(es)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
