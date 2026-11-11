# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare one Gated DeltaNet decode step with an fp64 recurrence from random state.

  python check_decode.py [small] [qwen] [--gnorm] [--time N] [--batch B]

Run on ttsim or a device. q, k, and v share one flat convolution output. Raw a and b share a
tensor, with a at a tile boundary and b HV columns later, as in a fused projection output.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_decode
import gdn_scan
from check import dev

# (H, HV), K = V = 128; q9b: Qwen3.5-9B
CASES = {"small": (2, 6), "qwen": (16, 48), "q9b": (16, 32)}


def make(device, name, gnorm, seed=3, B=1):
    """The decode inputs of case `name` on the device and the fp64 reference of one step: (args of
    gated_delta_decode as a tuple, its keywords, ref o, ref state, the state tensor)."""
    H, HV = CASES[name]
    K = V = 128
    torch.manual_seed(seed)
    q = torch.randn(B, 1, H, K).to(torch.bfloat16).float()
    k = torch.randn(B, 1, H, K).to(torch.bfloat16).float()
    v = (torch.randn(B, 1, HV, V) * 0.5).to(torch.bfloat16).float()
    A = torch.rand(HV) * 2 - 1.5
    a_neg = (-A.exp()).to(torch.bfloat16).float()
    dt = (0.5 * torch.randn(HV)).to(torch.bfloat16).float()
    a_raw = (torch.randn(B, 1, HV) * 0.8 - 1.0 - dt).to(torch.bfloat16).float()
    b_raw = torch.randn(B, 1, HV).to(torch.bfloat16).float()
    s0 = (torch.randn(B, HV, K, V) * 0.1).to(torch.bfloat16).float()
    g = a_neg * torch.nn.functional.softplus(a_raw + dt)
    beta = torch.sigmoid(b_raw)
    ref_o, ref_s = gdn_scan.torch_recurrent(q, k, v, g, beta, s0)
    x = torch.cat([q.reshape(B, 1, -1), k.reshape(B, 1, -1), v.reshape(B, 1, -1)], -1)
    width = -(-(2 * HV + 32) // 32) * 32
    ab = torch.zeros(B, 1, width)
    ab[..., 32 : 32 + HV] = a_raw
    ab[..., 32 + HV : 32 + 2 * HV] = b_raw
    kw = {
        "num_heads": H,
        "num_v_heads": HV,
        "cols": (0, H * K // 32, 2 * H * K // 32),
        "a_col": 32,
        "b_col": 32 + HV,
        "raw_gates": (a_neg, dt),
        "out_dtype": ttnn.float32,
    }
    tx, tab = dev(x, ttnn.bfloat16, device), dev(ab, ttnn.bfloat16, device)
    if gnorm:
        gate = (torch.randn(B, 1, HV * V) * 2).to(torch.bfloat16).float()
        w = (1 + 0.2 * torch.randn(V)).to(torch.bfloat16).float()
        o_ = ref_o
        ref_o = o_ / torch.sqrt((o_ * o_).mean(-1, keepdim=True) + 1e-6) * w.double()
        ref_o = ref_o * torch.nn.functional.silu(gate.double().reshape(B, 1, HV, V))
        kw.update(
            gate=dev(gate, ttnn.bfloat16, device),
            norm_weight=ttnn.from_torch(
                w.reshape(1, 1, V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            ),
            norm_eps=1e-6,
        )
    ts = dev(s0, ttnn.bfloat16, device)
    return (tx, tab, tab, ts), kw, ref_o, ref_s, ts


def run(device, name, a):
    H, HV = CASES[name]
    args, kw, ref_o, ref_s, ts = make(device, name, a.gnorm, B=a.batch)
    tx, tab, _, _ = args
    t0 = time.monotonic()
    o = gdn_decode.gated_delta_decode(tx, tab, tab, ts, **kw)
    ttnn.synchronize_device(device)
    first = time.monotonic() - t0
    got_o = ttnn.to_torch(o).float().reshape(ref_o.shape)
    got_s = ttnn.to_torch(ts).float().reshape(ref_s.shape)
    print(
        f"[decode {name}] B {a.batch} H {H} HV {HV}{' gnorm' if a.gnorm else ''}: "
        f"first call {first:.2f} s"
    )
    print(gdn_scan.compare("  o", ref_o, got_o)[0], flush=True)
    print(gdn_scan.compare("  state (in place)", ref_s, got_s)[0], flush=True)
    if a.time:
        times = []
        for _ in range(a.time):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            o2 = gdn_decode.gated_delta_decode(tx, tab, tab, ts, **kw)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
            ttnn.deallocate(o2)
        times.sort()
        print(
            f"  time per call (eager): median {times[len(times) // 2] * 1e3:.3f} ms, "
            f"min {times[0] * 1e3:.3f} ms ({a.time} calls)",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", choices=list(CASES))
    ap.add_argument("--gnorm", action="store_true")
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1, help="B sequences, each with its own state")
    a = ap.parse_args()
    a.cases = a.cases or ["small"]  # not argparse's default: newer Pythons check it against choices
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T %Z')}", flush=True)
    try:
        for name in a.cases:
            run(device, name, a)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T %Z')}", flush=True)


if __name__ == "__main__":
    import _verdict

    _verdict.watch()
    main()
    sys.exit(_verdict.finish("check_decode"))
