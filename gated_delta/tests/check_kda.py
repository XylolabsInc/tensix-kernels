# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare the verify kernel's KDA mode with an fp64 recurrence using synthetic inputs.

  python check_kda.py [--rows 1,2,4,20,32] [--state zero,random] [--seeds 1,2] [--bar 1.0]

Use 16 heads with 128 channels, as in Ling-3.0-tiny. The fused input is bf16 [q | k | v | f | b |
gate], with padding before the output gate. The state is fp32, initially zero or random.
The fp64 reference computes g = -5 * sigmoid(exp(A_log) * (f + dt_bias)) and beta = sigmoid(b)
from the same bf16 inputs. Compare the gated output and every row's state, including slots,
by relative L2 error. --bar sets the maximum allowed percentage error. Run on ttsim or a device.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import gdn_verify

H = 16
D = 128
TILE = 32
LB = -5.0
N = H * D
F_COL, B_COL = 3 * N, 4 * N  # element columns: f after q, k, v; b after f
G_COL = B_COL + TILE  # the output gate after b's padded tile column
W = G_COL + N


def bf(t):
    return t.float().to(torch.bfloat16).float()


def put(t, device, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def l2n(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def reference(x, S0, params, nw):
    """float64 over the rows of x (the bf16 fused projection output): (out [R, H, D], the state after each row)."""
    a = params["A_log"].double().exp()
    db = params["dt_bias"].double().view(H, D)
    S = S0.double().clone()
    outs, states = [], []
    for r in range(x.shape[0]):
        row = x[r].double()
        q, k, v = (row[i * N : (i + 1) * N].view(H, D) for i in range(3))
        f = row[F_COL : F_COL + N].view(H, D)
        beta = torch.sigmoid(row[B_COL : B_COL + H])
        gate = row[G_COL : G_COL + N].view(H, D)
        g = LB * torch.sigmoid(a[:, None] * (f + db))
        qn, kn = l2n(q) * D**-0.5, l2n(k)
        S = S * g.exp()[..., None]
        u = (kn[..., None] * S).sum(-2)
        S = S + (beta[:, None] * kn)[..., None] * (v - u)[..., None, :]
        o = (qn[..., None] * S).sum(-2)
        rms = torch.sqrt((o * o).mean(-1, keepdim=True) + 1e-6)
        outs.append(o / rms * nw.double() * torch.sigmoid(gate))
        states.append(S.clone())
    return torch.stack(outs), states


def synthetic(rows, seed, state):
    """Random parameters and inputs at this shape: (params, the norm weight, the fused projection output in bf16 [rows, W], the
    state before the first row: zeros, or random when `state`)."""
    gen = torch.Generator().manual_seed(seed)
    params = {
        "A_log": torch.randn(H, generator=gen) * 0.5,
        "dt_bias": torch.randn(N, generator=gen) * 0.5,
    }
    nw = 1 + 0.1 * torch.randn(D, generator=gen)
    x = torch.zeros(rows, W)
    x[:, : 3 * N] = torch.randn(rows, 3 * N, generator=gen)
    x[:, F_COL : F_COL + N] = torch.randn(rows, N, generator=gen)
    x[:, B_COL : B_COL + H] = torch.randn(rows, H, generator=gen)
    x[:, G_COL : G_COL + N] = torch.randn(rows, N, generator=gen)
    S0 = 0.1 * torch.randn(H, D, D, generator=gen) if state else torch.zeros(H, D, D)
    return params, nw, bf(x), S0


def rel(got, ref):
    got, ref = got.double().flatten(), ref.double().flatten()
    return ((got - ref).norm() / ref.norm()).item() if ref.norm() > 0 else math.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,20,32", help="rows per call (at most 32)")
    ap.add_argument("--state", default="zero,random", help="the state before the first row")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument(
        "--bar", type=float, default=1.0, help="largest relative error in %% for a pass"
    )
    a = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T %Z')}", flush=True)
    worst, fails = 0.0, 0
    try:
        for seed in (int(v) for v in a.seeds.split(",")):
            for st0 in a.state.split(","):
                for rows in (int(v) for v in a.rows.split(",")):
                    params, nw, x, S0 = synthetic(rows, seed * 1000 + rows, st0 == "random")
                    ref_out, ref_states = reference(x, S0, params, nw)
                    call, L = f"{st0} state, seed {seed}", "synthetic"
                    tx = put(x.view(1, rows, W), device, ttnn.bfloat16)
                    st = put(S0.view(1, H, D, D), device, ttnn.float32)
                    slots = [
                        put(torch.zeros(1, H, D, D), device, ttnn.float32) for _ in range(rows - 1)
                    ]
                    tnw = put(nw.view(1, 1, D).expand(1, TILE, D).clone(), device, ttnn.bfloat16)
                    o = gdn_verify.gated_delta_verify(
                        tx,
                        tx,
                        tx,
                        st,
                        slots,
                        num_heads=H,
                        num_v_heads=H,
                        head_k_dim=D,
                        head_v_dim=D,
                        cols=(0, N // TILE, 2 * N // TILE),
                        a_col=B_COL,
                        b_col=B_COL,
                        raw_gates=([0.0] * H, [0.0] * H),
                        gate=tx,
                        gate_col=G_COL // TILE,
                        norm_weight=tnw,
                        norm_eps=1e-6,
                        out_dtype=ttnn.bfloat16,
                        kda={
                            "f_col": F_COL,
                            "dt_bias": params["dt_bias"].float(),
                            "a": params["A_log"].float().exp().tolist(),
                        },
                    )
                    ttnn.synchronize_device(device)
                    got_out = ttnn.to_torch(o).float().view(rows, H, D)
                    got_states = [ttnn.to_torch(s).float().view(H, D, D) for s in slots]
                    got_states.append(ttnn.to_torch(st).float().view(H, D, D))
                    eo = 100 * rel(got_out, ref_out)
                    es = [100 * rel(gs, rs) for gs, rs in zip(got_states, ref_states, strict=True)]
                    bad = not (eo <= a.bar and max(es) <= a.bar)
                    fails += bad
                    worst = max(worst, eo, max(es))
                    print(
                        f"[KDA {call}, {rows} row(s)] out {eo:.3f} %, state after the last row {es[-1]:.3f} %, "
                        f"worst row's state {max(es):.3f} % ({'FAIL' if bad else 'pass'} at {a.bar} %)",
                        flush=True,
                    )
                    for t in (tx, st, *slots, tnw, o):
                        ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T %Z')}", flush=True)
    print(f"check_kda: {fails} failed, worst {worst:.3f} %", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
