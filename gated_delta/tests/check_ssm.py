# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare Mamba-2 decode and verification with an fp64 SSM recurrence.

  python check_ssm.py nemo small [--rows 1,2,3,4] [--batch 1,2] [--state fp32] [--out fp32] [--l1]

Run on ttsim or a device. Per head h in group g = h // (HV / H), per row:
  dt = softplus(dt_raw + dt_bias)
  h_state = exp(A dt) h_state + B_g^T (dt x)    # [N, P]
  y = C_g h_state + D x
Inputs use the flat conv layout [x | B | C], with dt_raw in a separate tensor by column.
A = -exp(A_log). Compare output, every slot, and final state with fp64 using PCC and relative
error. Compare R rows in one call bit for bit with R chained single-row calls.
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

# (groups H, heads HV, state N, head dim P): Nemotron 3 Nano's Mamba-2 layer and a small one
CASES = {"nemo": (8, 64, 128, 64), "small": (2, 4, 128, 64)}
DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}
TILE = 32


def inputs(H, HV, N, P, R, B, seed):
    torch.manual_seed(seed)

    def bf(t):
        return t.to(torch.bfloat16).float()

    d = {
        "x": bf(torch.randn(B, R, HV, P)),
        "Bm": bf(torch.randn(B, R, H, N) * 0.3),
        "Cm": bf(torch.randn(B, R, H, N) * 0.3),
        "dt_raw": bf(torch.randn(B, R, HV) * 0.8 - 2.0),
        "A": -torch.exp(torch.log(torch.arange(1, HV + 1, dtype=torch.float32)) * 0.5),
        "dt_bias": bf(torch.randn(HV) * 0.3),
        "D": bf(torch.randn(HV) * 0.5 + 1.0),
        "s0": torch.randn(B, HV, N, P) * 0.1,
    }
    d["flat"] = torch.cat(
        [d["x"].reshape(B, R, -1), d["Bm"].reshape(B, R, -1), d["Cm"].reshape(B, R, -1)], -1
    )
    width = -(-(32 + HV) // 32) * 32
    ab = torch.zeros(B, R, width)
    ab[..., 32 : 32 + HV] = d["dt_raw"]
    d["ab"] = ab
    return d


def reference(d, H, HV):
    """float64: o [B, R, HV, P] and the state after each row."""
    G = HV // H
    s = d["s0"].double()
    A = d["A"].double()
    outs, states = [], []
    R = d["x"].shape[1]
    for r in range(R):
        dt = torch.nn.functional.softplus(
            d["dt_raw"][:, r].double() + d["dt_bias"].double()
        )  # [B, HV]
        dA = torch.exp(A[None] * dt)
        x = d["x"][:, r].double()  # [B, HV, P]
        Bg = d["Bm"][:, r].double().repeat_interleave(G, dim=1)  # [B, HV, N]
        Cg = d["Cm"][:, r].double().repeat_interleave(G, dim=1)
        s = dA[..., None, None] * s + Bg[..., :, None] * (dt[..., None] * x)[..., None, :]
        y = torch.einsum("bhn,bhnp->bhp", Cg, s) + d["D"].double()[None, :, None] * x
        outs.append(y)
        states.append(s.clone())
    return torch.stack(outs, 1), states


def run(device, d, H, HV, N, P, R0, R1, state_dtype, out_dtype, l1, state=None):
    """Rows R0 .. R1 - 1 in one call, from `state` (a device tensor) or s0: (o, slots, the state)."""
    mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG

    def put(t, dtype, memory_config=mc):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

    B = d["x"].shape[0]
    R = R1 - R0
    if state is None:
        state = put(d["s0"], state_dtype, ttnn.DRAM_MEMORY_CONFIG)
    slots = [
        ttnn.zeros(
            list(state.shape),
            dtype=state.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=state.memory_config(),
        )
        for _ in range(R - 1)
    ]
    xcols = HV * P // TILE
    bcols = H * N // TILE
    tx = put(d["flat"][:, R0:R1], ttnn.bfloat16)
    tab = put(d["ab"][:, R0:R1], ttnn.bfloat16)
    o = gdn_verify.gated_delta_verify(
        tx,
        tab,
        tab,
        state,
        slots,
        num_heads=H,
        num_v_heads=HV,
        head_k_dim=N,
        head_v_dim=P,
        cols=(xcols + bcols, xcols, 0),
        a_col=32,
        b_col=32,
        raw_gates=(d["A"], d["dt_bias"]),
        out_dtype=out_dtype,
        memory_config=mc,
        ssm_d=d["D"],
    )
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(o).float()[:, :R].reshape(B, R, HV, P)
    return got, [ttnn.to_torch(s).float() for s in slots], state


def cmp(name, ref, got):
    ref, got = ref.double().flatten(), got.double().flatten()
    rel = (got - ref).norm() / ref.norm()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    return f"  {name}: rel {100 * rel:.3f} %, max abs {(got - ref).abs().max():.3e}, PCC {pcc:.7f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", choices=list(CASES))
    ap.add_argument("--rows", default="1,2,3,4")
    ap.add_argument("--batch", default="1")
    ap.add_argument("--state", default="fp32", choices=list(DTYPES))
    ap.add_argument("--out", default="fp32", choices=list(DTYPES))
    ap.add_argument("--l1", action="store_true")
    a = ap.parse_args()
    sd, od = DTYPES[a.state], DTYPES[a.out]
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T %Z')}", flush=True)
    bad = 0
    try:
        for name in a.cases:
            H, HV, N, P = CASES[name]
            for R in (int(v) for v in a.rows.split(",")):
                for B in (int(v) for v in a.batch.split(",")):
                    d = inputs(H, HV, N, P, R, B, seed=11 + R + 100 * B)
                    ref_o, ref_s = reference(d, H, HV)
                    o, slots, state = run(device, d, H, HV, N, P, 0, R, sd, od, a.l1)
                    s_fin = ttnn.to_torch(state).float()
                    print(
                        f"[{name} R {R} B {B}] state {a.state}, out {a.out}{', L1' if a.l1 else ''}",
                        flush=True,
                    )
                    print(cmp("o vs fp64", ref_o, o), flush=True)
                    for r, s in enumerate(slots):
                        print(cmp(f"slot {r} vs fp64", ref_s[r], s), flush=True)
                    print(cmp("final state vs fp64", ref_s[-1], s_fin), flush=True)
                    # the decode step: R calls of one row, chained on the state
                    st = None
                    rows = []
                    for r in range(R):
                        o1, _, st = run(device, d, H, HV, N, P, r, r + 1, sd, od, a.l1, state=st)
                        rows.append(o1)
                    o_dec = torch.cat(rows, 1)
                    s_dec = ttnn.to_torch(st).float()
                    same = bool(torch.equal(o_dec, o) and torch.equal(s_dec, s_fin))
                    bad += not same
                    print(
                        f"  rows in one call against {R} decode steps: {'bit-identical' if same else 'DIFFERENT'}"
                        f" (o max |diff| {(o_dec - o).abs().max():.3e}, state {(s_dec - s_fin).abs().max():.3e})",
                        flush=True,
                    )
                    if not math.isfinite(o.abs().max().item()):
                        bad += 1
                        print("  NON-FINITE output", flush=True)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T %Z')}", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    import _verdict

    _verdict.watch()
    main()
    sys.exit(_verdict.finish("check_ssm"))
