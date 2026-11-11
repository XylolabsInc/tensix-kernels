# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare multi-row verification with chained decode steps and an fp64 recurrence.

  python check_rows.py [small] [qwen] [--rows 1,2,3,4] [--gnorm] [--rollback]

Run on ttsim or a device. For each R, compare every output row, state slot, and final state
against R calls of gdn_decode.py, carrying the state in DRAM. Report maximum absolute
difference and PCC; the device paths should match bit for bit. Compare both against fp64.
--rollback runs four rows, then three new rows from slot 1, the state after accepting rows
0 and 1. Compare with decode continued from its own state after row 1. Inputs follow
check_decode.py: q, k, and v in a flat conv output, raw a and b in one tensor.
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
import gdn_verify
from check import dev
from check_decode import CASES

K = V = 128


def inputs(H, HV, R, seed, gnorm):
    """Host tensors of R rows with check_decode.py's statistics (bf16-representable)."""
    B = 1
    torch.manual_seed(seed)

    def bf(t):
        return t.to(torch.bfloat16).float()

    d = {
        "q": bf(torch.randn(B, R, H, K)),
        "k": bf(torch.randn(B, R, H, K)),
        "v": bf(torch.randn(B, R, HV, V) * 0.5),
        "a_neg": bf(-(torch.rand(HV) * 2 - 1.5).exp()),
        "dt": bf(0.5 * torch.randn(HV)),
    }
    d["a_raw"] = bf(torch.randn(B, R, HV) * 0.8 - 1.0 - d["dt"])
    d["b_raw"] = bf(torch.randn(B, R, HV))
    d["gate"] = bf(torch.randn(B, R, HV * V) * 2) if gnorm else None
    d["x"] = torch.cat(
        [d["q"].reshape(B, R, -1), d["k"].reshape(B, R, -1), d["v"].reshape(B, R, -1)], -1
    )
    width = -(-(2 * HV + 32) // 32) * 32
    ab = torch.zeros(B, R, width)
    ab[..., 32 : 32 + HV] = d["a_raw"]
    ab[..., 32 + HV : 32 + 2 * HV] = d["b_raw"]
    d["ab"] = ab
    return d


def fp64_rows(d, s0, w, gnorm):
    """The exact recurrence row by row: outputs [B, R, HV, V] (gated norm applied) and the state
    after each row."""
    g = d["a_neg"] * torch.nn.functional.softplus(d["a_raw"] + d["dt"])
    beta = torch.sigmoid(d["b_raw"])
    R = d["q"].shape[1]
    s, outs, states = s0, [], []
    for r in range(R):
        sl = slice(r, r + 1)
        o, s = gdn_scan.torch_recurrent(
            d["q"][:, sl], d["k"][:, sl], d["v"][:, sl], g[:, sl], beta[:, sl], s
        )
        if gnorm:
            o = o / torch.sqrt((o * o).mean(-1, keepdim=True) + 1e-6) * w.double()
            o = o * torch.nn.functional.silu(d["gate"][:, sl].double().reshape(o.shape))
        outs.append(o)
        states.append(s)
    return torch.cat(outs, 1), states


def slots_like(state, n):
    """n zero tensors laid out as the state (the verify's slots)."""
    return [
        ttnn.zeros(
            list(state.shape),
            dtype=state.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=state.memory_config(),
        )
        for _ in range(n)
    ]


def diff(name, ref, got):
    """Max abs difference and PCC of got against ref (both torch)."""
    ref, got = ref.double().flatten(), got.double().flatten()
    mad = (ref - got).abs().max().item()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item() if mad > 0 else 1.0
    return f"{name}: max abs {mad:.3e}, PCC {pcc:.7f}{' (bit-identical)' if mad == 0 else ''}"


class Case:
    def __init__(self, device, name, gnorm):
        self.device, self.gnorm = device, gnorm
        self.H, self.HV = CASES[name]
        torch.manual_seed(100)
        self.w = (1 + 0.2 * torch.randn(V)).to(torch.bfloat16).float()
        self.tw = (
            ttnn.from_torch(
                self.w.reshape(1, 1, V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            if gnorm
            else None
        )

    def kw(self, d, gate):
        k = {
            "num_heads": self.H,
            "num_v_heads": self.HV,
            "cols": (0, self.H * K // 32, 2 * self.H * K // 32),
            "a_col": 32,
            "b_col": 32 + self.HV,
            "raw_gates": (d["a_neg"], d["dt"]),
            "out_dtype": ttnn.float32,
        }
        if self.gnorm:
            k.update(gate=gate, norm_weight=self.tw, norm_eps=1e-6)
        return k

    def verify(self, d, state):
        """The kernel on all rows: (o [B, R, HV, V], the slots' states, the final state)."""
        dv = self.device
        R = d["x"].shape[1]
        slots = slots_like(state, R - 1)
        tx, tab = dev(d["x"], ttnn.bfloat16, dv), dev(d["ab"], ttnn.bfloat16, dv)
        gate = dev(d["gate"], ttnn.bfloat16, dv) if self.gnorm else None
        o = gdn_verify.gated_delta_verify(tx, tab, tab, state, slots, **self.kw(d, gate))
        got = ttnn.to_torch(o).float()[:, :R].reshape(1, R, self.HV, V)
        return got, [ttnn.to_torch(s).float() for s in slots], ttnn.to_torch(state).float(), slots

    def decode_rows(self, d, state):
        """The decode step row by row on `state`: (o [B, R, HV, V], the state after each row)."""
        dv = self.device
        R = d["x"].shape[1]
        outs, states = [], []
        for r in range(R):
            sl = slice(r, r + 1)
            tx = dev(d["x"][:, sl], ttnn.bfloat16, dv)
            tab = dev(d["ab"][:, sl], ttnn.bfloat16, dv)
            gate = dev(d["gate"][:, sl], ttnn.bfloat16, dv) if self.gnorm else None
            o = gdn_decode.gated_delta_decode(tx, tab, tab, state, **self.kw(d, gate))
            outs.append(ttnn.to_torch(o).float()[:, :1].reshape(1, 1, self.HV, V))
            states.append(ttnn.to_torch(state).float())
        return torch.cat(outs, 1), states


def rows(device, name, R, gnorm):
    c = Case(device, name, gnorm)
    d = inputs(c.H, c.HV, R, seed=7 + R, gnorm=gnorm)
    torch.manual_seed(11)
    s0 = (torch.randn(1, c.HV, K, V) * 0.1).to(torch.bfloat16).float()
    t0 = time.monotonic()
    got_o, got_slots, got_s, _ = c.verify(d, dev(s0, ttnn.bfloat16, device))
    first = time.monotonic() - t0
    dec_o, dec_states = c.decode_rows(d, dev(s0, ttnn.bfloat16, device))
    ref_o, ref_states = fp64_rows(d, s0, c.w, gnorm)
    print(
        f"[rows {name}] R {R} H {c.H} HV {c.HV}{' gnorm' if gnorm else ''}: "
        f"first call {first:.2f} s"
    )
    for r in range(R):
        print("  " + diff(f"row {r} o vs decode", dec_o[:, r], got_o[:, r]))
        s = got_slots[r] if r < R - 1 else got_s
        what = f"slot {r}" if r < R - 1 else "final state"
        print("  " + diff(f"row {r} {what} vs decode", dec_states[r], s))
    print("  " + gdn_scan.compare("o vs fp64", ref_o, got_o)[0].strip())
    print(
        "  " + gdn_scan.compare("final state vs fp64", ref_states[-1], got_s)[0].strip(), flush=True
    )


def rollback(device, name, gnorm):
    c = Case(device, name, gnorm)
    d1 = inputs(c.H, c.HV, 4, seed=21, gnorm=gnorm)
    d2 = inputs(c.H, c.HV, 3, seed=22, gnorm=gnorm)
    torch.manual_seed(12)
    s0 = (torch.randn(1, c.HV, K, V) * 0.1).to(torch.bfloat16).float()
    _, got_slots, _, _ = c.verify(d1, dev(s0, ttnn.bfloat16, device))
    # Accepted rows 0 and 1: continue from slot 1.
    got_o2, got_slots2, got_s2, _ = c.verify(d2, dev(got_slots[1], ttnn.bfloat16, device))
    _, dec_states = c.decode_rows(d1, dev(s0, ttnn.bfloat16, device))
    dec_o2, dec_states2 = c.decode_rows(d2, dev(dec_states[1], ttnn.bfloat16, device))
    print(f"[rollback {name}] verify 4 rows, then 3 rows from slot 1{' gnorm' if gnorm else ''}")
    print("  " + diff("slot 1 vs decode after row 1", dec_states[1], got_slots[1]))
    for r in range(3):
        print("  " + diff(f"continued row {r} o vs decode", dec_o2[:, r], got_o2[:, r]))
    for r in range(2):
        print("  " + diff(f"continued slot {r} vs decode", dec_states2[r], got_slots2[r]))
    print("  " + diff("continued final state vs decode", dec_states2[2], got_s2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", choices=list(CASES))
    ap.add_argument("--rows", default="1,2,3,4")
    ap.add_argument("--gnorm", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()
    a.cases = a.cases or ["small"]  # not argparse's default: newer Pythons check it against choices
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T %Z')}", flush=True)
    try:
        for name in a.cases:
            for R in (int(v) for v in a.rows.split(",") if v):
                rows(device, name, R, a.gnorm)
            if a.rollback:
                rollback(device, name, a.gnorm)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T %Z')}", flush=True)


if __name__ == "__main__":
    import _verdict

    _verdict.watch()
    main()
    sys.exit(_verdict.finish("check_rows"))
