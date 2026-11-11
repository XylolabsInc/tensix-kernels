# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare causal conv1d with upstream device ops and an fp64 reference.

  python check_conv.py [small] [qwen] [--batch B] [--rows T] [--time N]

Run on ttsim or a device. The upstream references are _causal_conv1d_decode_t1 for one token
and _causal_conv1d_fir for multiple rows. x is a fused projection output [B, T, W], with the
C convolution channels first. The state is [B, 3, C]; four taps and bias are [1, 1, C], all bf16.
For T > 1, also check the window [s0, s1, s2, x0 .. x(T-1)] and compare output rows bit for bit
with T single-token calls from the same state. The shifted state must match upstream exactly.
Upstream rounds each arithmetic step to bf16; this kernel accumulates in fp32.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_conv
from check import dev

CASES = {"small": (512, 1024), "qwen": (10240, 16480)}  # (channels C, fused projection width W)


def bf(t):
    return t.to(torch.bfloat16).float()


def make(name, B, T, seed=5):
    C, W = CASES[name]
    torch.manual_seed(seed)
    x = bf(torch.randn(B, T, W))
    s = bf(torch.randn(B, 3, C))
    w = [bf(torch.randn(1, 1, C) * 0.5) for _ in range(4)]
    b = bf(torch.randn(1, 1, C) * 0.1)
    return C, x, s, w, b


def reference(C, x, s, w, b):
    """fp64: the output, the shifted state and the window."""
    win = torch.cat([s, x[..., :C]], 1)
    wd = win.double()
    T = x.shape[1]
    outs = []
    for t in range(T):
        acc = b.double() + sum(w[k].double() * wd[:, t + k : t + k + 1] for k in range(4))
        outs.append(torch.nn.functional.silu(acc))
    return torch.cat(outs, 1), win[:, T : T + 3], win


def upstream(device, C, x, s, w, b):
    """Upstream's conv on the device (T = 1: the decode one, else the FIR one), or None where its
    module is missing: (out, new state)."""
    try:
        from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet as m
    except ImportError:
        return None
    tx = ttnn.to_layout(dev(x, ttnn.bfloat16, device)[:, :, :C], ttnn.TILE_LAYOUT)
    taps = [dev(t, ttnn.bfloat16, device) for t in w]
    tb, ts = dev(b, ttnn.bfloat16, device), dev(s, ttnn.bfloat16, device)
    if x.shape[1] == 1:
        out, new = m._causal_conv1d_decode_t1(tx, ts, 4, device, weight_taps=taps, bias_dev=tb)
    else:
        out, new = m._causal_conv1d_fir(
            tx, None, None, 4, device, conv_state=ts, weight_taps=taps, bias_dev=tb, valid_len=None
        )
    return ttnn.to_torch(out).float(), ttnn.to_torch(new).float()


def stat(name, ref, got):
    ref, got = ref.double().flatten(), got.double().flatten()
    mad = (got - ref).abs().max().item()
    rel = ((got - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item() if mad else 1.0
    tag = " (bit-identical)" if mad == 0 else ""
    return f"  {name}: max abs {mad:.3e}, rel {rel:.3e}, PCC {pcc:.7f}{tag}"


SLOTS = [0, 0]  # slot comparisons, bit-identical ones (--slots)


def slot_line(line):
    SLOTS[0] += 1
    SLOTS[1] += "(bit-identical)" in line
    return line


def run(device, name, a):
    B, T = a.batch, a.rows
    C, x, s, w, b = make(name, B, T)
    tx, ts = dev(x, ttnn.bfloat16, device), dev(s, ttnn.bfloat16, device)
    tw, tb = [dev(t, ttnn.bfloat16, device) for t in w], dev(b, ttnn.bfloat16, device)
    win = dev(torch.zeros(B, 3 + T, C), ttnn.bfloat16, device) if T > 1 else None
    # --slots checks the conv state after each row a < T - 1 in slot a.
    slots = (
        [dev(torch.full((B, 3, C), 7.0), ttnn.bfloat16, device) for _ in range(T - 1)]
        if a.slots
        else []
    )
    t0 = time.monotonic()
    out = gdn_conv.gated_conv_step(tx, ts, tw, tb, channels=C, window=win, slots=slots)
    ttnn.synchronize_device(device)
    first = time.monotonic() - t0
    got_o, got_s = ttnn.to_torch(out).float(), ttnn.to_torch(ts).float()
    ref_o, ref_s, ref_w = reference(C, x, s, w, b)
    print(f"[conv {name}] B {B} T {T} C {C}: first call {first:.2f} s", flush=True)
    print(stat("out vs fp64", ref_o, got_o), flush=True)
    print(stat("state vs the shift", ref_s, got_s), flush=True)
    if win is not None:
        print(stat("window", ref_w, ttnn.to_torch(win).float()), flush=True)
    for k, sl in enumerate(slots):
        print(
            slot_line(
                stat(
                    f"slot {k} vs window rows {k + 1}..{k + 3}",
                    ref_w[:, k + 1 : k + 4],
                    ttnn.to_torch(sl).float(),
                )
            ),
            flush=True,
        )
    if win is not None:
        # The decode steps: T calls at T = 1 on the same starting state.
        ts1 = dev(s, ttnn.bfloat16, device)
        rows = []
        for t in range(T):
            xt = dev(x[:, t : t + 1], ttnn.bfloat16, device)
            o1 = gdn_conv.gated_conv_step(xt, ts1, tw, tb, channels=C)
            rows.append(ttnn.to_torch(o1).float())
            if t < len(slots):
                print(
                    slot_line(
                        stat(
                            f"slot {t} vs the state after step {t}",
                            ttnn.to_torch(ts1).float(),
                            ttnn.to_torch(slots[t]).float(),
                        )
                    ),
                    flush=True,
                )
        print(stat("out vs T steps of one token", torch.cat(rows, 1), got_o), flush=True)
        print(stat("state vs T steps of one token", ttnn.to_torch(ts1).float(), got_s), flush=True)
    up = upstream(device, C, x, s, w, b)
    if up is not None:
        what = "t1" if T == 1 else "FIR"
        print(stat(f"out vs upstream's {what} conv", up[0], got_o), flush=True)
        print(stat("upstream's out vs fp64", ref_o, up[0]), flush=True)
        print(stat("state vs upstream's", up[1], got_s), flush=True)
    if a.time:
        times = []
        for _ in range(a.time):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            o2 = gdn_conv.gated_conv_step(tx, ts, tw, tb, channels=C, window=win, slots=slots)
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
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rows", type=int, default=1, help="T tokens (T > 1: a verify's rows)")
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument(
        "--slots", action="store_true", help="T > 1: also the T - 1 conv slots"
    )
    a = ap.parse_args()
    a.cases = a.cases or ["small"]  # not argparse's default: newer Pythons check it against choices
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    print(f"device open {time.strftime('%F %T %Z')}", flush=True)
    try:
        for name in a.cases:
            run(device, name, a)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T %Z')}", flush=True)
    if a.slots:
        print(f"check_conv: slots {SLOTS[1]} of {SLOTS[0]} bit-identical", flush=True)


if __name__ == "__main__":
    import _verdict

    _verdict.watch()
    main()
    sys.exit(_verdict.finish("check_conv"))
