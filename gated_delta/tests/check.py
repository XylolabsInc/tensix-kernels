# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare chunked gated delta-rule prefill with an fp64 recurrence on ttsim or a device.

  python check.py smoke                 # 1 head, 64 tokens
  python check.py small qwen            # named cases (CASES), in order
  python check.py qwen --time 5         # report the median of 5 timed calls
  python check.py qwen --native         # compare upstream's gated_delta_attn_seq path on ttsim

Inputs use random Qwen3.8-27B-shaped DeltaNet tensors. q, k, and v come from randn; the kernel
L2-normalizes q and k. beta = sigmoid(randn), and g = -exp(A) * softplus(a) is negative.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_scan

# name: (B, T, H, HV, K, V, initial state)
CASES = {
    "smoke": (1, 64, 1, 1, 128, 128, False),
    "chunks": (1, 200, 1, 1, 128, 128, True),
    "gqa": (1, 96, 2, 6, 128, 128, True),
    "small": (1, 100, 2, 6, 128, 128, True),
    "qwen": (1, 256, 16, 48, 128, 128, True),
    "qwen20": (1, 20, 16, 48, 128, 128, False),
    "qwen1k": (1, 1024, 16, 48, 128, 128, True),
    # one head with a constant strong decay: the chunk's log decay reaches -64 * G_STRONG
    "decay": (1, 128, 1, 1, 128, 128, False),
}
G_STRONG = 3.0


def inputs(B, T, H, HV, K, V, state, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, HV))
    A = torch.rand(HV) * 2 - 1.5  # A_log
    g = -A.exp() * torch.nn.functional.softplus(torch.randn(B, T, HV) * 0.8 - 1.0)
    s0 = torch.randn(B, HV, K, V) * 0.1 if state else None
    if (B, T, H, HV) == (1, 128, 1, 1):
        g = torch.full_like(g, -G_STRONG)
    # bf16 as the model hands them over; the reference sees the same rounded values.
    q, k, v = (t.to(torch.bfloat16).float() for t in (q, k, v))
    return q, k, v, g, beta, s0


def dev(t, dtype, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# name: (DST, BLOCK_INV) of gdn_scan, switched between calls in one run (--variants)
VARIANTS = {"dst1": (1, True), "dst4": (4, True), "full": (1, False), "full-dst4": (4, False)}


def run_case(device, name, a, variant=None):
    if variant is not None:
        gdn_scan.DST, gdn_scan.BLOCK_INV = VARIANTS[variant]
    B, T, H, HV, K, V, state = CASES[name]
    q, k, v, g, beta, s0 = inputs(B, T, H, HV, K, V, state)
    ref_o, ref_s = gdn_scan.torch_recurrent(q, k, v, g, beta, s0)
    tq = dev(q.reshape(B, T, H * K), ttnn.bfloat16, device)
    tk = dev(k.reshape(B, T, H * K), ttnn.bfloat16, device)
    tv = dev(v.reshape(B, T, HV * V), ttnn.bfloat16, device)
    tg = dev(g, ttnn.float32, device)
    tb = dev(beta, ttnn.float32, device)
    ts = None if s0 is None else dev(s0.reshape(B * HV, K, V), ttnn.float32, device)
    kw = {
        "num_heads": H,
        "num_v_heads": HV,
        "head_k_dim": K,
        "head_v_dim": V,
        "out_dtype": ttnn.float32,
    }
    t0 = time.monotonic()
    o, s = gdn_scan.gated_delta_rule(tq, tk, tv, tg, tb, ts, **kw)
    ttnn.synchronize_device(device)
    first = time.monotonic() - t0
    got_o = ttnn.to_torch(o).float().reshape(B, T, HV, V)
    got_s = ttnn.to_torch(s).float().reshape(B, HV, K, V)
    print(
        f"[{name}] B {B} T {T} H {H} HV {HV} DST {gdn_scan.DST} block {int(gdn_scan.BLOCK_INV)}: "
        f"first call {first:.2f} s (JIT included)",
        flush=True,
    )
    res = {}
    for label, r, x in (("o", ref_o, got_o), ("state", ref_s, got_s)):
        line, m = gdn_scan.compare(f"  {label}", r, x)
        print(line, flush=True)
        res[label] = m
    # The worst head of o.
    per = [gdn_scan.compare("", ref_o[:, :, h], got_o[:, :, h])[1]["rel"] for h in range(HV)]
    print(
        f"  o per head: worst rel {max(per) * 100:.3f} % (head {per.index(max(per))})", flush=True
    )
    if a.detail:
        print("  o rel % by head: " + " ".join(f"{h}:{r * 100:.2f}" for h, r in enumerate(per)))
        hw = per.index(max(per))
        for c0 in range(0, T, gdn_scan.CHUNK):
            sl = slice(c0, min(T, c0 + gdn_scan.CHUNK))
            r = gdn_scan.compare("", ref_o[:, sl, hw], got_o[:, sl, hw])[1]
            print(f"  head {hw} tokens {sl.start}-{sl.stop - 1}: rel {r['rel'] * 100:.3f} %")
        print(f"  head {hw}: mean g {g[..., hw].mean():.3f}, min g {g[..., hw].min():.3f}")
    if a.time:
        ts_ = []
        for _ in range(a.time):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            o2, s2 = gdn_scan.gated_delta_rule(tq, tk, tv, tg, tb, ts, **kw)
            ttnn.synchronize_device(device)
            ts_.append(time.perf_counter() - t0)
            ttnn.deallocate(o2)
            ttnn.deallocate(s2)
        ts_.sort()
        print(
            f"  time per call (eager: host descriptor and dispatch included): median "
            f"{ts_[len(ts_) // 2] * 1e3:.2f} ms, min {ts_[0] * 1e3:.2f} ms ({a.time} calls)",
            flush=True,
        )
    if a.native:
        native(device, name, q, k, v, g, beta, s0, ref_o, ref_s)
    if a.adapter:
        adapter(device, name, q, k, v, g, beta, s0, ref_o, ref_s)
    if a.gnorm:
        gated_norm(device, tq, tk, tv, tg, tb, ts, kw, ref_o, ref_s)
    if a.rawgates:
        raw_gates(device, q, k, v, s0, tq, tk, tv, ts, kw)
    return res


def raw_gates(device, q, k, v, s0, tq, tk, tv, ts, kw):
    """g and beta made in the kernel (raw_gates) from raw a and b laid out as in the fused projection output: a
    from a tile boundary, b HV columns later (not tile-aligned), in one bf16 tensor."""
    B, T, HV, _ = v.shape
    torch.manual_seed(2)
    A = torch.rand(HV) * 2 - 1.5
    a_neg = (-A.exp()).to(torch.bfloat16).float()
    dt = (0.5 * torch.randn(HV)).to(torch.bfloat16).float()
    a_raw = (torch.randn(B, T, HV) * 0.8 - 1.0 - dt).to(torch.bfloat16).float()
    b_raw = torch.randn(B, T, HV).to(torch.bfloat16).float()
    width = -(-(2 * HV + 32) // 32) * 32
    ab = torch.zeros(B, T, width)
    ab[..., 32 : 32 + HV] = a_raw  # a from tile column 1
    ab[..., 32 + HV : 32 + 2 * HV] = b_raw
    g = a_neg * torch.nn.functional.softplus(a_raw + dt)
    beta = torch.sigmoid(b_raw)
    ref_o, ref_s = gdn_scan.torch_recurrent(q, k, v, g, beta, s0)
    tab = dev(ab, ttnn.bfloat16, device)
    o, s = gdn_scan.gated_delta_rule(
        tq, tk, tv, tab, tab, ts, **kw, g_col=32, beta_col=32 + HV, raw_gates=(a_neg, dt)
    )
    got_o = ttnn.to_torch(o).float().reshape(ref_o.shape)
    got_s = ttnn.to_torch(s).float().reshape(ref_s.shape)
    print(gdn_scan.compare("  raw gates o", ref_o, got_o)[0], flush=True)
    print(gdn_scan.compare("  raw gates state", ref_s, got_s)[0], flush=True)


a_detail = [False]


def gated_norm(device, tq, tk, tv, tg, tb, ts, kw, ref_o, ref_s, eps=1e-6):
    """The kernel's gated RMSNorm output stage (gate, norm_weight) against
    rms_norm(o, w, eps) * silu(gate) of the fp64 o."""
    B, T, HV, V = ref_o.shape
    torch.manual_seed(1)
    gate = (torch.randn(B, T, HV * V) * 2).to(torch.bfloat16).float()
    w = (1 + 0.2 * torch.randn(V)).to(torch.bfloat16).float()
    o = ref_o
    ref = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps) * w.double()
    ref = ref * torch.nn.functional.silu(gate.double().reshape(B, T, HV, V))
    tw = ttnn.from_torch(
        w.reshape(1, 1, V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    o2, s2 = gdn_scan.gated_delta_rule(
        tq,
        tk,
        tv,
        tg,
        tb,
        ts,
        **kw,
        gate=dev(gate, ttnn.bfloat16, device),
        norm_weight=tw,
        norm_eps=eps,
    )
    got = ttnn.to_torch(o2).float().reshape(B, T, HV, V)
    print(gdn_scan.compare("  gnorm o", ref, got)[0], flush=True)
    if a_detail[0]:
        for h in range(HV):
            row = []
            for c0 in range(0, T, 32):
                sl = slice(c0, min(T, c0 + 32))
                row.append(
                    f"{gdn_scan.compare('', ref[:, sl, h], got[:, sl, h])[1]['rel'] * 100:.1f}"
                )
            print(f"  gnorm head {h} rel % per 32 tokens: {' '.join(row)}", flush=True)
    got_s = ttnn.to_torch(s2).float().reshape(ref_s.shape)
    print(gdn_scan.compare("  gnorm state", ref_s, got_s)[0], flush=True)


def adapter(device, name, q, k, v, g, beta, s0, ref_o, ref_s):
    """The 4-D drop-in (gdn_scan.chunk_gated_delta_rule_seq_adapter) as the Qwen3.6 forward calls
    it: q and k repeated to the value heads, bf16 everywhere, the state in bf16."""
    rep = v.shape[2] // q.shape[2]
    o, s = gdn_scan.chunk_gated_delta_rule_seq_adapter(
        dev(q.repeat_interleave(rep, dim=2), ttnn.bfloat16, device),
        dev(k.repeat_interleave(rep, dim=2), ttnn.bfloat16, device),
        dev(v, ttnn.bfloat16, device),
        dev(beta, ttnn.bfloat16, device),
        dev(g, ttnn.bfloat16, device),
        initial_state=None if s0 is None else dev(s0, ttnn.bfloat16, device),
        device=device,
    )
    got_o = ttnn.to_torch(o).float().reshape(ref_o.shape)
    got_s = ttnn.to_torch(s).float().reshape(ref_s.shape)
    print(gdn_scan.compare(f"  adapter o ({gdn_scan.RELAYOUT})", ref_o, got_o)[0], flush=True)
    print(gdn_scan.compare(f"  adapter state ({gdn_scan.RELAYOUT})", ref_s, got_s)[0], flush=True)


def native(device, name, q, k, v, g, beta, s0, ref_o, ref_s):
    """Upstream's own path (ttnn_delta_rule_seq adapter + gated_delta_attn_seq), exact form
    (QWEN_GDN_DIAG_ALPHA=0), for comparison; ttsim only."""
    os.environ["QWEN_GDN_DIAG_ALPHA"] = "0"
    os.environ["QWEN_GDN_FP32_STATE"] = "1"
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
        chunk_gated_delta_rule_seq_adapter,
    )

    rep = v.shape[2] // q.shape[2]
    qe, ke = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    o, s = chunk_gated_delta_rule_seq_adapter(
        dev(qe, ttnn.bfloat16, device),
        dev(ke, ttnn.bfloat16, device),
        dev(v, ttnn.bfloat16, device),
        dev(beta, ttnn.bfloat16, device),
        dev(g, ttnn.float32, device),
        initial_state=None if s0 is None else dev(s0, ttnn.float32, device),
        device=device,
    )
    print(gdn_scan.compare(f"  native o ({name})", ref_o, ttnn.to_torch(o).float())[0], flush=True)
    print(
        gdn_scan.compare(
            f"  native state ({name})", ref_s, ttnn.to_torch(s).float().reshape(ref_s.shape)
        )[0],
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", choices=list(CASES))
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--adapter", action="store_true")
    ap.add_argument("--gnorm", action="store_true", help="also the gated RMSNorm output stage")
    ap.add_argument("--rawgates", action="store_true", help="also g and beta made in the kernel")
    ap.add_argument("--variants", default="", help=f"comma-separated, of {list(VARIANTS)}")
    a = ap.parse_args()
    a.cases = a.cases or ["smoke"]  # not argparse's default: newer Pythons check it against choices
    a_detail[0] = a.detail
    t0 = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    device = ttnn.open_device(device_id=0, **gdn_scan.worker_l1_kw())
    g = device.compute_with_storage_grid_size()
    print(
        f"device open {t0}, grid {g.x}x{g.y}, ttnn {getattr(ttnn, '__version__', '?')}", flush=True
    )
    try:
        for name in a.cases:
            for variant in [v for v in a.variants.split(",") if v] or [None]:
                run_case(device, name, a, variant)
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)


if __name__ == "__main__":
    import _verdict

    _verdict.watch()
    main()
    sys.exit(_verdict.finish("check"))
