# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Compare the Mamba-2 mixer with a host fp64 reference using the device's stored weights.

  python check_mixer.py [--prompt 64] [--decode 4] [--weights bf16|bfp8|bfp4|mixed] [--seeds 1,2]

Check in_proj, causal convolution, SSM recurrence, gated RMSNorm, and out_proj. Weights and
inputs are random at mamba2.py's dimensions. Process --prompt tokens from an empty cache in
calls of at most 29 tokens, then run --decode single-token steps on the resulting cache.
Run on ttsim or a device. mixed uses bfloat4_b for in_proj and bfloat8_b for out_proj.

Every prompt and decode output must have relative L2 error at most 1.5% and PCC at least
0.9999. The SSM state after the prompt and final decode step must have relative L2 error
at most 1%. Exit 1 if a check fails.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import mamba2 as M  # noqa: E402

DTYPES = {
    "bf16": (ttnn.bfloat16,) * 2,
    "bfp8": (ttnn.bfloat8_b,) * 2,
    "bfp4": (ttnn.bfloat4_b,) * 2,
    "mixed": (ttnn.bfloat4_b, ttnn.bfloat8_b),
}
BAR_OUT, BAR_PCC, BAR_STATE = 0.015, 0.9999, 0.01
bad = 0


def cmp(name, ref, got, bar=None, pcc_bar=None):
    """Print comparison metrics and count failures against error and PCC thresholds."""
    global bad
    ro = False
    ref, got = ref.double().flatten(), got.double().flatten()
    finite = bool(torch.isfinite(got).all())
    rel = ((got - ref).norm() / ref.norm()).item() if finite else float("inf")
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item() if finite else float("nan")
    fault = not finite or (not ro and (not pcc >= 0.999 or rel > 0.05))
    miss = not ro and (
        (bar is not None and not rel <= bar) or (pcc_bar is not None and not pcc >= pcc_bar)
    )
    bad += fault or miss
    ratio = (got.norm() / ref.norm()).item() if finite else float("nan")
    tag = "  FAIL" if fault else ("  MISS" if miss else "")
    print(
        f"  {name}: rel {100 * rel:.3f} %, max abs {(got - ref).abs().max():.3e}, PCC {pcc:.7f}, norm {ratio:.5f}{tag}",
        flush=True,
    )


def host(t, T):
    return ttnn.to_torch(t).float()[:, :T]


def run_calls(dev, mx, cache, hid):
    """hid [1, T, 2688] float: the calls of up to 29 tokens; each stage's rows on the host, concatenated."""
    T = hid.shape[1]
    got = {k: [] for k in ("z", "xd", "conv", "y", "yn", "out")}
    for t0 in range(0, T, M.MAX_T):
        n = min(M.MAX_T, T - t0)
        h = ttnn.from_torch(
            hid[:, t0 : t0 + n].contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mx.mc,
        )
        out, parts = mx(h, cache, keep=True)
        ttnn.synchronize_device(dev)
        got["out"].append(host(out, n))
        for k in ("z", "xd", "conv", "y", "yn"):
            got[k].append(host(parts[k], n))
        for t in (h, out, *parts.values()):
            ttnn.deallocate(t)
    return {k: torch.cat(v, 1) for k, v in got.items()}


class HostMixer:
    """Evaluate the mixer token by token in float64 using weights read back from the device.

    Use the same bf16 inputs as the device. Comparing against the stored weights separates
    device arithmetic error from weight quantization error."""

    def __init__(self, w, mx, floor):
        self.w_in = torch.cat(
            [ttnn.to_torch(mx.w_z), ttnn.to_torch(mx.w_xd)], 1
        ).double()  # [2688, 10304]
        self.w_out = ttnn.to_torch(mx.w_out).double()  # [4096, 2688]
        self.cw = w["mixer.conv1d.weight"].double()[:, 0, :]  # [C, 4], oldest tap first
        self.cb = w["mixer.conv1d.bias"].double()
        self.A = -torch.exp(w["mixer.A_log"].double())
        self.D = w["mixer.D"].double()
        self.dtb = w["mixer.dt_bias"].double()
        self.nw = w["mixer.norm.weight"].double()
        self.eps, self.floor = float(w["config"]["eps"]), floor
        self.reset()

    def reset(self):
        self.win = torch.zeros(3, M.C, dtype=torch.float64)  # the last 3 conv inputs
        self.S = torch.zeros(M.HV, M.P, M.N, dtype=torch.float64)

    def __call__(self, hid):
        """hid [1, T, 2688] (bf16 values): out [1, T, 2688]."""
        F = torch.nn.functional
        outs = []
        for row in hid[0].double():
            p = row @ self.w_in
            z, xbc, dt_raw = p[: M.INTER], p[M.INTER : M.INTER + M.C], p[M.INTER + M.C :]
            win = torch.cat([self.win, xbc[None]], 0)  # [4, C]
            self.win = win[1:]
            xc = F.silu((win.T * self.cw).sum(-1) + self.cb)
            x = xc[: M.INTER].reshape(M.HV, M.P)
            B = (
                xc[M.INTER : M.INTER + M.H * M.N]
                .reshape(M.H, M.N)
                .repeat_interleave(M.HV // M.H, 0)
            )
            Cm = xc[M.INTER + M.H * M.N :].reshape(M.H, M.N).repeat_interleave(M.HV // M.H, 0)
            dt = F.softplus(dt_raw + self.dtb)
            if self.floor:
                dt = dt.clamp(min=self.floor)
            self.S = (
                self.S * torch.exp(self.A * dt)[:, None, None]
                + (dt[:, None] * x)[:, :, None] * B[:, None, :]
            )
            y = (torch.einsum("hpn,hn->hp", self.S, Cm) + self.D[:, None] * x).flatten()
            g = (y * F.silu(z)).reshape(M.H, M.GROUP)
            yn = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)).flatten() * self.nw
            outs.append(yn @ self.w_out)
        return torch.stack(outs)[None]


def synthetic(seed):
    """Random weights at the mixer's shapes, scaled as a freshly initialized Mamba-2 layer."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = torch.log(torch.tensor(1e-3)), torch.log(torch.tensor(0.1))
    dt = torch.exp(torch.rand(M.HV, generator=g) * (hi - lo) + lo)
    return {
        "mixer.in_proj.weight": (torch.randn(M.INTER + M.XD, M.HIDDEN, generator=g) * 0.02).bfloat16(),
        "mixer.out_proj.weight": (torch.randn(M.HIDDEN, M.INTER, generator=g) * 0.02).bfloat16(),
        "mixer.A_log": torch.log(torch.rand(M.HV, generator=g) * 15 + 1),
        "mixer.D": torch.ones(M.HV),
        "mixer.dt_bias": (dt + torch.log(-torch.expm1(-dt))).bfloat16(),
        "mixer.conv1d.weight": (torch.randn(M.C, 1, 4, generator=g) * 0.5).bfloat16(),
        "mixer.conv1d.bias": (torch.randn(M.C, generator=g) * 0.1).bfloat16(),
        "mixer.norm.weight": (1 + 0.1 * torch.randn(M.INTER, generator=g)).bfloat16(),
        "config": {"eps": 1e-5, "n_groups": M.H, "time_step_min": 1e-3},
    }


def device_state(cache):
    return ttnn.to_torch(cache[1]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--weights", default="bf16", choices=list(DTYPES))
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T %Z')}, weights {a.weights}", flush=True)
    try:
        for seed in (int(s) for s in a.seeds.split(",")):
            w = synthetic(seed)
            wi, wo = DTYPES[a.weights]
            mx = M.Mamba2Mixer(dev, w, in_proj_dtype=wi, out_proj_dtype=wo)
            hm = HostMixer(w, mx, mx.floor)
            cache = mx.new_cache()
            g = torch.Generator().manual_seed(seed + 100)
            hid = torch.randn(1, a.prompt, M.HIDDEN, generator=g).bfloat16().float()
            got = run_calls(dev, mx, cache, hid)
            cmp(f"seed {seed} prompt of {a.prompt} out", hm(hid.double()), got["out"], BAR_OUT, BAR_PCC)
            cmp(f"seed {seed} SSM state after the prompt", hm.S.transpose(-1, -2), device_state(cache), BAR_STATE)
            for i in range(a.decode):
                h = torch.randn(1, 1, M.HIDDEN, generator=g).bfloat16().float()
                got = run_calls(dev, mx, cache, h)
                cmp(f"seed {seed} decode {i} out", hm(h.double()), got["out"], BAR_OUT, BAR_PCC)
            cmp(f"seed {seed} SSM state after the last step", hm.S.transpose(-1, -2), device_state(cache), BAR_STATE)
            for t in cache:
                ttnn.deallocate(t)
            mx.free()
    finally:
        ttnn.close_device(dev)
    print(f"check_mixer: {'PASS' if bad == 0 else f'{bad} FAIL or MISS'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
