# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""A Nemotron-H Mamba-2 mixer on the device, mamba2.Mamba2Mixer, with random weights at its dimensions: a prompt from
an empty cache (the conv state and the SSM state, both updated in place), then one-token decode steps on that cache.
Every output and the SSM state are checked against the same mixer in float64 on the host, which runs on the weights
read back from the device, so the comparison measures the device's arithmetic and not the weights' rounding.

  python examples/mamba2_mixer.py [--prompt 16] [--decode 2]

The random weights (scaled as a freshly initialized layer, in bf16) and the float64 mixer are those of
tests/check_mixer.py. A prompt longer than 29 tokens goes through in several calls, as mamba2.py says."""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import mamba2  # noqa: E402
from check_mixer import HostMixer, synthetic  # noqa: E402

OUT_BAR, STATE_BAR, PCC_BAR = 1.5, 1.0, 0.9999  # against float64: relative L2 error in % (output, state), PCC
results = []  # one bool per comparison


def close(ref, got, bar):
    """'REL % (PCC P)': got's relative L2 error and PCC against the float64 ref, recorded against the bars."""
    ref, got = ref.double().flatten(), got.double().flatten()
    rel = 100 * ((got - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    ok = rel <= bar and pcc >= PCC_BAR  # False on NaN
    results.append(ok)
    return f"{rel:.3f} % (PCC {pcc:.7f}){'' if ok else ' MISS'}"


def run(mx, cache, h):
    """One call of the mixer on the host rows h [1, n, 2688] (n at most 29), the cache updated in place: its output on
    the host."""
    t = ttnn.from_torch(
        h.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mx.device, memory_config=mx.mc
    )
    out = mx(t, cache)
    got = ttnn.to_torch(out).float()[:, : h.shape[1]]
    ttnn.deallocate(t)
    ttnn.deallocate(out)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=16)
    ap.add_argument("--decode", type=int, default=2)
    a = ap.parse_args()
    T, N = a.prompt, a.decode
    if T < 1 or N < 0:
        ap.error("--prompt at least 1, --decode at least 0")
    w = synthetic(1)  # host tensors keyed by the checkpoint's names, and its config
    g0 = torch.Generator().manual_seed(101)
    hid = torch.randn(1, T + N, mamba2.HIDDEN, generator=g0).bfloat16().float()  # the block's RMSNorm output

    device = ttnn.open_device(device_id=0)
    try:
        mx = mamba2.Mamba2Mixer(device, w)  # bf16 weights; the SSM state in float32
        ref = HostMixer(w, mx, mx.floor)  # float64, on the weights as the device holds them
        cache = mx.new_cache()  # [conv state [1, 3, 6144] bf16, SSM state [1, 64, 128, 64] float32], zeros

        # 1. the prompt, in calls of up to 29 tokens
        got = torch.cat([run(mx, cache, hid[:, t : min(t + mamba2.MAX_T, T)]) for t in range(0, T, mamba2.MAX_T)], 1)
        print(f"prompt of {T}: out {close(ref(hid[:, :T].double()), got, OUT_BAR)}", flush=True)
        # the host's state is [HV, P, N], the device's [1, HV, N, P]
        got_s = ttnn.to_torch(cache[1]).float()
        print(f"SSM state after the prompt: {close(ref.S.transpose(-1, -2), got_s, STATE_BAR)}", flush=True)

        # 2. decode: one token a call on the cache the prompt left
        for i in range(N):
            r = T + i
            got = run(mx, cache, hid[:, r : r + 1])
            print(f"decode {i}: out {close(ref(hid[:, r : r + 1].double()), got, OUT_BAR)}", flush=True)
        got_s = ttnn.to_torch(cache[1]).float()
        print(f"SSM state after the last step: {close(ref.S.transpose(-1, -2), got_s, STATE_BAR)}", flush=True)
        mx.free()
    finally:
        ttnn.close_device(device)
    bad = results.count(False)
    print(f"mixer: {len(results)} comparisons, {f'{bad} outside the bars' if bad else 'all within the bars'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
