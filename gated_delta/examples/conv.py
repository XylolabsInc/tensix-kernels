# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""The causal depthwise conv in front of a Gated DeltaNet layer's recurrence (4 taps, a bias, SiLU) with its rolling
state: a chunk of T tokens through gdn_conv.gated_conv_step in one call from a zero state, which also writes the
window it read (the 3 state rows, then the T tokens), then one-token steps on the state the chunk left. The outputs
are checked against a float64 conv over the whole sequence; the state and the window, which are copies of inputs, bit
for bit against the bf16 inputs.

  python examples/conv.py [--chunk 8] [--steps 3]

512 channels (in a model, the q, k and v channels of the fused projection's output), one sequence, all in bf16:
out_t = silu(bias + w0 x(t-3) + w1 x(t-2) + w2 x(t-1) + w3 x(t)) per channel, with x(t) = 0 before the first token."""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_conv  # noqa: E402

C = 512
MAX_T = 32 - 3  # gated_conv_step: the 3 state rows and the tokens share one tile's 32 rows
REL_BAR, PCC_BAR = 2.0, 0.9999  # against float64: relative L2 error in %, PCC
results = []  # one bool per comparison


def put(t, device):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def close(ref, got):
    """'REL % (PCC P)': got's relative L2 error and PCC against the float64 ref, recorded against the bars."""
    ref, got = ref.double().flatten(), got.double().flatten()
    rel = 100 * ((got - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    ok = rel <= REL_BAR and pcc >= PCC_BAR  # False on NaN
    results.append(ok)
    return f"{rel:.3f} % (PCC {pcc:.7f}){'' if ok else ' MISS'}"


def exact(name, ref, got):
    """'NAME: bit for bit' when the device tensor got holds the bf16 ref's bits, recorded as a comparison."""
    ref, got = ref.bfloat16().contiguous().view(torch.int16), ttnn.to_torch(got).contiguous().view(torch.int16)
    ok = ref.shape == got.shape and torch.equal(ref, got)
    results.append(ok)
    return f"{name}: {'bit for bit' if ok else 'DIFFERENT'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8, help="T tokens in the first call (1 .. 29)")
    ap.add_argument("--steps", type=int, default=3, help="one-token calls after it")
    a = ap.parse_args()
    T, N = a.chunk, a.steps
    if not 1 <= T <= MAX_T:
        ap.error(f"--chunk 1 to {MAX_T}")
    g0 = torch.Generator().manual_seed(0)
    x = torch.randn(1, T + N, C, generator=g0).bfloat16()
    taps = [(0.5 * torch.randn(1, 1, C, generator=g0)).bfloat16() for _ in range(4)]  # tap 3 multiplies x(t)
    bias = (0.1 * torch.randn(1, 1, C, generator=g0)).bfloat16()
    # float64 over the whole sequence: win = the zero state's 3 rows, then every token; out_t reads win[t .. t + 3]
    win = torch.cat([torch.zeros(1, 3, C, dtype=torch.bfloat16), x], 1)
    wd = win.double()
    ref = torch.nn.functional.silu(bias.double() + sum(taps[j].double() * wd[:, j : j + T + N] for j in range(4)))

    device = ttnn.open_device(device_id=0)
    try:
        state = put(torch.zeros(1, 3, C), device)  # the last three inputs, updated in place
        tt_taps, tt_bias = [put(t, device) for t in taps], put(bias, device)
        # 1. the chunk: T tokens in one call, the window [s0, s1, s2, x0 .. x(T-1)] written as well
        window = put(torch.zeros(1, 3 + T, C), device)
        out = gdn_conv.gated_conv_step(put(x[:, :T], device), state, tt_taps, tt_bias, channels=C, window=window)
        print(f"chunk of {T}: out {close(ref[:, :T], ttnn.to_torch(out)[:, :T])}", flush=True)
        print(exact("  the window", win[:, : 3 + T], window), flush=True)
        print(exact("  the state = the last three inputs", win[:, T : T + 3], state), flush=True)
        # 2. one token a call on the state the chunk left
        for i in range(N):
            r = T + i
            out = gdn_conv.gated_conv_step(put(x[:, r : r + 1], device), state, tt_taps, tt_bias, channels=C)
            print(f"step {i}: out {close(ref[:, r : r + 1], ttnn.to_torch(out)[:, :1])}", flush=True)
        print(exact("the state after the last step = the last three inputs", win[:, T + N :], state), flush=True)
    finally:
        ttnn.close_device(device)
    bad = results.count(False)
    print(f"conv: {len(results)} comparisons, {f'{bad} outside the bars' if bad else 'all within the bars'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
