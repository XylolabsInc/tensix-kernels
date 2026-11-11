# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""A Gated DeltaNet layer's recurrence on the device: a prompt through the chunked prefill scan, then decode steps
through the one-row kernel from the state the prompt left, checked against the exact float64 recurrence.

  python examples/gated_deltanet.py [--prompt 64] [--decode 4]

Shapes: 2 key heads and 4 value heads of 128 (each key head serves 2 value heads), one sequence. q and k are
L2-normalized inside the kernels; g is the log of each step's decay and beta the write strength, as in the gated delta
rule (S' = exp(g) S + beta k^T (v - k exp(g) S), o = q S')."""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_decode  # noqa: E402
import gdn_scan  # noqa: E402

H, HV, K, V = 2, 4, 128, 128
TILE = 32


def put(t, dtype, device):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def rel(ref, got):
    ref, got = ref.double().flatten(), got.double().flatten()
    return 100 * ((got - ref).norm() / ref.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--decode", type=int, default=4)
    a = ap.parse_args()
    T, N = a.prompt, a.decode
    g0 = torch.Generator().manual_seed(0)
    # the inputs of the whole sequence: the prompt's T rows, then N decode rows (bf16 values, as the device holds them)
    q = torch.randn(1, T + N, H, K, generator=g0).bfloat16().float()
    k = torch.randn(1, T + N, H, K, generator=g0).bfloat16().float()
    v = (0.5 * torch.randn(1, T + N, HV, V, generator=g0)).bfloat16().float()
    g = -torch.rand(1, T + N, HV, generator=g0) * 0.5
    beta = torch.sigmoid(torch.randn(1, T + N, HV, generator=g0))
    ref_o, ref_s = gdn_scan.torch_recurrent(q, k, v, g, beta, None)  # float64, the whole sequence

    device = ttnn.open_device(device_id=0)
    try:
        # 1. the prompt: one call of the chunked prefill scan, from an empty state
        o, s = gdn_scan.gated_delta_rule(
            put(q[:, :T].reshape(1, T, H * K), ttnn.bfloat16, device),
            put(k[:, :T].reshape(1, T, H * K), ttnn.bfloat16, device),
            put(v[:, :T].reshape(1, T, HV * V), ttnn.bfloat16, device),
            put(g[:, :T], ttnn.float32, device),
            put(beta[:, :T], ttnn.float32, device),
            None,
            num_heads=H,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            out_dtype=ttnn.float32,
        )
        prompt_o = ttnn.to_torch(o).float().reshape(1, T, HV, V)
        print(f"prompt of {T}: o {rel(ref_o[:, :T], prompt_o):.3f} % from the float64 recurrence", flush=True)

        # 2. decode: the state the prompt left, [1, HV, K, V] in float32, updated in place by each step
        state = put(ttnn.to_torch(s).float().reshape(1, HV, K, V), ttnn.float32, device)
        for i in range(N):
            r = T + i
            # one row: q | k | v side by side, then g and beta in a second tensor at their own columns
            x = torch.cat([q[:, r].reshape(1, 1, -1), k[:, r].reshape(1, 1, -1), v[:, r].reshape(1, 1, -1)], -1)
            gb = torch.zeros(1, 1, TILE)
            gb[..., :HV] = g[:, r]
            gb[..., HV : 2 * HV] = beta[:, r]
            o = gdn_decode.gated_delta_decode(
                put(x, ttnn.bfloat16, device),
                tgb := put(gb, ttnn.float32, device),
                tgb,
                state,
                num_heads=H,
                num_v_heads=HV,
                cols=(0, H * K // TILE, 2 * H * K // TILE),
                a_col=0,
                b_col=HV,
                out_dtype=ttnn.float32,
            )
            step_o = ttnn.to_torch(o).float().reshape(1, 1, HV, V)
            print(f"decode {i}: o {rel(ref_o[:, r : r + 1], step_o):.3f} %", flush=True)
        got_s = ttnn.to_torch(state).float().reshape(1, HV, K, V)
        print(f"state after the last step: {rel(ref_s, got_s):.3f} %", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
