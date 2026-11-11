# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""moe: one token through a 32-expert, top-4 MoE layer (route, gate_up, down on the device), against a float64
reference in torch.

  python -s examples/moe_example.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe import moe8  # noqa: E402

E, H, I, K = 32, 1024, 512, 4
BF = torch.bfloat16
device = ttnn.open_device(device_id=0)
try:
    g = torch.Generator().manual_seed(0)
    # experts on the BFP4_b grid, so the packed weights hold them exactly: gate_up [E, H, 2I], down [E, I, H]
    gu = torch.stack([torch.randn(H, 2 * I, generator=g) * 0.02 for _ in range(E)])
    dn = torch.stack([torch.randn(I, H, generator=g) * 0.02 for _ in range(E)])

    def on_grid(m):
        return torch.from_numpy(
            moe8.bfp4_values(moe8.bfp4_tiles(m)).transpose(0, 2, 1, 3).reshape(m.shape).copy()
        ).to(BF)

    gu = torch.stack([on_grid(m) for m in gu])
    dn = torch.stack([on_grid(m) for m in dn])
    P = moe8.pack(device, gu, dn)
    x = torch.randn(1, H, generator=g).to(BF)
    logits = torch.randn(1, E, generator=g) * 3  # the router's output for this token
    tl = ttnn.from_torch(logits.reshape(1, 1, 1, E), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tid = ttnn.from_torch(torch.zeros(1, 1, 1, K, dtype=torch.int16), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                          device=device)
    twt = ttnn.from_torch(torch.zeros(1, 1, 1, K, dtype=BF), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=device)
    moe8.route(tl, tid, twt, K, kind="softmax")  # the top 4 and their softmax weights, written in place
    tx = ttnn.from_torch(x.reshape(1, 1, 1, H), layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    th = moe8.gate_up(tx, P, tid, twt, K, act="silu", fidelity="hifi4")
    y = ttnn.to_torch(moe8.down(th, P, tid, K, fidelity="hifi4")).float().reshape(-1, H)[:1]
    ids = (ttnn.to_torch(tid).reshape(-1)[:K].to(torch.int64) & 0xFFFF).tolist()
    wts = ttnn.to_torch(twt).reshape(-1)[:K].float()
    ref, _ = moe8.reference(x.float(), gu, dn, ids, wts, "silu")
    rel = ((y.double() - ref).norm() / ref.norm()).item()
    print(f"moe example: experts {ids}, relative error {100 * rel:.3f} % against the float64 reference")
    assert sorted(ids) == sorted(torch.topk(logits[0], K).indices.tolist())
    assert rel < 0.01, rel
finally:
    ttnn.close_device(device)
