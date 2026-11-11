# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""ternary_matmul: one token through a ternary-weight matmul (trits -1, 0, +1 with a scale per 128 inputs), against
torch.

  python -s examples/ternary_matmul_example.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ternary_matmul import t2  # noqa: E402

N, K, G = 1024, 1024, 128
device = ttnn.open_device(device_id=0)
try:
    g = torch.Generator().manual_seed(0)
    trits = torch.randint(-1, 2, (N, K), generator=g).to(torch.int8)  # [out, in]
    scales = (0.01 + 0.02 * torch.rand(N, K // G, generator=g)).to(torch.float16)
    x = torch.randn(1, K, generator=g).to(torch.bfloat16)
    w = t2.ternary_weight(device, trits)  # BFP2_b tiles, DRAM width-sharded
    ts = t2.group_scales(device, scales)
    tx = ttnn.from_torch(x.reshape(1, 1, 1, K), layout=ttnn.TILE_LAYOUT, device=device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    y = ttnn.to_torch(t2.t2_linear(tx, w, ts, G, fidelity="hifi4")).float().reshape(-1, N)[:1]
    ref = t2.reference(x, trits, scales, G)  # x @ (q * bf16(s)).T in float32
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"ternary_matmul example: one token, relative error {100 * rel:.3f} % against torch")
    assert rel < 0.005, rel
finally:
    ttnn.close_device(device)
