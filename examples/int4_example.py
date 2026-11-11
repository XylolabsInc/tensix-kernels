# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""int4: one token through an int4-weight matmul with a scale per 32 inputs, against torch.

  python -s examples/int4_example.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from int4 import w4a16  # noqa: E402

N, K = 1024, 1024
device = ttnn.open_device(device_id=0)
try:
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(-7, 8, (N, K), generator=g).to(torch.int8)  # q in -7..7, [out, in]
    scales = (torch.rand(N, K // 32, generator=g) * 0.01 + 0.001).to(torch.bfloat16)  # one per 32 inputs
    x = torch.randn(1, K, generator=g).to(torch.bfloat16)
    wq = w4a16.int4_weight(device, codes)
    ts = w4a16.group_scales(device, scales)
    tx = ttnn.from_torch(x.reshape(1, 1, 1, K), layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(w4a16.w4a16_linear(tx, wq, ts, fidelity="hifi4")).float().reshape(-1, N)[:1]
    ref = w4a16.reference(x, codes, scales)  # x @ (q * s).T in float32
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"int4 example: y = x @ (q * s).T for one token, relative error {100 * rel:.3f} % against torch")
    assert rel < 0.01, rel
finally:
    ttnn.close_device(device)
