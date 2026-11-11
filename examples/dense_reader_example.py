# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""dense_reader: one token through a bfloat8_b weight-streaming matmul, against torch.

  python -s examples/dense_reader_example.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_reader import linear  # noqa: E402

device = ttnn.open_device(device_id=0)
try:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(2048, 1024, generator=g) * 0.02  # [out, in]
    x = torch.randn(1, 1024, generator=g).to(torch.bfloat16)
    tw, _ = linear.native_weight(device, w, ttnn.bfloat8_b)  # [1, 1, in, out], DRAM width-sharded
    tx = ttnn.from_torch(x.reshape(1, 1, 1, 1024), layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(linear.matmul(tx, tw, None, fidelity="hifi4")).float().reshape(-1, 2048)[:1]
    wk = ttnn.to_torch(tw).float().reshape(1024, -1)[:, :2048]  # the weight's own bfloat8_b values
    ref = x.double() @ wk.double()
    rel = ((y.double() - ref).norm() / ref.norm()).item()
    print(f"dense_reader example: y = x @ w for one token, relative error {100 * rel:.3f} % against torch")
    assert rel < 0.01, rel
finally:
    ttnn.close_device(device)
