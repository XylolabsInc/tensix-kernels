# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""lm_head_topk: token ids for two rows from a bfloat4_b device head, rescored and picked on the device, against a
bf16 head in torch.

  python -s examples/lm_head_topk_example.py
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lm_head_topk import lmhead  # noqa: E402

VOCAB, DIM, ROWS = 4096, 1024, 2
device = ttnn.open_device(device_id=0)
try:
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(VOCAB, DIM, generator=g) * 0.05).to(torch.bfloat16)  # the head [vocab, dim]
    x = torch.randn(ROWS, DIM, generator=g).to(torch.bfloat16)
    head = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=2, device_pick=True)
    tx = ttnn.from_torch(x.reshape(1, 1, ROWS, DIM), layout=ttnn.TILE_LAYOUT, device=device)
    got = head(tx, ROWS)
    z = (x.float() @ w.float().T).to(torch.bfloat16).float()  # bf16 logits; the lowest id among equal ones
    want = lmhead.pick_token(z, torch.arange(VOCAB).expand(ROWS, -1), VOCAB)
    print(f"lm_head_topk example: tokens {got.tolist()}, torch's bf16 head {want.tolist()}")
    assert torch.equal(got, want)
    head.deallocate()
finally:
    ttnn.close_device(device)
