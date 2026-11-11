# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Check batched page copies against their sources, bit for bit, on ttsim or a device.

  python check_commit.py [--layers 4] [--ncores N]

Use recurrent states [1, 48, 128, 128] in bf16 and fp32 and conv states [1, 3, 10240] in bf16.
Copy several layers in one launch, then reuse the same buffers with new data to check replay
of the cached pair table.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_commit


def put(t, dtype, device):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def bits(t):
    x = ttnn.to_torch(t).contiguous()
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ncores", type=int, default=None)
    a = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    print(f"device open {time.strftime('%F %T')}", flush=True)
    fails = 0
    try:
        shapes = [
            ((1, 48, 128, 128), ttnn.bfloat16),
            ((1, 3, 10240), ttnn.bfloat16),
            ((1, 16, 128, 128), ttnn.float32),
        ]
        pairs = []
        for _ in range(a.layers):
            for shp, dt in shapes:
                pairs.append((put(torch.zeros(shp), dt, device), put(torch.zeros(shp), dt, device)))
        g = torch.Generator().manual_seed(3)
        for rnd in range(2):
            for src, _ in pairs:
                shp = list(src.shape)
                ttnn.copy(put(torch.randn(*shp, generator=g), src.dtype, device), src)
            gdn_commit.commit(pairs, ncores=a.ncores)
            ttnn.synchronize_device(device)
            print(f"[commit round {rnd}] {len(pairs)} pairs: kernel done", flush=True)
            same = [torch.equal(bits(s), bits(d)) for s, d in pairs]
            bad = same.count(False)
            for i, ((s, d), ok) in enumerate(zip(pairs, same)):
                if not ok:
                    bs, bd = bits(s), bits(d)
                    nd = int((bs != bd).sum())
                    first = (bs != bd).flatten().nonzero()[0].item()
                    print(
                        f"  pair {i} {list(s.shape)} {s.dtype}: {nd} of {bs.numel()} differ, the first at flat {first}",
                        flush=True,
                    )
            print(f"  {len(pairs) - bad} of {len(pairs)} pairs bit-identical", flush=True)
            fails += bad
    finally:
        ttnn.close_device(device)
        print(f"device closed {time.strftime('%F %T')}", flush=True)
    print(f"check_commit: {fails} mismatch(es)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
