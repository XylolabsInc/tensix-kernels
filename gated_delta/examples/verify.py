# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""The verify step of speculative decoding for a Gated DeltaNet layer: R rows (the last accepted token, then R - 1
draft tokens) through gdn_verify.gated_delta_verify in one call, which writes the state after each of rows 0 .. R - 2
to a slot and the state after the last row into the state. Then the rollback: with only the first k rows accepted,
gdn_commit copies slot k - 1 into the state, and decode goes on from there. Every row's o, every slot and the final
state are checked against the exact float64 recurrence run row by row; the rollback, a copy, bit for bit.

  python examples/verify.py [--rows 4] [--accept 2]

Shapes as in gated_deltanet.py: 2 key heads and 4 value heads of 128, one sequence, a random float32 state. g and
beta go in as they are (no raw gates), at columns 0 and HV of their tensor."""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gdn_commit  # noqa: E402
import gdn_decode  # noqa: E402
import gdn_scan  # noqa: E402
import gdn_verify  # noqa: E402

H, HV, K, V = 2, 4, 128, 128
TILE = 32
REL_BAR, PCC_BAR = 2.0, 0.9999  # against float64: relative L2 error in %, PCC
KW = {
    "num_heads": H,
    "num_v_heads": HV,
    "cols": (0, H * K // TILE, 2 * H * K // TILE),  # q, k, v by tile column in x
    "a_col": 0,
    "b_col": HV,
    "out_dtype": ttnn.float32,
}
results = []  # one bool per comparison


def put(t, dtype, device):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def bits(t):
    return ttnn.to_torch(t).contiguous().view(torch.int32)  # float32 tensors


def close(ref, got):
    """'REL % (PCC P)': got's relative L2 error and PCC against the float64 ref, recorded against the bars."""
    ref, got = ref.double().flatten(), got.double().flatten()
    rel = 100 * ((got - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    ok = rel <= REL_BAR and pcc >= PCC_BAR  # False on NaN
    results.append(ok)
    return f"{rel:.3f} % (PCC {pcc:.7f}){'' if ok else ' MISS'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--accept", type=int, default=2, help="rows accepted (1 .. rows - 1)")
    a = ap.parse_args()
    R, acc = a.rows, a.accept
    if not (2 <= R <= TILE and 1 <= acc < R):
        ap.error(f"--rows 2 to {TILE}, --accept 1 to rows - 1")
    g0 = torch.Generator().manual_seed(0)
    n = R + 1  # the R rows of the verify, then the row decoded after the rollback
    q = torch.randn(1, n, H, K, generator=g0).bfloat16().float()
    k = torch.randn(1, n, H, K, generator=g0).bfloat16().float()
    v = (0.5 * torch.randn(1, n, HV, V, generator=g0)).bfloat16().float()
    g = -torch.rand(1, n, HV, generator=g0) * 0.5
    beta = torch.sigmoid(torch.randn(1, n, HV, generator=g0))
    s0 = 0.1 * torch.randn(1, HV, K, V, generator=g0)

    def ref_row(r, s):
        """float64: row r's o and the state after it, from the state s."""
        sl = slice(r, r + 1)
        return gdn_scan.torch_recurrent(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], s)

    ref_o, ref_s, s = [], [], s0
    for r in range(R):
        o, s = ref_row(r, s)
        ref_o.append(o)
        ref_s.append(s)

    def rows(r0, r1):
        """Rows r0 .. r1 - 1 as the kernel reads them: x = q | k | v side by side; g and beta in a second tensor."""
        x = torch.cat([t[:, r0:r1].reshape(1, r1 - r0, -1) for t in (q, k, v)], -1)
        gb = torch.zeros(1, r1 - r0, TILE)
        gb[..., :HV] = g[:, r0:r1]
        gb[..., HV : 2 * HV] = beta[:, r0:r1]
        return put(x, ttnn.bfloat16, device), put(gb, ttnn.float32, device)

    device = ttnn.open_device(device_id=0)
    try:
        # 1. verify: R rows in one call; slot r gets the state after row r, and the state the one after the last row
        state = put(s0, ttnn.float32, device)
        slots = [put(torch.zeros(1, HV, K, V), ttnn.float32, device) for _ in range(R - 1)]
        x, gb = rows(0, R)
        o = gdn_verify.gated_delta_verify(x, gb, gb, state, slots, **KW)
        got_o = ttnn.to_torch(o).float().reshape(1, R, HV, V)
        print(f"{R} rows in one verify call from a random state, against float64 row by row:", flush=True)
        for r in range(R):
            where = f"slot {r}" if r < R - 1 else "the final state"
            got_s = ttnn.to_torch(slots[r] if r < R - 1 else state).float()
            print(f"row {r}: o {close(ref_o[r], got_o[:, r])}; {where}: {close(ref_s[r], got_s)}", flush=True)

        # 2. rollback: rows 0 .. acc - 1 accepted, so the next call starts from the state after row acc - 1
        gdn_commit.commit([(slots[acc - 1], state)])
        ok = torch.equal(bits(state), bits(slots[acc - 1]))
        results.append(ok)
        print(f"{acc} of {R} rows accepted: slot {acc - 1} into the state, {'bit for bit' if ok else 'DIFFERENT'}")

        # 3. decode goes on from the rolled-back state
        x, gb = rows(R, R + 1)
        o = gdn_decode.gated_delta_decode(x, gb, gb, state, **KW)
        ref_o1, ref_s1 = ref_row(R, ref_s[acc - 1])
        got_o1 = ttnn.to_torch(o).float().reshape(1, 1, HV, V)
        got_s1 = ttnn.to_torch(state).float()
        print(f"the next row from there: o {close(ref_o1, got_o1)}; the state: {close(ref_s1, got_s1)}", flush=True)
    finally:
        ttnn.close_device(device)
    bad = results.count(False)
    print(f"verify: {len(results)} comparisons, {f'{bad} outside the bars' if bad else 'all within the bars'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
