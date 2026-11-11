# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Correctness of the top-k LM head (lm_head_topk/lmhead.py, head_pick.py) against torch, on a Tensix device or ttsim.

  python -s lm_head_topk/tests/test_lm_head.py [test names]     # or, with pytest: pytest -s lm_head_topk/tests

A random bf16 head [vocab, dim] on a bfloat4_b device head (the dense reader's top-k blocks) and random x rows:
- the records: every kept (value, id) is the bf16 logit the block matmul computes at that id, and each row's best k
  over the records equal its best k over all logits (the same op's y, written once with write_y), bit for bit;
- the host rescore's tokens against a bf16 host head (bf16 x times bf16 rows, float32 sums, rounded to bf16, argmax with
  the lowest id on ties), token for token;
- the device pick (head_pick.pick) against the host rescore on the same records, token for token, in bf16 and float32
  logits, also on a head whose rows repeat in pairs (exact ties: the lowest id must win) and on a padded head (valid=),
  its table also given as the valid rows only (a tied embedding of the real vocabulary);
- the records-only pick (pick_records) with and without a token map against the host's argmax over the same records;
- both picks with k 4 at 1 and 3 rows, whose 32-byte records must land 64 bytes apart in L1.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.testing import main, same  # noqa: E402
from dense_reader import linear  # noqa: E402
from lm_head_topk import head_pick, lmhead  # noqa: E402

VOCAB, DIM, CHUNKS = 8192, 1024, 4


def head(seed, pairs=False):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(VOCAB, DIM, generator=g) * 0.05).to(torch.bfloat16)
    if pairs:
        w[1::2] = w[0::2]  # rows 2i and 2i + 1 equal: every logit ties with its pair
    return w


def x_rows(device, rows, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, DIM, generator=g).to(torch.bfloat16)
    tx = ttnn.from_torch(x.reshape(1, 1, rows, DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x, tx


def host_tokens(x, w, logits="bf16"):
    z = x.float() @ w.float().T
    if logits == "bf16":
        z = z.to(torch.bfloat16).float()
    return lmhead.pick_token(z, torch.arange(VOCAB).expand(x.shape[0], -1), VOCAB)


def test_records(device):
    w = head(1)
    rows, k = 3, 8
    x, tx = x_rows(device, rows, 2)
    dh = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=k, rescore=False)
    vals, ids = dh.candidates(tx, rows)
    # the same blocks' logits, written: y of each block through the same op (write_y) and concatenated
    ys = []
    for tw, _ in dh.blocks:
        ys.append(ttnn.to_torch(linear.matmul(tx, tw, None)).float().reshape(-1, dh.step)[:rows])
    y = torch.cat(ys, 1)
    for t in range(rows):
        ok = ids[t] >= 0
        same(f"records row {t}: values at their ids", y[t][ids[t][ok]], vals[t][ok])
        top = torch.sort(vals[t][ok], descending=True).values[:k]
        same(f"records row {t}: the best {k} of all logits", top, torch.topk(y[t], k).values)
    dh.deallocate()


def test_host_rescore(device):
    w = head(3)
    dh = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS)
    for rows in (1, 5):
        x, tx = x_rows(device, rows, 4 + rows)
        same(f"host rescore rows {rows}: tokens against the bf16 host head", dh(tx, rows), host_tokens(x, w))
    dh.deallocate()


def test_device_pick(device):
    for kind, w in (("random", head(5)), ("ties", head(6, pairs=True))):
        for logits in ("bf16", "float32"):
            host = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, logits=logits)
            dev = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, logits=logits, device_pick=True)
            for rows in (1, 4):
                x, tx = x_rows(device, rows, 7 + rows)
                want = host(tx, rows)
                got = dev(tx, rows)
                same(f"device pick ({kind}, {logits} logits) rows {rows}: tokens against the host rescore", got, want)
                if kind == "random" and logits == "bf16":
                    same(f"device pick rows {rows}: tokens against the bf16 host head", got, host_tokens(x, w))
            dev.deallocate()
            host.deallocate()


def test_small_records(device):
    """k 4: a record of one row is 32 bytes, so records sit 32 bytes apart in DRAM's pages but must land 64 bytes apart
    in L1 where DRAM reads keep their offset modulo 64 (Blackhole's DRAM alignment)."""
    w = head(14)
    for rows in (1, 3):
        x, tx = x_rows(device, rows, 15 + rows)
        host = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=4)
        dev = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=4, device_pick=True)
        same(f"device pick, k 4, rows {rows}: tokens against the host rescore", dev(tx, rows), host(tx, rows))
        plain = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=4, rescore=False)
        vals, ids = plain.candidates(tx, rows)
        picked = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=4, rescore=False, device_pick=True)
        best = ids[torch.arange(rows), vals.argmax(dim=1)]
        same(f"records pick, k 4, rows {rows}: rows against the host's argmax over the records", picked(tx, rows), best)
        for h in (host, dev, plain, picked):
            h.deallocate()


def test_padded_head(device):
    w = head(9)
    valid = VOCAB - 3 * 32 - 5  # the last rows are padding: never candidates
    rows, k = 2, 8
    x, tx = x_rows(device, rows, 10)
    dh = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=k, rescore=False)
    r = dh.logits_topk(tx, rows)
    rows_t = head_pick.head_rows(device, w)
    out = head_pick.pick(r, tx, rows_t, rows, k, valid=valid)
    tok, _, _ = head_pick.tokens(out, rows)
    vals, ids = linear.topk_records(r, rows, k)
    vals = torch.where(ids >= valid, torch.full_like(vals, float("-inf")), vals)
    best = vals.topk(16, dim=1).indices
    cand = ids.gather(1, best)
    z = torch.einsum("rd,rcd->rc", x.float(), w[cand.clamp(min=0).reshape(-1)].float().reshape(rows, 16, DIM))
    z = z.to(torch.bfloat16).float()
    z = torch.where((cand >= 0) & (cand < valid), z, torch.full_like(z, float("-inf")))
    want = lmhead.pick_token(z, cand, VOCAB)
    same(f"padded head (valid {valid}): tokens against the host rescore of the valid candidates", tok, want)
    assert int(tok.max()) < valid
    # a table of the valid rows only (a tied embedding of the real vocabulary) serves the padded head
    short = head_pick.head_rows(device, w[:valid])
    dv = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=k, device_pick=True, valid=valid, rows=short)
    same(f"padded head (valid {valid}), a table of the valid rows only: tokens", dv(tx, rows), want)
    dv.deallocate()
    ttnn.deallocate(short)
    ttnn.deallocate(rows_t)
    dh.deallocate()


def test_records_pick(device):
    w = head(11)
    rows, k = 3, 8
    g = torch.Generator().manual_seed(12)
    tmap = torch.randperm(1 << 20, generator=g)[:VOCAB]  # row i of the sub-head is token tmap[i]
    x, tx = x_rows(device, rows, 13)
    plain = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=k, rescore=False)
    vals, ids = plain.candidates(tx, rows)
    best = ids[torch.arange(rows), vals.argmax(dim=1)]  # the host's argmax over the same records
    picked = lmhead.DeviceHead(device, w, fmt="bfp4", chunks=CHUNKS, k=k, rescore=False, device_pick=True)
    same("records pick: rows against the host's argmax over the records", picked(tx, rows), best)
    mapped = lmhead.DeviceHead(
        device, w, fmt="bfp4", chunks=CHUNKS, k=k, rescore=False, device_pick=True, token_map=tmap.tolist()
    )
    same("records pick with a token map: tokens", mapped(tx, rows), tmap[best])
    for h in (plain, picked, mapped):
        h.deallocate()


if __name__ == "__main__":
    sys.exit(main(globals()))
