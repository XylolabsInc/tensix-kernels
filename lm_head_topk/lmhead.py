# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""lmhead.py: an LM head on the device that returns token ids, not logits.

The vocabulary is cut into `chunks` column blocks of the [dim, vocab] weight. Each block runs through the dense
reader's matmul (dense_reader/linear.py, linear.matmul) with block-float weights, DRAM width-sharded, for up to 16
rows (tokens) at a time. No logits are written: each core's writer keeps a running top-`k` of each row over the core's
columns (linear.matmul's topk) and writes it as one small record, so one small tensor holds every block's candidates.

With `rescore` (the default) the candidates' logits are recomputed exactly from the bf16 weight rows (x in bf16,
products summed in float32, rounded to bf16 unless logits="float32") and the largest is picked, the lowest token id
first on ties and a NaN counting as the largest (pick_token). The token then equals a bf16 host head's whenever the
host's top-1 is among the device's candidates. The rescore and the pick run on the host (collect), or with
device_pick=True on the device (head_pick.py), which then returns only the token ids.

  head = DeviceHead(device, lm_head_weight, fmt="bfp4")                     # [vocab, dim] torch
  ids = head(x, rows)                                                       # x [1, 1, >= rows, dim] on the device
  head = DeviceHead(device, lm_head_weight, device_pick=True)               # rescore and pick on the device

A draft head needs no rescore: with rescore=False and device_pick=True each row's best record by the device's own
logit is taken on the device and mapped through token_map (head_pick.pick_records), so a sub-head of the most frequent
tokens gives token ids directly:

  head = DeviceHead(device, sub_weight, rescore=False, device_pick=True, token_map=ids)   # sub_weight = w[ids]
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_reader import linear  # noqa: E402
from lm_head_topk import head_pick  # noqa: E402

FORMATS = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}


def pick_token(exact, ids, vocab):
    """The argmax over the rescored candidates: per row the lowest token id among the largest exact logits, a NaN
    counting as the largest (torch.argmax's rule). exact: [rows, c] float with -inf for empty slots, ids: [rows, c]
    token ids."""
    top = exact.max(dim=1, keepdim=True).values  # NaN where the row has one
    hit = (exact == top) | (exact.isnan() & top.isnan())
    return torch.where(hit, ids, torch.full_like(ids, vocab)).min(dim=1).values


class DeviceHead:
    """The LM head as `chunks` device matmuls with top-k records; see the module docstring."""

    def __init__(
        self,
        device,
        weight,
        fmt="bfp4",
        chunks=16,
        k=8,
        rescore=True,
        top=16,
        blocks=None,
        logits="bf16",
        device_pick=False,
        rows=None,
        valid=None,
        token_map=None,
        l1x=True,
        **opts,
    ):
        """weight: the head [vocab, dim] (torch; kept on the host in bf16 for a host rescore). chunks: vocabulary
        blocks, one dense-reader call each (vocab / chunks a multiple of 32; enough blocks that a block's accumulators
        fit a core's L1). blocks: device weights already made, [(w [1, 1, dim, vocab / chunks] DRAM width-sharded,
        scale or None)] in vocabulary order; else made here in fmt ("bfp8", "bfp4" or "bf16"). k: candidates per core
        and block; top: of all blocks' candidates, the best `top` by device logit are rescored. logits: the rescore's
        logits, "bf16" (rounded) or "float32" (the two can pick different tokens where candidates tie in bf16).
        device_pick: the rescore and the pick on the device (head_pick.py), with a bf16 copy of the head in DRAM instead
        of the host's; `top` at most 32. rows: that bf16 copy when it is already on the device (a tied embedding table
        in ttnn.embedding's layout, row-major interleaved DRAM, one page a row: head_pick.check_rows), used instead of
        a second copy; the caller keeps it. valid: a padded head's real vocabulary for the device pick (pad rows never
        candidates). Without a rescore, device_pick=True is the records-only pick (head_pick.pick_records: each row's
        best record by the device's logit) and token_map maps the head's rows to token ids: a list or tensor of ids
        (row i of weight is token token_map[i]) or head_pick.token_map's device tensor (the caller keeps that one).
        l1x: x copied into L1 (interleaved) once a call when it is elsewhere, so the blocks' cores read it from L1.
        opts go to linear.matmul."""
        vocab, dim = weight.shape
        chunks = len(blocks) if blocks is not None else chunks
        step = vocab // chunks
        assert step * chunks == vocab and step % 32 == 0, (vocab, chunks)
        assert fmt in FORMATS, fmt
        assert logits in ("bf16", "float32"), logits
        self.round = logits == "bf16"
        self.logits = logits
        self.device_pick = bool(device_pick)
        self.l1x = l1x
        self.rows_t = None  # the bf16 head on the device (device_pick)
        self.own_rows = False  # rows_t made here (freed by deallocate) rather than the caller's
        self.valid = valid  # the device pick's real vocabulary (None: the table's rows)
        self.picks = {}  # rows -> head_pick's output tensor (device_pick)
        self.records_pick = not rescore and self.device_pick  # the records-only pick, no rescore
        self.tmap, self.own_tmap = None, False  # the records pick's token map on the device (token_map)
        assert token_map is None or self.records_pick, "lmhead: token_map is for rescore=False, device_pick=True"
        grid = device.compute_with_storage_grid_size()
        self.slots = grid.x * grid.y  # records per block: at least the cores of any plan
        self.device, self.fmt, self.step, self.k, self.opts = device, fmt, step, k, opts
        self.vocab, self.dim, self.top = vocab, dim, top
        self.results = {}  # rows -> the top-k result tensor
        self.y = None  # the matmuls' placeholder output
        self.host = None
        if self.records_pick and token_map is not None:
            if isinstance(token_map, ttnn.Tensor):
                self.tmap = token_map
            else:
                assert len(token_map) == vocab, (len(token_map), vocab)
                self.tmap, self.own_tmap = head_pick.token_map(device, token_map), True
        if rescore and self.device_pick:
            assert 1 <= top <= 32, f"lmhead: the device pick rescores at most 32 candidates, top {top}"
            if rows is not None:
                head_pick.check_rows(rows, dim, vocab)
                self.rows_t = rows
            else:
                self.rows_t = head_pick.head_rows(device, weight)
                self.own_rows = True
        elif rescore:
            self.host = weight.to(torch.bfloat16)
        self.miss = 0  # rows whose device-best candidate was not the rescored best
        if blocks is not None:
            self.blocks = list(blocks)
            return
        self.blocks = []
        for a in range(0, vocab, step):
            tw, ts = linear.native_weight(device, weight[a : a + step], FORMATS[fmt])
            ttnn.deallocate(ts)
            self.blocks.append((tw, None))  # unit scales: no scale tensor, no multiply

    def logits_topk(self, x, rows=1, ids_out=None):
        """The device part: per block the matmul, whose writers keep each core's running top-k of the first `rows` rows
        (linear.matmul's topk; no logits are written), then with device_pick the rescore and the pick. Returns the
        result tensor (records: block b's cores at b * slots ..). ids_out (a device pick only): row 0's token also into
        this one-element uint32 tensor (head_pick.check_ids_out). The device part reads nothing back and keeps its
        tensors from call to call, so a trace can capture it."""
        assert 1 <= rows <= 16 and rows <= x.shape[-2], f"lmhead: 1 to 16 rows, at most x's ({rows})"
        r = self.results.get(rows)
        if r is None:
            r = self.results[rows] = linear.topk_result(self.device, len(self.blocks) * self.slots, rows, self.k)
        if self.y is None or self.y.shape[-2] != x.shape[-2]:
            # the matmuls' y: never written (write_y False), one placeholder for every block
            if self.y is not None:
                ttnn.deallocate(self.y)
            self.y = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, x.shape[-2], self.step]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        xb, mc = x, x.memory_config()
        if self.l1x and (
            mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        ):
            xb = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        for b, (tw, ts) in enumerate(self.blocks):
            spec = {"r": r, "k": self.k, "rows": rows, "slot0": b * self.slots}
            spec |= {"base": b * self.step, "n": self.step, "write_y": False}
            linear.matmul(xb, tw, ts, topk=spec, out=self.y, **self.opts)
        if self.rows_t is not None:
            # the rescore and the pick on the device: one output page a row, the same tensor every step
            self.picks[rows] = head_pick.pick(
                r,
                xb,
                self.rows_t,
                rows,
                self.k,
                top=self.top,
                logits=self.logits,
                out=self.picks.get(rows),
                valid=self.valid,
                ids_out=ids_out,
            )
        elif self.records_pick:
            # the records-only pick on the device: the best record's row, mapped (token_map); one output page a row
            self.picks[rows] = head_pick.pick_records(
                r,
                rows,
                self.k,
                self.vocab if self.valid is None else self.valid,
                tmap=self.tmap,
                out=self.picks.get(rows),
                ids_out=ids_out,
            )
        assert ids_out is None or rows in self.picks, "lmhead: ids_out needs the device pick"
        if xb is not x:
            ttnn.deallocate(xb)
        return r

    def candidates(self, x, rows):
        """(values [rows, c] float, global ids [rows, c] long) of the device's candidates."""
        r = self.logits_topk(x, rows)
        return linear.topk_records(r, rows, self.k)

    def rescore(self, xh, ids):
        """Exact logits of the candidates: x (bf16) times the bf16 weight rows, float32 sums, rounded to bf16 (unless
        logits="float32")."""
        w = self.host[ids.reshape(-1)].float().reshape(*ids.shape, self.dim)
        z = torch.einsum("rd,rcd->rc", xh.float(), w)
        return z.to(torch.bfloat16).float() if self.round else z

    def __call__(self, x, rows=1):
        """Token ids [rows] (long) for the first `rows` rows of x [1, 1, >= rows, dim] bf16 TILE."""
        self.logits_topk(x, rows)
        return self.collect(x, rows)

    def collect(self, x, rows=1, xh=None):
        """The host part, after logits_topk ran (directly, or inside a trace being replayed): with a device pick the
        token ids read back; else the records' readback, x's readback (or xh, a host copy of x's first rows) and the
        rescore. Token ids [rows] (long)."""
        if self.rows_t is not None:
            tok, best, _ = head_pick.tokens(self.picks[rows], rows)
            self.miss += int((best != tok).sum())
            return tok
        if self.records_pick:
            return head_pick.tokens(self.picks[rows], rows)[0]
        vh, ids = linear.topk_records(self.results[rows], rows, self.k)
        best = vh.argmax(dim=1)
        if self.host is None:
            return ids[torch.arange(rows), best]
        if xh is None:
            xh = ttnn.to_torch(x)
        xh = xh.reshape(-1, self.dim)[:rows].to(torch.bfloat16)
        if self.top and self.top < ids.shape[1]:
            pick = vh.topk(self.top, dim=1).indices
            vh, ids = vh.gather(1, pick), ids.gather(1, pick)
            best = vh.argmax(dim=1)
        valid = ids >= 0  # (empty records: -1)
        ids = ids.clamp(min=0)
        exact = self.rescore(xh, ids)
        exact = torch.where(valid, exact, torch.full_like(exact, float("-inf")))
        tok = pick_token(exact, ids, self.vocab)
        self.miss += int((ids[torch.arange(rows), best] != tok).sum())
        return tok

    def deallocate(self):
        for t in self.picks.values():
            ttnn.deallocate(t)
        self.picks = {}
        if self.tmap is not None:
            if self.own_tmap:
                ttnn.deallocate(self.tmap)
            self.tmap = None
        if self.rows_t is not None:
            if self.own_rows:
                ttnn.deallocate(self.rows_t)
            self.rows_t = None
        for r in self.results.values():
            ttnn.deallocate(r)
        self.results = {}
        if self.y is not None:
            ttnn.deallocate(self.y)
            self.y = None
        for tw, ts in self.blocks:
            ttnn.deallocate(tw)
            if ts is not None:
                ttnn.deallocate(ts)
        self.blocks = []
