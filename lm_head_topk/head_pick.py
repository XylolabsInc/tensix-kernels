# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""head_pick.py: the top-k LM head's exact rescore and pick on the device.

lmhead.DeviceHead's device part leaves, per row, each block core's top-k candidates in a result tensor (linear.matmul's
topk records). pick() takes the best `top` of them by the device's logit, recomputes their logits from the bf16 head
rows (bf16 x times bf16 rows, summed in float32, rounded to bf16 unless logits="float32") and picks the largest (the
lowest id on ties, a NaN counting as the largest: lmhead.pick_token's rule), one core a row, in one dispatch that a
trace can hold (no host reads, fixed buffers):

  rows = head_rows(device, weight)                    # the bf16 head [vocab, dim], row-major in DRAM (one page a row)
  out = pick(records, x, rows, nrows, k, top=16)      # [1, 1, 32, 32] uint32 row-major; row r: token, device best, count

A draft head needs no rescore (a draft is a guess a verify step checks): pick_records takes each row's best record by
the device's own logit and maps its row through a token map (a sub-head of the most frequent tokens), on one core:

  tmap = token_map(device, ids)                       # row i of the sub-head is token ids[i]
  out = pick_records(records, nrows, k, len(ids), tmap)   # the same output page a row: token, sub-head row, count

The token equals a host rescore's whenever both sum the same products to the same float32 (the device's sum order
differs from torch's, so a near tie in the last bits of the float32 sum can part them). Kernels:
kernels/head_pick_reader.cpp, head_pick_compute.cpp, head_pick_writer.cpp; pick_records: kernels/head_pick_records.cpp.
"""

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import source as src  # noqa: E402

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32
_PROGRAMS = {}


def source(name):
    return src.load(KERNELS, name)


def head_rows(device, weight):
    """The bf16 head [vocab, dim] (torch) as a row-major DRAM tensor [1, 1, vocab, dim], one page a vocabulary row."""
    vocab, dim = weight.shape
    return ttnn.from_torch(
        weight.to(torch.bfloat16).reshape(1, 1, vocab, dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_rows(rows_t, dim, vocab=None):
    """A bf16 head table pick can gather from: row-major, interleaved in DRAM, one page a vocabulary row ([rows, dim]
    or [1, 1, rows, dim]; head_rows', or a tied embedding table in ttnn.embedding's layout). vocab, when given: the ids
    the pick reads (a padded head's valid count), so the table holds at least that many rows (a tied embedding table of
    the real vocabulary serves a head padded past it)."""
    shape = tuple(rows_t.shape)
    mc = rows_t.memory_config()
    assert (
        rows_t.dtype == ttnn.bfloat16
        and rows_t.layout == ttnn.ROW_MAJOR_LAYOUT
        and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and mc.buffer_type == ttnn.BufferType.DRAM
        and all(s == 1 for s in shape[:-2])
        and shape[-1] == dim
        and (vocab is None or shape[-2] >= vocab)
    ), (shape, rows_t.dtype, rows_t.layout, mc, dim, vocab)


def check_ids_out(t):
    """ids_out: a one-element uint32 row-major tensor, interleaved in DRAM ([1, 1] from ttnn.from_torch): an
    embedding's index for a draft's next step. The kernels write its page 0 as 32 bytes, [token, 0, ...], inside the
    page's DRAM-aligned slot (the allocator aligns a DRAM page to at least 32 bytes)."""
    mc = t.memory_config()
    n = 1
    for d in t.shape:
        n *= d
    assert (
        t.dtype == ttnn.uint32
        and t.layout == ttnn.ROW_MAJOR_LAYOUT
        and n == 1
        and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and mc.buffer_type == ttnn.BufferType.DRAM
    ), (tuple(t.shape), t.dtype, t.layout, mc)


def _cw(dt):
    """A gathered block's tiles: the largest of 8, 4, 2, 1 that divides the row's tiles."""
    return next(c for c in (8, 4, 2, 1) if dt % c == 0)


def _out_page(device):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, TILE, TILE]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def pick(records, x, rows_t, nrows, k, top=16, logits="bf16", out=None, valid=None, ids_out=None):
    """Token ids for the first nrows rows of x [1, 1, >= nrows, dim] bf16 TILE: per row the best `top` candidates of
    the top-k result `records` (linear.topk_result's, made for nrows rows and k), rescored with the bf16 rows of rows_t
    (head_rows') and picked (lmhead.pick_token's rule). out: [1, 1, 32, 32] uint32 row-major (made here unless given; a
    trace passes the same one each step); row r = [token, the device's own best candidate, the valid count, 0, ...].
    valid: a padded head's real vocabulary (ids at or past it are empty slots, dropped before the best `top` are
    chosen; default the table's rows). A record id at or past the table's rows counts as an empty slot (never read). A
    row with no valid candidate gets token 0 and count 0. ids_out: also row 0's token into this one-element uint32
    tensor (check_ids_out)."""
    assert logits in ("bf16", "float32"), logits
    assert 1 <= top <= 32 and 1 <= nrows, (top, nrows)
    device = x.device()
    dim = x.shape[-1]
    assert dim % TILE == 0, dim
    check_rows(rows_t, dim)
    dt = dim // TILE
    cw = _cw(dt)
    nrec = records.shape[-2]
    page = 2 * nrows * k * 4
    assert page % 32 == 0, (nrows, k)  # the reader reads records into L1 a page apart: 32-byte aligned DRAM reads
    vocab = rows_t.shape[-2] if valid is None else int(valid)  # ids at or past it: empty slots (the reader's VOCAB)
    assert 1 <= vocab <= rows_t.shape[-2], (valid, rows_t.shape)
    assert records.shape[-1] * (2 if records.dtype == ttnn.bfloat16 else 4) == page, (records.shape, nrows, k)
    grid = device.compute_with_storage_grid_size()
    assert nrows <= grid.x * grid.y, nrows
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nrows)]
    if out is None:
        out = _out_page(device)
    rnd = logits == "bf16"
    if ids_out is not None:
        check_ids_out(ids_out)
    dt_ = ids_out if ids_out is not None else out  # (no ids_out: out's accessor stands in, never written)
    key = (
        ids_out is not None,
        str(dt_.memory_config()),
        nrec,
        nrows,
        k,
        top,
        rnd,
        dim,
        vocab,
        str(x.memory_config()),
        str(records.memory_config()),
        str(rows_t.memory_config()),
        str(out.memory_config()),
    )
    prog = _PROGRAMS.get(key)
    if prog is None:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        bf = ttnn.bfloat16

        def cb(index, fmt, page_size, pages):
            return ttnn.CBDescriptor(
                total_size=page_size * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)],
            )

        cbs = [
            cb(0, bf, 2048, 2),  # x, a tile at a time
            cb(1, bf, 2048, 2 * cw),  # two gathered blocks of 32 row-major rows
            cb(2, bf, 2048, dt),  # the candidates' tiles
            cb(3, ttnn.uint32, -(-(4 * (1 + 2 * top)) // 64) * 64, 1),  # the candidates, to the writer
            cb(4, ttnn.uint32, 128, 1),  # the output page's staging
            cb(9, ttnn.uint32, -(-(nrec * page) // 64) * 64, 1),  # the records
            cb(16, bf if rnd else ttnn.float32, 2048 if rnd else 4096, 1),  # the logits' tile
        ]
        reader_ct = [dt, cw, top, k, nrows, nrec, vocab]
        for t in (records, rows_t, x):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = [
            top,
            int(rnd),
            int(ids_out is not None),
            *ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            *ttnn.TensorAccessorArgs(dt_).get_compile_time_args(),
        ]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for r, c in enumerate(cores):
            rd[c.x][c.y] = [records.buffer_address(), rows_t.buffer_address(), x.buffer_address(), r, dim * 2]
            wr[c.x][c.y] = [out.buffer_address(), r, dt_.buffer_address()]

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=source(name),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        prog = ttnn.ProgramDescriptor(
            kernels=[
                kernel("head_pick_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
                kernel("head_pick_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
                kernel(
                    "head_pick_compute.cpp",
                    [dt, cw, int(rnd)],
                    ttnn.RuntimeArgs(),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:  # the same program: the tensors' addresses rewritten in place (a trace replays those of its capture)
        rd, wr = prog.kernels[0].runtime_args, prog.kernels[1].runtime_args
        for c in cores:
            rd[c.x][c.y][0] = records.buffer_address()
            rd[c.x][c.y][1] = rows_t.buffer_address()
            rd[c.x][c.y][2] = x.buffer_address()
            wr[c.x][c.y][0] = out.buffer_address()
            wr[c.x][c.y][2] = dt_.buffer_address()
    ttnn.generic_op([records, rows_t, x] + ([ids_out] if ids_out is not None else []) + [out], prog)
    return out


def token_map(device, ids):
    """A sub-head's token map for pick_records: row i of the sub-head is token ids[i] (below 2**31). [1, 1, ceil(n /
    32), 32] uint32 row-major in DRAM: 32 ids a 128-byte page, the tail past n zero (pick_records never reads it)."""
    ids = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
    n = ids.numel()
    assert n >= 1 and int(ids.min()) >= 0 and int(ids.max()) < 1 << 31, (n, ids.min(), ids.max())
    pages = -(-n // TILE)
    t = torch.zeros(pages * TILE, dtype=torch.int32)
    t[:n] = ids.to(torch.int32)
    return ttnn.from_torch(
        t.reshape(1, 1, pages, TILE),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def pick_records(records, nrows, k, vocab, tmap=None, out=None, ids_out=None):
    """Token ids for the first nrows rows of a top-k result (linear.topk_result's, made for nrows rows and k), with no
    rescore: per row the best record slot by the device's logit (the first of equal keys, record by record and slot by
    slot: torch.argmax over linear.records_host's values; a NaN counts as the largest), its id (a row of the head the
    records came from) mapped through tmap (token_map's, at least vocab ids; None: the id itself). vocab: the head's rows
    that count (a slot of id at or past it is empty: a padded head's real rows). out: [1, 1, 32, 32] uint32 row-major
    (made here unless given); row r = [token, the best id, the valid slots' count, 0, ...], tokens() reads it. A row with
    no valid slot gets token 0, id 0xFFFFFFFF and count 0. One core does every row (kernels/head_pick_records.cpp).
    ids_out: as pick's (row 0's token into a one-element uint32 tensor)."""
    assert 1 <= nrows <= TILE and k >= 1, (nrows, k)
    device = records.device()
    nrec = records.shape[-2]
    page = 2 * nrows * k * 4
    assert page % 32 == 0, (nrows, k)  # records read into L1 a page apart: 32-byte aligned DRAM reads
    assert records.shape[-1] * (2 if records.dtype == ttnn.bfloat16 else 4) == page, (records.shape, nrows, k)
    vocab = int(vocab)
    assert 1 <= vocab < 1 << 32, vocab
    if tmap is not None:
        mc = tmap.memory_config()
        assert (
            tmap.dtype == ttnn.uint32
            and tmap.layout == ttnn.ROW_MAJOR_LAYOUT
            and tmap.shape[-1] == TILE
            and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and mc.buffer_type == ttnn.BufferType.DRAM
            and tmap.shape[-2] * TILE >= vocab
        ), (tuple(tmap.shape), tmap.dtype, tmap.layout, mc, vocab)
    if out is None:
        out = _out_page(device)
    mapped = tmap is not None
    mt = tmap if mapped else out  # (no map: out's accessor stands in, never read)
    if ids_out is not None:
        check_ids_out(ids_out)
    dt_ = ids_out if ids_out is not None else out  # (no ids_out: out's accessor stands in, never written)
    key = (
        "records",
        ids_out is not None,
        str(dt_.memory_config()),
        nrec,
        nrows,
        k,
        vocab,
        mapped,
        str(records.memory_config()),
        str(mt.memory_config()),
        str(out.memory_config()),
    )
    core = ttnn.CoreCoord(0, 0)
    prog = _PROGRAMS.get(key)
    if prog is None:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        u32 = ttnn.uint32

        def cb(index, page_size, pages):
            return ttnn.CBDescriptor(
                total_size=page_size * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=u32, page_size=page_size)],
            )

        size = -(-(nrec * page) // 64) * 64
        cbs = [cb(4, 128, nrows), cb(5, 128, 1), cb(9, size, 1)]  # the output pages, a map page, the records
        ct = [k, nrows, nrec, vocab, int(mapped), int(ids_out is not None)]
        for t in (records, mt, out, dt_):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rd = ttnn.RuntimeArgs()
        rd[core.x][core.y] = [records.buffer_address(), mt.buffer_address(), out.buffer_address(), dt_.buffer_address()]
        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=source("head_pick_records.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                )
            ],
            semaphores=[],
            cbs=cbs,
        )
        _PROGRAMS[key] = prog
    else:  # the same program: the addresses rewritten in place (a trace replays those of its capture)
        a = prog.kernels[0].runtime_args[core.x][core.y]
        a[0], a[1], a[2] = records.buffer_address(), mt.buffer_address(), out.buffer_address()
        a[3] = dt_.buffer_address()
    io = [records] + ([mt] if mapped else []) + ([ids_out] if ids_out is not None else []) + [out]
    ttnn.generic_op(io, prog)
    return out


def tokens(out, nrows):
    """The host's read of pick's output: (tokens [nrows], the device's own best [nrows], valid counts [nrows])."""
    t = ttnn.to_torch(out).reshape(32, 32)[:nrows].to(torch.int64) & 0xFFFFFFFF
    return t[:, 0], t[:, 1], t[:, 2]
