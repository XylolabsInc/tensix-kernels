# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Run causal conv1d with four taps and SiLU for T Gated DeltaNet tokens.

The device kernel uses ttnn.generic_op (kernels/gdn_conv_*.cpp). Per channel and token t, with
the window w = [s0, s1, s2, x0 .. x(T-1)] (the conv state, then the tokens): out_t = silu(bias +
w3·w(t+3) + w0·w(t) + w1·w(t+1) + w2·w(t+2)); the conv state shifted in place (s0..s2 <- w(T..T+2));
optionally the window written to a buffer (the speculative verify's commit takes its rows from
there). T = 1 is the decode step; a verify's T rows give, row by row, the decode steps' results bit
for bit (one order of operations for every t and T). It replaces upstream's
_causal_conv1d_decode_t1_inplace and, for T rows, _causal_conv1d_fir (tenstorrent/tt-metal v0.81.1,
models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py, Apache-2.0): their
slices, to_layouts, multiplies, adds, concat, copy and SiLU, and the fused projection output's qkv slice.

  out = gated_conv_step(x, state, taps, bias, channels=10240)              # x [B, 1, W]
  out = gated_conv_step(x, state, taps, bias, channels=10240, window=win)  # x [B, T, W]

x: flat [B, T, W] bf16 (the fused projection output), the C = `channels` conv inputs from element column
x_col (a multiple of 32); T is 1..29. state: [B, 3, C] bf16, rows 0..2 the last three tokens,
updated in place (same buffer: trace-safe). taps: 4 tensors [1, 1, C] bf16 (tap k for state row k,
tap 3 for x); bias: [1, 1, C] bf16 or None. window: None or [B, 3 + T, C] bf16. Returns out,
[B, T, C] bf16 (rows 0..T-1; the other rows zero).

The kernel packs up to 32 channel tiles into the rows of one tile per operand, so each core does a
few tile operations for all its channels. The taps and the bias are packed that way once on the
host and cached per taps tensor and work plan. The first call reads them back from the device,
so it must run before a trace capture."""

import math

import ttnn
from gdn_scan import TILE, source, tile_bytes

# The CB plan of kernels/gdn_conv_*.cpp.
CB = {
    "in": 0,
    "w": 1,
    **{f"m{i}": 2 + i for i in range(8)},
    "out": 16,
    "rx": 20,
    "rs": 21,
    "wo": 22,
    "rw": 23,
    "rsl": 24,
}
GROUP = 32  # items (batch row, channel tile) per packed tile
_PACKED = {}


def plan(device, items):
    """[(core, first item, items)] over the compute grid, balanced, column-major."""
    g = device.compute_with_storage_grid_size()
    ncores = min(items, g.x * g.y)
    base, extra = divmod(items, ncores)
    out, i = [], 0
    for c in range(ncores):
        n = base + (1 if c < extra else 0)
        out.append((ttnn.CoreCoord(c // g.y, c % g.y), i, n))
        i += n
    return out


def _packed(device, taps, bias, C, work):
    """The taps and the bias packed per core and group of GROUP items (row j = item j's 32
    channels; 5 pages per group: w0..w3, bias), and each core's first page."""
    key = (id(device), *(id(t) for t in taps), id(bias), C, tuple((i0, n) for _, i0, n in work))
    if key not in _PACKED:
        import torch

        w = [ttnn.to_torch(t).float().flatten()[:C] for t in taps]
        bv = torch.zeros(C) if bias is None else ttnn.to_torch(bias).float().flatten()[:C]
        CT = C // TILE
        pages, first = [], []
        for _, i0, n in work:
            first.append(5 * len(pages))
            for j0 in range(0, n, GROUP):
                blk = torch.zeros(5, TILE, TILE)
                for j in range(min(GROUP, n - j0)):
                    ct = (i0 + j0 + j) % CT
                    sl = slice(ct * TILE, (ct + 1) * TILE)
                    for k in range(4):
                        blk[k, j] = w[k][sl]
                    blk[4, j] = bv[sl]
                pages.append(blk)
        packed = ttnn.from_torch(
            torch.cat(pages),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # The tensors are held with the packing, so their ids cannot be reused.
        _PACKED[key] = (taps, bias, packed, first)
    return _PACKED[key][2], _PACKED[key][3]


def gated_conv_step(
    x, state, taps, bias=None, *, channels, x_col=0, window=None, slots=None, memory_config=None
):
    device = x.device()
    C = channels
    assert C % TILE == 0 and x_col % TILE == 0, (C, x_col)
    assert len(taps) == 4, f"gdn_conv: kernel size 4, got {len(taps)} taps"
    shape = list(x.shape)
    T = shape[-2]
    assert 1 <= T <= TILE - 3, f"gdn_conv: 1 to {TILE - 3} tokens, got {x.shape}"
    B = math.prod(shape[:-2])
    assert list(state.shape) == [B, 3, C], f"gdn_conv: state {state.shape}, want {[B, 3, C]}"
    if window is not None:
        assert list(window.shape) == [B, 3 + T, C], f"gdn_conv: window {window.shape}"
    # slots: T - 1 tensors [B, 3, C] bf16 TILE; slot a holds the conv state after row a.
    slots = list(slots or ())
    assert not slots or len(slots) == T - 1, f"gdn_conv: {len(slots)} slots for T {T}, want T - 1"
    for sl in slots:
        assert (
            list(sl.shape) == [B, 3, C]
            and sl.dtype == ttnn.bfloat16
            and sl.layout == ttnn.TILE_LAYOUT
        ), sl.shape
        assert not sl.memory_config().is_sharded(), "gdn_conv: interleaved slots"
    extra = () if bias is None else (bias,)
    extra += () if window is None else (window,)
    for t in (x, state, *taps, *extra):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT, (t.shape, t.dtype)
    CT = C // TILE
    work = plan(device, B * CT)
    packed, first = _packed(device, taps, bias, C, work)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, T, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc
    )
    per = min(GROUP, max(n for _, _, n in work))
    fmt = {n: ttnn.float32 for n in CB}
    fmt.update({"in": ttnn.bfloat16, "w": ttnn.bfloat16, "out": ttnn.bfloat16})
    fmt.update({"rx": x.dtype, "rs": state.dtype, "wo": ttnn.bfloat16, "rw": state.dtype})
    tiles = {n: 1 for n in CB}
    tiles.update({"in": T + 3, "w": 5, "rx": per, "rs": per, "wo": per})
    tiles["rw"] = per if window is not None else 1
    tiles["rsl"] = max(1, len(slots))
    fmt["rsl"] = state.dtype
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles[n] * tile_bytes(fmt[n]),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB[n], data_format=fmt[n], page_size=tile_bytes(fmt[n])
                )
            ],
        )
        for n in CB
    ]
    reader_ct = [T, int(window is not None), len(slots)]
    for t in (x, state, packed, state if window is None else window):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [T, *ttnn.TensorAccessorArgs(out).get_compile_time_args()]
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    x_pitch = list(x.padded_shape)[-1] // TILE
    for (c, i0, n), w0 in zip(work, first, strict=True):
        rd[c.x][c.y] = [
            i0,
            n,
            CT,
            x_pitch,
            x_col // TILE,
            w0,
            x.buffer_address(),
            state.buffer_address(),
            packed.buffer_address(),
            0 if window is None else window.buffer_address(),
            *(sl.buffer_address() for sl in slots),
        ]
        wr[c.x][c.y] = [i0, n, CT, out.buffer_address()]
        cp[c.x][c.y] = [-(-n // GROUP)]

    def kernel(name, ct_args, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct_args,
            runtime_args=rt,
            config=config,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("gdn_conv_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("gdn_conv_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "gdn_conv_compute.cpp",
                [T, int(bias is not None)],
                cp,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    io = [x, packed] + ([] if window is None else [window])
    ttnn.generic_op([*io, out, state, *slots], program)
    return out
