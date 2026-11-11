# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Run prompt causal conv1d with four taps, no bias, and SiLU on fused q|k|v channels.

The device kernel (kernels/gdn_pconv_*.cpp) matches upstream's
_causal_conv1d_fir (tenstorrent/tt-metal v0.81.1, models/experimental/
gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py, Apache-2.0): the same SFPU steps in the same
order (mul_binary_tile, three addcmul_tile, silu_tile, Dst in 16-bit mode), with the shifted copies of
x made by exact 0/1 matmuls instead of the concat, slices and layout conversions around them. It reads
x in place from the fused projection output's columns and also writes the new conv state, whole and split into
q, k, and v, as required by the prefill interface.

  out, state, conv_q, conv_k, conv_v = prefill_conv(x, state, taps, channels=C, q_dim=.., k_dim=..)

x: [1, T, W] bf16 TILE (x = its columns x_col .. x_col + C); state: [1, 3, C] bf16 TILE or
None (zeros); taps: 4 tensors [1, 1, C] bf16 TILE (tap k for x(t - 3 + k)). Returns out [1, T, C]
bf16 (DRAM; rows T.. of the last tile: the conv of whatever the fused projection output holds there) and the
states [1, 3, C], [1, 3, q_dim], [1, 3, k_dim], [1, 3, C - q_dim - k_dim] (DRAM). The taps are
row-replicated on the host once per taps tensor (the first call reads them back, so it must run before
a trace capture), as gdn_conv does."""

import ttnn
from gdn_conv import plan
from gdn_scan import TILE, source

CB = {"x": 0, "s": 1, "w": 2, "k": 3, "out": 16, "ns": 17}
MAX_RT = 64  # T up to 2048 a call (cb_x holds 2 RT tiles)


CALLS = [0]  # prefill_conv calls; tests use this to check dispatch
_WREP = {}
_CONSTS = {}


def _wrep(device, taps, C):
    """The taps with row 0 copied to all 32 rows: [4 * 32, C] bf16, page k * C / 32 + ct."""
    key = (id(device), *(id(t) for t in taps), C)
    if key not in _WREP:
        import torch

        w = [ttnn.to_torch(t).flatten()[:C] for t in taps]  # bf16 as stored
        rep = torch.stack([wk.expand(TILE, C) for wk in w]).reshape(4 * TILE, C).contiguous()
        dev = ttnn.from_torch(
            rep,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _WREP[key] = (taps, dev)  # held with the taps, so their ids are not reused
    return _WREP[key][1]


def consts(T):
    """The 12 constant 0/1 tiles (torch [12, 32, 32]) and (NS_PREV, NS_STATE) for T tokens."""
    import torch

    k = torch.zeros(12, TILE, TILE)
    for s in (1, 2, 3):
        for r in range(TILE):
            if r >= s:
                k[s - 1, r, r - s] = 1  # L_s: row r <- x's row r - s, this row tile
            else:
                k[3 + s - 1, r, TILE + r - s] = 1  # C_s: the previous row tile's row 32 + r - s
                k[6 + s - 1, r, 3 + r - s] = (
                    1  # P_s: the conv state's row 3 + r - s (x(r - s), r - s < 0)
                )
    last = (T - 1) // TILE
    prev = state = False
    for j in range(3):
        u = (
            T - 3 + j
        )  # the new state's row j is x(u); x(-1) .. x(-3) are the old state's rows 2 .. 0
        if u >= last * TILE:
            k[9, j, u - last * TILE] = 1
        elif u >= 0:
            k[10, j, u - (last - 1) * TILE] = 1
            prev = True
        else:
            k[11, j, 3 + u] = 1
            state = True
    return k, prev, state


def _consts(device, T):
    # the tiles depend on T only through (T - 1) mod 32 and T < 3 (the old state's rows): at most 34 tensors
    key = (id(device), min(T, 3), (T - 1) % TILE)
    if key not in _CONSTS:
        k, prev, state = consts(T)
        dev = ttnn.from_torch(
            k.reshape(12 * TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _CONSTS[key] = (dev, prev, state)
    return _CONSTS[key]


def fits(x, state, taps, C, bias=None):
    """Whether prefill_conv takes this call (else the stock path)."""
    shape = list(x.shape)
    return (
        bias is None
        and len(taps) == 4
        and all(
            list(t.shape) == [1, 1, C] and t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT
            for t in taps
        )
        and len(shape) == 3
        and shape[0] == 1
        and 2 <= shape[1] <= MAX_RT * TILE  # T = 1 is the decode step's conv (another stock path)
        and C % TILE == 0
        and x.dtype == ttnn.bfloat16
        and x.layout == ttnn.TILE_LAYOUT
        and not x.memory_config().is_sharded()
        and (
            state is None
            or (
                not isinstance(state, (list, tuple))  # the stock path also takes [B, 1, D] rows
                and list(state.shape) == [1, 3, C]
                and state.dtype == ttnn.bfloat16
                and state.layout == ttnn.TILE_LAYOUT
                and not state.memory_config().is_sharded()
            )
        )
    )


def prefill_conv(x, state, taps, *, channels, q_dim, k_dim, x_col=0):
    device = x.device()
    C = channels
    T = x.shape[1]
    assert fits(x, state, taps, C), "gdn_pconv: this call is not supported (use the stock path)"
    CALLS[0] += 1
    assert x_col % TILE == 0 and q_dim % TILE == 0 and k_dim % TILE == 0 and q_dim + k_dim < C
    RT, CT = -(-T // TILE), C // TILE
    wrep = _wrep(device, taps, C)
    kdev, ns_prev, ns_state = _consts(device, T)
    dram = ttnn.DRAM_MEMORY_CONFIG

    def alloc(shape):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
        )

    out = alloc([1, T, C])
    fused, cq, ck, cv = (
        alloc([1, 3, C]),
        alloc([1, 3, q_dim]),
        alloc([1, 3, k_dim]),
        alloc([1, 3, C - q_dim - k_dim]),
    )
    work = plan(device, CT)
    tb = 2 * TILE * TILE
    tiles = {"x": 2 * RT, "s": 2, "w": 8, "k": 12, "out": 2, "ns": 2}
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles[n] * tb,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB[n], data_format=ttnn.bfloat16, page_size=tb)
            ],
        )
        for n in CB
    ]
    s_in = state if state is not None else x  # a placeholder accessor when there is no state
    reader_ct = [RT, int(state is not None)]
    for t in (x, s_in, wrep, kdev):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [RT]
    for t in (out, fused, cq, ck, cv):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    x_pitch = list(x.padded_shape)[-1] // TILE
    for c, i0, n in work:
        rd[c.x][c.y] = [
            i0,
            n,
            CT,
            x_pitch,
            x_col // TILE,
            x.buffer_address(),
            s_in.buffer_address(),
            wrep.buffer_address(),
            kdev.buffer_address(),
        ]
        wr[c.x][c.y] = [
            i0,
            n,
            CT,
            q_dim // TILE,
            k_dim // TILE,
            out.buffer_address(),
            fused.buffer_address(),
            cq.buffer_address(),
            ck.buffer_address(),
            cv.buffer_address(),
        ]
        cp[c.x][c.y] = [n]

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
            kernel("gdn_pconv_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("gdn_pconv_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "gdn_pconv_compute.cpp",
                [RT, int(ns_prev), int(ns_state)],
                cp,
                # as the stock ops: Dst in 16-bit mode, HiFi4 (the 0/1 matmuls: exact for finite bf16, |x| >= 2^-121)
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    io = [x, wrep, kdev] + ([state] if state is not None else [])
    ttnn.generic_op([*io, out, fused, cq, ck, cv], program)
    return out, fused, cq, ck, cv


def stock(x, state, taps, *, channels, q_dim, k_dim, x_col=0, dram=False):
    """Run the reference prompt convolution through upstream ops.

    dram places FIR intermediates in DRAM instead of L1 for T <= 512. This keeps the same
    arithmetic while allowing wide tensors to coexist with the test's other allocations.
    """
    from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet as m

    C, T = channels, x.shape[1]
    mc = None if dram else m._seq_memory_config(T)
    qkv = ttnn.to_layout(x[:, :, x_col : x_col + C], ttnn.TILE_LAYOUT)
    out, cs = m._causal_conv1d_fir(
        qkv,
        None,
        None,
        4,
        x.device(),
        memory_config=mc,
        conv_state=state,
        weight_taps=taps,
        bias_dev=None,
        valid_len=None,
    )
    cq = ttnn.to_layout(cs[:, :, :q_dim], ttnn.TILE_LAYOUT)
    ck = ttnn.to_layout(cs[:, :, q_dim : q_dim + k_dim], ttnn.TILE_LAYOUT)
    cv = ttnn.to_layout(cs[:, :, q_dim + k_dim :], ttnn.TILE_LAYOUT)
    return out, cs, cq, ck, cv


# --- the prompt's gates (gdn_pgates_*.cpp) ---------------------------------------------------------------------------
GCB = {"a": 0, "b": 1, "db": 2, "an": 3, "k": 4, "g": 16, "beta": 17}
GCALLS = [0]  # calls of prefill_gates (the checks read it)
_GCONSTS = {}


def _gconsts(device, dt_bias, A_neg, HV, off):
    """dt_bias's and A_neg's rows copied to all 32 rows (NT tiles each), then M (column c -> c - off) and M2
    (c -> c + 32 - off): [2 NT + 2, 32, 32] bf16 pages. Read back once per tensor pair (A_neg as the stock path makes
    it: a device op's bf16 values), held with them."""
    key = (id(device), id(dt_bias), id(A_neg), HV, off)
    if key not in _GCONSTS:
        import torch

        NT = -(-HV // TILE)
        db = ttnn.to_torch(dt_bias).flatten()[:HV]
        an = ttnn.to_torch(A_neg).flatten()[:HV]
        pages = torch.zeros(2 * NT + 2, TILE, TILE, dtype=torch.bfloat16)
        for t in range(NT):
            w = min(TILE, HV - t * TILE)
            pages[t, :, :w] = db[t * TILE : t * TILE + w]
            pages[NT + t, :, :w] = an[t * TILE : t * TILE + w]
        for c in range(TILE):
            if c >= off:
                pages[2 * NT, c, c - off] = 1
            else:
                pages[2 * NT + 1, c, c + TILE - off] = 1
        dev = ttnn.from_torch(
            pages.reshape((2 * NT + 2) * TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _GCONSTS[key] = (dt_bias, A_neg, dev)
    return _GCONSTS[key][2]


def gates_fit(x, a_col, b_col, HV, dt_bias, A_neg):
    shape = list(x.shape)
    return (
        len(shape) == 3
        and shape[0] == 1
        and 2 <= shape[1] <= MAX_RT * TILE
        and x.dtype == ttnn.bfloat16
        and x.layout == ttnn.TILE_LAYOUT
        and not x.memory_config().is_sharded()
        and a_col % TILE == 0
        and b_col == a_col + HV
        and 1 <= HV
        and all(t.dtype == ttnn.bfloat16 for t in (dt_bias, A_neg))
    )


def prefill_gates(x, *, a_col, b_col, HV, dt_bias, A_neg):
    """Compute prompt gates from raw a and b columns in the fused projection output.

    beta = sigmoid(b), g = A_neg * softplus(a + dt_bias). The kernel matches the stock
    slices, layout conversions, sigmoid, add, softplus, and multiply bit for bit.
    Return g, beta as [1, T, HV] bf16 TILE in DRAM. Padding columns may differ.
    """
    device = x.device()
    T = x.shape[1]
    assert gates_fit(x, a_col, b_col, HV, dt_bias, A_neg), (
        "gdn_pconv: these gates are not supported (stock path)"
    )
    GCALLS[0] += 1
    RT, NT = -(-T // TILE), -(-HV // TILE)
    off = b_col % TILE
    b_t0 = b_col // TILE
    NB = (b_col + HV - 1) // TILE - b_t0 + 1
    kdev = _gconsts(device, dt_bias, A_neg, HV, off)
    dram = ttnn.DRAM_MEMORY_CONFIG
    g = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, T, HV]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )
    beta = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, T, HV]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
    )
    work = plan(device, RT)
    tb = 2 * TILE * TILE
    tiles = {"a": 2 * NT, "b": 2 * NB, "db": NT, "an": NT, "k": 2, "g": 2, "beta": 2}
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles[n] * tb,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=GCB[n], data_format=ttnn.bfloat16, page_size=tb
                )
            ],
        )
        for n in GCB
    ]
    reader_ct = [NT, NB]
    for t in (x, kdev):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [NT]
    for t in (g, beta):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    x_pitch = list(x.padded_shape)[-1] // TILE
    for c, i0, n in work:
        rd[c.x][c.y] = [
            i0,
            n,
            x_pitch,
            a_col // TILE,
            b_t0,
            x.buffer_address(),
            kdev.buffer_address(),
        ]
        wr[c.x][c.y] = [i0, n, g.buffer_address(), beta.buffer_address()]
        cp[c.x][c.y] = [n]

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
            kernel("gdn_pgates_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("gdn_pgates_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "gdn_pgates_compute.cpp",
                [NT, int(off != 0), NB],
                cp,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                ),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([x, kdev, g, beta], program)
    return g, beta


def stock_gates(x, *, a_col, b_col, HV, dt_bias, A_neg, dram=False):
    """Compute prompt gates with stock ttnn ops for comparison."""
    T = x.shape[1]
    mc = None if (dram or T > 512) else ttnn.L1_MEMORY_CONFIG
    a_raw = ttnn.to_layout(x[:, :, a_col : a_col + HV], ttnn.TILE_LAYOUT)
    b_raw = ttnn.to_layout(x[:, :, b_col : b_col + HV], ttnn.TILE_LAYOUT)
    beta = ttnn.sigmoid(b_raw, memory_config=mc)
    sp = ttnn.softplus(ttnn.add(a_raw, dt_bias, memory_config=mc), memory_config=mc)
    g = ttnn.multiply(A_neg, sp, memory_config=mc)
    return g, beta
