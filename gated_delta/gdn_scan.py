# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Run chunked Gated DeltaNet prefill in one device kernel through ttnn.generic_op.

The kernel processes all heads in one launch per layer call.

The kernels (kernels/, adapted from upstream tt-metal v0.81.1's chunk_gated_delta_rule "mono"
program, Apache-2.0; README.md) put the heads on the compute grid, several per core, and run each
head's chunks in sequence with the 128 x 128 state in L1 (fp32). Per chunk of CHUNK tokens they
compute the decays, the WY matrix and its inverse (Horner), the intra-chunk terms and the state
update; optionally the q/k L2 norm and the q scale. q/k/v may be flat token-major ([B, T, H*K], as
the projection and conv emit them; also the fused [q | k | v] tensor with column offsets) or
head-major ([B*H, T, K]); g and beta are [B, T, HV] as the model makes them. Value head hv reads
key head hv // (HV / H), upstream's repeat_interleave.

Entry points:
- gated_delta_rule(q, k, v, g, beta, initial_state, ...): the op; returns (o, final_state).
- chunk_gated_delta_rule_seq_adapter(...): upstream's ttnn_delta_rule_seq adapter interface
  ([B, T, H, *] in, [B, T, H, V] and [B, H, K, V] out), for a DeltaNet forward in chunk mode.
- torch_recurrent(...): the token-by-token recurrence in fp64 (the reference).

The kernels are JIT-compiled from source at the first call.
"""

import math
import os
import struct
from pathlib import Path

import torch
import ttnn

KERNELS = Path(__file__).resolve().parent / "kernels"
CHUNK = 64
EPS = 1e-6
TILE = 32
# The CB plan of kernels/gdn_compute.cpp.
CB = {
    "qr": 0,
    "kr": 1,
    "v": 2,
    "g": 3,
    "beta": 4,
    "eye": 5,
    "tril": 6,
    "ones": 7,
    "S0": 8,
    "sa": 9,
    "gt": 10,
    "q": 11,
    "k": 12,
    "decay": 13,
    "dexp": 14,
    "dfac": 15,
    "lmask": 16,
    "tinv": 17,
    "vbeta": 18,
    "kbeta": 19,
    "u": 20,
    "w": 21,
    "qdec": 22,
    "intra": 23,
    "vnew": 24,
    "kdect": 25,
    "out": 26,
    "final": 27,
    "s1": 28,
    "s2": 29,
    "s3": 30,
    "h": 31,
}
L1_BUDGET = 1_400_000  # circular-buffer budget per core, in bytes
KERNEL_CONFIG_RING = None  # bytes, for worker_l1_kw (None: unset)
# Compute the WY inverse per 32 x 32 diagonal block (1) or over the whole chunk (0).
# Both forms evaluate the same inverse; block inversion reduces matrix work.
BLOCK_INV = os.environ.get("GDN_BLOCK_INV", "1") != "0"
# Destination tiles the compute kernel fills per acquire: 1, or 4 (half of Dst with fp32 accumulation).
DST = int(os.environ.get("GDN_DST", "1"))
assert DST in (1, 2, 4), "GDN_DST: 1, 2 or 4"
OPT = os.environ.get("GDN_OPT", "O2")  # the compute kernel's KernelBuildOptLevel
# F32S updates an fp32 state in Dst without tf32 truncation. It does not apply to bf16 states.
# GDN_SCAN_F32S=0 uses the FPU state-update path. With F32S, GDN_DN_INP32 selects fp32
# softplus so that small positive decay steps are not rounded to zero by a low-precision tail.
F32S = int(os.environ.get("GDN_SCAN_F32S", "1"))
assert F32S in (0, 1), "GDN_SCAN_F32S: 0 or 1"
DN_INP32 = os.environ.get("GDN_DN_INP32", "1") != "0"

_SOURCES = {}
_CONSTS = {}


def _opt_level(name):
    """Find KernelBuildOptLevel by name across ttnn module layouts.

    None selects the kernel's default optimization level.
    """
    for mod in (ttnn, getattr(ttnn, "_ttnn", None)):
        if mod is not None and hasattr(mod, "KernelBuildOptLevel"):
            return getattr(mod.KernelBuildOptLevel, name)
    try:
        from ttnn._ttnn.operations.experimental import fabric_mux

        return getattr(fabric_mux.KernelBuildOptLevel, name)
    except (ImportError, AttributeError):
        return None


def worker_l1_kw(less=None):
    """Return open_device keywords that reduce worker L1 by `less` bytes.

    None reads GDN_WORKER_L1_LESS; zero returns {}. Use the same size for every device open in a
    process. It needs KERNEL_CONFIG_RING.
    """
    if less is None:
        less = int(os.environ.get("GDN_WORKER_L1_LESS", "0"))
    if not less:
        return {}
    assert less > 0 and less % 64 == 0, (
        f"GDN_WORKER_L1_LESS={less}: a positive multiple of 64 bytes"
    )
    assert KERNEL_CONFIG_RING, "gdn_scan.worker_l1_kw: set gdn_scan.KERNEL_CONFIG_RING to the device's default ring"
    most = None
    for mod in (
        getattr(ttnn, "device", None),
        getattr(getattr(ttnn, "_ttnn", None), "device", None),
    ):
        if mod is not None and hasattr(mod, "get_max_worker_l1_unreserved_size"):
            most = int(mod.get_max_worker_l1_unreserved_size())
            break
    assert most is not None, "gdn_scan: no get_max_worker_l1_unreserved_size in this ttnn"
    size = most - KERNEL_CONFIG_RING - less
    print(
        f"gdn_scan: worker L1 {size} B, {less} B under the default: the kernel config ring {KERNEL_CONFIG_RING + less} B",
        flush=True,
    )
    return {"worker_l1_size": size}


def source(name):
    """Return kernel source with gdn_compat.h and gdn_ops.h inlined.

    SOURCE_CODE kernels have no include path of their own."""
    if name not in _SOURCES:
        text = (KERNELS / name).read_text()
        for header in ("gdn_compat.h", "gdn_ops.h"):
            inc = f'#include "{header}"'
            if inc in text:
                text = text.replace(inc, (KERNELS / header).read_text())
        _SOURCES[name] = text
    return _SOURCES[name]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def tile_bytes(dtype):
    if dtype == ttnn.float32:
        return 4096
    if dtype == ttnn.bfloat16:
        return 2048
    raise ValueError(f"gdn_scan: {dtype} not supported (float32 or bfloat16)")


def constants(device, chunk=CHUNK):
    """eye, tril and ones of [chunk, chunk] as one fp32 [3, C, C] tensor, made once per device."""
    key = (id(device), chunk)
    if key not in _CONSTS:
        c = torch.stack(
            [torch.eye(chunk), torch.tril(torch.ones(chunk, chunk)), torch.ones(chunk, chunk)]
        )
        _CONSTS[key] = ttnn.from_torch(
            c.float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _CONSTS[key]


def head_cores(device, n):
    """Distribute n heads over the compute grid in column-major order.

    Return (core, first head, heads) tuples. Each core gets n // ncores heads;
    the first n % ncores cores get one extra."""
    g = device.compute_with_storage_grid_size()
    ncores = min(n, g.x * g.y)
    base, extra = divmod(n, ncores)
    out, h = [], 0
    for i in range(ncores):
        nh = base + (1 if i < extra else 0)
        out.append((ttnn.CoreCoord(i // g.y, i % g.y), h, nh))
        h += nh
    return out


def cb_tiles(ct, kt, vt):
    """Tiles per CB (gdn_compute.cpp), by name."""
    cc, ck, cv, kv = ct * ct, ct * kt, ct * vt, kt * vt
    scr = max(cc, ck, cv, kv)
    return {
        "qr": ck,
        "kr": ck,
        "v": cv,
        "g": ct,
        "beta": ct,
        "eye": cc,
        "tril": cc,
        "ones": cc + vt,
        "S0": kv,
        "sa": kv,
        "gt": cv,
        "q": ck,
        "k": ck,
        "decay": ct,
        "dexp": ct,
        "dfac": ct,
        "lmask": cc,
        "tinv": cc,
        "vbeta": cv,
        "kbeta": ck,
        "u": cv,
        "w": ck,
        "qdec": ck,
        "intra": cc,
        "vnew": cv,
        "kdect": kt * ct,
        "out": 2 * cv,
        "final": kv,
        "s1": scr,
        "s2": scr,
        "s3": scr,
        "h": max(cc, 2),
    }


def _rows(t):
    """(leading rows, T, padded T, last dim) of a TILE tensor."""
    s, p = list(t.shape), list(t.padded_shape)
    return math.prod(s[:-2]), s[-2], p[-2], p[-1]


def gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    *,
    num_heads,
    num_v_heads,
    head_k_dim=128,
    head_v_dim=128,
    scale=None,
    normalize=True,
    chunk=CHUNK,
    flat=True,
    cols=(0, 0, 0),
    out_flat=None,
    out_dtype=ttnn.bfloat16,
    state_dtype=ttnn.float32,
    memory_config=None,
    out_state=None,
    gate=None,
    gate_col=0,
    norm_weight=None,
    norm_eps=1e-6,
    g_col=0,
    beta_col=0,
    raw_gates=None,
):
    """The chunked gated delta rule on the device; returns (o, final_state).

    q, k: flat [B, T, W] (flat=True; head h's columns from tile column cols[0|1] + h*K/32) or
        head-major [B*H, T, K]; bf16 or fp32, TILE, interleaved. Raw, L2-normalized in the kernel
        (normalize=True; q then times scale, default K**-0.5), otherwise supply normalized inputs;
        the kernel still applies q scaling.
    v: flat [B, T, W] (columns from cols[2]) or head-major [B*HV, T, V].
    g (log decay, <= 0), beta: [B, T, HV], fp32 or bf16, TILE.
    initial_state: [B*HV, K, V] in any shape with that tile order, fp32 or bf16; None = zeros.
    o: flat [B, T, HV*V] (out_flat, default = flat) or head-major [B*HV, T, V], out_dtype.
    final_state: [B, HV, K, V], state_dtype; or written into out_state (any shape with that tile
    order, e.g. the layer's own recurrent-state buffer, also when it is initial_state: each head's
    initial state is read before its final state is written, and heads share no pages).
    gate (flat [B, T, W], head hv's V columns from tile column gate_col + hv*V/32; e.g. the fused
    projection's output) and norm_weight ([1, 1, V]): o becomes upstream's rms_norm_gated_ttnn of it,
    rms_norm(o, norm_weight, norm_eps) * silu(gate) per value head (without its clip of silu(gate)
    at 1e4), computed in the kernel.
    raw_gates = (a_neg, dt_bias), HV floats each: g and beta are the raw a and b of upstream's
    projection (flat [B, T, W], head hv at column g_col + hv / beta_col + hv; e.g. both in the fused
    projection output), and the kernel makes g = a_neg * softplus(a + dt_bias) and beta = sigmoid(b) itself."""
    device = q.device()
    H, HV, K, V = num_heads, num_v_heads, head_k_dim, head_v_dim
    assert HV % H == 0 and K % TILE == 0 and V % TILE == 0 and chunk % TILE == 0
    ct, kt, vt = chunk // TILE, K // TILE, V // TILE
    if scale is None:
        scale = K**-0.5
    if out_flat is None:
        out_flat = flat
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG

    gb, T, Tp, _ = _rows(g)
    raw = raw_gates is not None
    if raw or g_col or beta_col:
        assert _rows(beta)[:3] == (gb, T, Tp), (g.shape, beta.shape)
        assert g_col + HV <= g.shape[-1] and beta_col + HV <= beta.shape[-1], (g.shape, g_col)
    else:
        assert g.shape[-1] == HV and list(beta.shape) == list(g.shape), (g.shape, beta.shape)
    if raw:
        a_neg, dt_bias = (list(map(float, x)) for x in raw_gates)
        assert len(a_neg) == HV and len(dt_bias) == HV
    B = gb
    Tt = Tp // TILE
    for t in (q, k, v):
        _, tq, tpq, _ = _rows(t)
        assert tq == T and tpq == Tp, f"q/k/v and g must have the same T: {t.shape} vs {g.shape}"
    if flat:
        pitches = [_rows(t)[3] // TILE for t in (q, k, v)]
        for t, c, w in zip((q, k, v), cols, (H * kt, H * kt, HV * vt)):
            assert _rows(t)[0] == B and c + w <= _rows(t)[3] // TILE, (t.shape, c, w)
    else:
        pitches = [kt, kt, vt]
        assert _rows(q)[0] == B * H and _rows(k)[0] == B * H and _rows(v)[0] == B * HV
    BH = B * HV
    NC = -(-T // chunk)

    consts = constants(device, chunk)
    tiles = cb_tiles(ct, kt, vt)
    fmt = {name: ttnn.float32 for name in CB}
    fmt.update(qr=q.dtype, kr=k.dtype, v=v.dtype, out=out_dtype, final=state_dtype)
    gnorm = gate is not None
    fmt["gt"] = gate.dtype if gnorm else ttnn.bfloat16
    if gnorm:
        assert norm_weight is not None and _rows(gate)[:3] == (B, T, Tp), (gate.shape, g.shape)
        assert gate_col + HV * vt <= _rows(gate)[3] // TILE, (gate.shape, gate_col)
        wp = list(norm_weight.padded_shape)
        assert wp[-1] == V and math.prod(wp[:-1]) == TILE, (
            f"norm weight {norm_weight.shape}: [1, 1, V]"
        )
        w32 = norm_weight  # bf16 or fp32; the reader widens row 0
    fmt["S0"] = initial_state.dtype if initial_state is not None else ttnn.float32
    f32s = F32S if state_dtype == ttnn.float32 else 0  # fp32 states only
    l1 = sum(tiles[n] * tile_bytes(fmt[n]) for n in CB)
    assert l1 <= L1_BUDGET, f"gdn_scan: {l1} B of circular buffers per core at chunk {chunk}"

    o_shape = [B, T, HV * V] if out_flat else [BH, T, V]
    o = ttnn.allocate_tensor_on_device(ttnn.Shape(o_shape), out_dtype, ttnn.TILE_LAYOUT, device, mc)
    if out_state is not None:
        assert out_state.dtype == state_dtype and math.prod(out_state.shape) == BH * K * V
        s_out = out_state
    else:
        s_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, HV, K, V]), state_dtype, ttnn.TILE_LAYOUT, device, mc
        )

    plan = head_cores(device, BH)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in plan])

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

    state_t = initial_state if initial_state is not None else consts
    gate_t, w_t = (gate, w32) if gnorm else (consts, consts)
    reader_ct = [ct, kt, vt]
    for t in (q, k, v, g, beta, consts, state_t, gate_t, w_t):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [ct, kt, vt]
    for t in (o, s_out):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, h0, nh in plan:
        rd[c.x][c.y] = [
            h0,
            nh,
            NC,
            T,
            Tt,
            H,
            HV,
            int(flat),
            *pitches,
            *cols,
            _rows(g)[3] // TILE,
            g_col,
            _rows(beta)[3] // TILE,
            beta_col,
            f32_bits(-1e4) if raw else 0,
            tile_bytes(g.dtype),
            tile_bytes(beta.dtype),
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            g.buffer_address(),
            beta.buffer_address(),
            consts.buffer_address(),
            state_t.buffer_address(),
            int(initial_state is not None),
            int(gnorm),
            gate_t.buffer_address(),
            _rows(gate_t)[3] // TILE if gnorm else 0,
            gate_col,
            w_t.buffer_address(),
            tile_bytes(w_t.dtype) if gnorm else 0,
        ]
        wr[c.x][c.y] = [
            h0,
            nh,
            NC,
            Tt,
            HV,
            int(out_flat),
            o.buffer_address(),
            s_out.buffer_address(),
        ]
        cp[c.x][c.y] = [
            nh,
            NC,
            f32_bits(scale),
            f32_bits(EPS),
            f32_bits(norm_eps),
            f32_bits(1.0 / V),
        ]
        if raw:
            per_head = []
            for bh in range(h0, h0 + nh):
                per_head += [f32_bits(a_neg[bh % HV]), f32_bits(dt_bias[bh % HV])]
            # Some ttnn builds need equal runtime-argument lengths for cache-hit refreshes.
            per_head += [0, 0] * (max(n for _, _, n in plan) - nh)
            cp[c.x][c.y] = [
                nh,
                NC,
                f32_bits(scale),
                f32_bits(EPS),
                f32_bits(norm_eps),
                f32_bits(1.0 / V),
                *per_head,
            ]

    def compute_config():
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        if (
            f32s
        ):  # the state between chunks, and an fp32 initial state, reach Dst as stored, not at tf32
            utd = [ttnn.UnpackToDestMode.Default] * 64  # covers ttnn builds with up to 64 CB entries
            utd[CB["sa"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
            if fmt["S0"] == ttnn.float32:
                utd[CB["S0"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = utd  # a field, not a constructor argument
        return cfg

    def kernel(name, ct_args, rt, config, opt=None, defines=""):
        kw = {} if opt is None else {"opt_level": opt}
        return ttnn.KernelDescriptor(
            kernel_source=defines + source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct_args,
            runtime_args=rt,
            config=config,
            **kw,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel("gdn_reader.cpp", reader_ct, rd, ttnn.ReaderConfigDescriptor()),
            kernel("gdn_writer.cpp", writer_ct, wr, ttnn.WriterConfigDescriptor()),
            kernel(
                "gdn_compute.cpp",
                [
                    ct,
                    kt,
                    vt,
                    int(normalize),
                    tiles["s1"],
                    int(BLOCK_INV),
                    DST,
                    int(gnorm),
                    int(raw),
                    f32s,
                ],
                cp,
                compute_config(),
                # O2: Os can break the LLK's immediate-operand assembly.
                _opt_level(OPT),
                "#define INP_FLOAT32 1\n" if (f32s and DN_INP32) else "",
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    io = [q, k, v, g, beta, consts] + ([gate, w32] if gnorm else [])
    if initial_state is not None and initial_state is not out_state:
        io.append(initial_state)
    try:
        ttnn.generic_op([*io, o, s_out], program)
    except RuntimeError as e:
        if f32s and "kernel config buffer" in str(e):
            raise RuntimeError(
                "gdn_scan: the F32S program is larger than the device's kernel config ring: open the device "
                "with a smaller worker L1 (gdn_scan.worker_l1_kw(), GDN_WORKER_L1_LESS) "
                f"or set GDN_SCAN_F32S=0: {e}"
            ) from e
        raise
    return o, s_out


RELAYOUT = os.environ.get("GDN_RELAYOUT", "permute")


def _head_major(t, B, T, h, d):
    """Convert [B, T, h, d] TILE to [B*h, T, d] TILE with permute and reshape.

    This keeps the tensor tiled while splitting heads, as chunk_gated_delta_rule does.
    """
    t = ttnn.permute(t, (0, 2, 1, 3))
    return ttnn.reshape(t, [B * h, T, d])


def chunk_gated_delta_rule_seq_adapter(
    q,
    k,
    v,
    beta,
    g,
    chunk_size=128,
    scale=None,
    initial_state=None,
    device=None,
    cached_masks=None,
    valid_len=None,
    qkv_head_dims=None,
    return_o_bh=False,
):
    """Upstream's ttnn_delta_rule_seq.chunk_gated_delta_rule_seq_adapter on this kernel: q, k
    [B, T, Hq, K] (raw; L2-normalized in the kernel), v [B, T, H, V], beta and g [B, T, H] (or flat
    q/k/v with qkv_head_dims = (Hq, K, H, V)); returns o [B, T, H, V] ([B*H, T, V] with return_o_bh)
    and the new state [B, H, K, V] in bf16 (fp32 with QWEN_GDN_FP32_STATE=1). chunk_size, device and
    cached_masks are upstream's and unused (the kernel's chunk is CHUNK)."""
    B, T = q.shape[0], q.shape[1]
    flat = True
    if qkv_head_dims is not None:
        Hq, K, H, V = qkv_head_dims
    else:
        Hq, K = q.shape[2], q.shape[3]
        H, V = v.shape[2], v.shape[3]
        if RELAYOUT == "reshape":
            q = ttnn.reshape(q, [B, T, Hq * K])
            k = ttnn.reshape(k, [B, T, Hq * K])
            v = ttnn.reshape(v, [B, T, H * V])
        else:
            q, k = _head_major(q, B, T, Hq, K), _head_major(k, B, T, Hq, K)
            v = _head_major(v, B, T, H, V)
            flat = False
    if valid_len is not None and valid_len < T:
        raise NotImplementedError("gdn_scan adapter: valid_len < T (pad the prompt to T instead)")
    s0 = None if initial_state is None else ttnn.reshape(initial_state, [B * H, K, V])
    fp32_state = os.environ.get("QWEN_GDN_FP32_STATE", "0") != "0"
    o, state = gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        s0,
        num_heads=Hq,
        num_v_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        scale=scale,
        normalize=True,
        flat=flat,
        out_flat=not return_o_bh and flat,
        out_dtype=ttnn.bfloat16,
        state_dtype=ttnn.float32 if fp32_state else ttnn.bfloat16,
    )
    if not return_o_bh:
        if flat:
            o = ttnn.reshape(o, [B, T, H, V])
        else:
            o = ttnn.permute(ttnn.reshape(o, [B, H, T, V]), (0, 2, 1, 3))
    return o, state


def l2norm(x, eps=EPS):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def torch_recurrent(q, k, v, g, beta, state=None, scale=None, normalize=True):
    """Evaluate the gated delta rule token by token in fp64 as a reference.

    q, k [B, T, H, K]; v [B, T, HV, V]; g, beta [B, T, HV]; state [B, HV, K, V] or None.
    Returns o [B, T, HV, V] and the final state [B, HV, K, V], fp64."""
    q, k, v, g, beta = (t.double() for t in (q, k, v, g, beta))
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if normalize:
        q, k = l2norm(q), l2norm(k)
    q = q * (K**-0.5 if scale is None else scale)
    rep = HV // H
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
    S = torch.zeros(B, HV, K, V, dtype=torch.float64) if state is None else state.double().clone()
    o = torch.zeros(B, T, HV, V, dtype=torch.float64)
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt = k[:, t], v[:, t]  # [B, HV, K], [B, HV, V]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        delta = (vt - pred) * beta[:, t][..., None]
        S = S + torch.einsum("bhk,bhv->bhkv", kt, delta)
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def compare(name, ref, got):
    """Format relative L2 error, maximum absolute error, PCC, and norm ratio as one line."""
    ref, got = ref.double().flatten(), got.double().flatten()
    rel = ((got - ref).norm() / ref.norm()).item()
    mx = (got - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    nr = (got.norm() / ref.norm()).item()
    line = f"{name}: rel {rel * 100:.3f} %, max abs {mx:.3e}, PCC {pcc:.7f}, norm {nr:.4f}"
    return line, {"rel": rel, "max": mx, "pcc": pcc, "norm": nr}
