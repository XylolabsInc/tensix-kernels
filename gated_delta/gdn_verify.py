# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Run R Gated DeltaNet decode steps in order in one device launch.

The multi-row kernel (kernels/gdn_verify_*.cpp) supports speculative verification through
ttnn.generic_op. Each row matches the decode step (gdn_decode.py) bit for bit:
q/k L2 normalization, gates, decay, the delta-rule update and o, optionally the gated RMSNorm of the output; the
state after rows 0..R-2 goes to the slots, the state after the last row back into `state`.

  o = gated_delta_verify(x, a, b, state, slots, num_heads=16, num_v_heads=48, cols=..., ...)

x: the conv output, flat [B, R, W] (q, k, v by tile column cols; row r = token r). a, b: flat
[B, R, W'] with head hv's raw a (b) at element column a_col + hv (b_col + hv), turned into g and
beta in the kernel with raw_gates = (a_neg, dt_bias); without raw_gates, a and b hold g and beta
themselves. state: [B, HV, K, V], read once, then the state after the last row written into it.
slots: R - 1 tensors laid out as the state (dtype, layout, memory config): slots[r] = the state
after row r. gate (flat [B, R, W'']) and norm_weight ([1, 1, V]): the output's gated RMSNorm.
Returns o, flat [B, R, HV*V]. R is 1..32 (one tile row); R = 1 is the decode step.

The reader zeros pages with NOC reads. Gates, q/k norms, and gated output norms operate on
packed groups of up to 32 // R heads. Data-movement cores gather and scatter rows with byte
copies. Decode uses this kernel at R = 1. clip=c clamps the output to [-c, c] in the output
stage, matching ttnn.clip after the kernel when c rounds identically in both paths.
"""

import math
import os

import gdn_scan
import ttnn
from gdn_scan import TILE, f32_bits, head_cores, source, tile_bytes

# The CB plan of kernels/gdn_verify_*.cpp. qg, kg: a group's raw q and k rows (reader -> compute);
# qn, kn: the same normalized (compute -> reader); q, k, v: one item's rows in row 0 (reader ->
# compute); gab: the group's a and b pages; gg: its gate rows; orow: an item's o (compute -> writer);
# oasm: the group's o rows (writer -> compute); rbuf: the reader's scratch and staging; stage: the
# writer's.
CB = {
    "qg": 0,
    "kg": 1,
    "v": 2,
    "gab": 3,
    "oasm": 4,
    "gg": 5,
    "S0": 6,
    "ones": 7,
    "q": 8,
    "k": 9,
    "sd": 10,
    "dec": 11,
    "bt": 12,
    "dl": 13,
    "kt": 14,
    "Sout": 15,
    "out": 16,
    "t4a": 17,
    "t4b": 18,
    "t4c": 19,
    "qk": 20,
    "t1a": 21,
    "t1b": 22,
    "t16": 23,
    "gd": 24,
    "gn": 25,
    "rbuf": 26,
    "qn": 27,
    "stage": 28,
    "Sc": 29,
    "kn": 30,
    "orow": 31,
}
assert len(set(CB.values())) == len(CB) and max(CB.values()) < 32, "gdn_verify: the CB plan"
# dt_bias and a_neg reach the SFPU in fp32 through copy_tile's unpack to Dst.
# Other CBs use the default unpack mode unless overridden; the FPU reads fp32 at tf32.
UNPACK_TO_DEST = [ttnn.UnpackToDestMode.Default] * 64  # covers ttnn builds with up to 64 CB entries
for _n in ("gd", "gn"):
    UNPACK_TO_DEST[CB[_n]] = ttnn.UnpackToDestMode.UnpackToDestFp32
# Dst tiles per acquire: 1, 2, or 4 (half of Dst with fp32 accumulation).
DST = int(os.environ.get("GDN_DEC_DST", "4"))
assert DST in (1, 2, 4), "GDN_DEC_DST: 1, 2 or 4"
SCR = 2 * 32 * 128  # the reader's gate scratch: a and b of 32 values, 128 bytes each
# GDN_NARROW restricts per-call SFPU chains to rows used by items. ITEM_ITER is
# 2 ceil(items / 4) for the largest group; groups above 16 items use full-tile ops.
# Zero selects full-tile ops throughout. Used elements receive the same math either way.
NARROW = os.environ.get("GDN_NARROW", "1") != "0"
# POISON tests unused rows in gate pages and gathered q, k, gate, and o sets with NaNs.
# An unintended read propagates NaN to the output.
# 2, the control: the item slots' unused rows too, which the outer product reads (outputs change).
POISON = 0
# GDN_SSM_INP32 selects the LLK's fp32 softplus for SSM mode. Some low-precision
# implementations use a degree-6 polynomial and clamp the tail to zero below -5.
# Softplus remains positive there (up to about 6.7e-3). A zero dt suppresses both
# decay and the input update.
SSM_DEFINES = "#define INP_FLOAT32 1\n" if os.environ.get("GDN_SSM_INP32", "1") != "0" else ""
# F32S applies only to fp32 DeltaNet states. Zero uses the FPU state-update path.
# Mode 1 uses two passes and computes decay in Dst. Mode 2 uses one update pass and
# computes k S from a separate FPU-readable copy of the state.
F32S = int(os.environ.get("GDN_F32S", "1"))
assert F32S in (0, 1, 2), "GDN_F32S: 0, 1 or 2"
# With F32S, GDN_DN_INP32 also selects fp32 softplus for DeltaNet. This preserves
# small positive decay steps that a low-precision softplus tail can round to zero.
DN_INP32 = os.environ.get("GDN_DN_INP32", "1") != "0"
# GDN_F32S_FULL=1 compares full-tile exp and mul_binary_tile with the narrow decay path.
F32S_DEFINES = "#define GDN_FULL_EXP 1\n" if os.environ.get("GDN_F32S_FULL", "0") == "1" else ""


def item_iter(items):
    """SFPU iterations a face for the rows of `items` items (one iteration covers rows 0-3 in every
    other column, two cover rows 0-3 whole); 0 above 16 items (the full ops)."""
    return 0 if items > 16 else 2 * -(-items // 4)


_ONES = {}


def _ones(device):
    if id(device) not in _ONES:
        import torch

        _ONES[id(device)] = ttnn.from_torch(
            torch.ones(1, TILE, TILE),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _ONES[id(device)]


def _pitch(t):
    return list(t.padded_shape)[-1] // TILE


# Fused convolution packs taps and bias per core: row c of five pages (w0..w3, bias)
# holds channel tile c. Cache by taps tensor and channel plan. The first call reads
# tensors back from the device and must run before trace capture.
_FOLD_TAPS = {}
FOLD_SKIP = int(
    os.environ.get("GDN_FOLD_SKIP", "0")
)  # diagnostic only; use 0 for correct outputs


def _fold_cores(device, BH, G):
    """Distribute whole key heads over cores, keeping their G value heads together.

    Each key head's convolution channels then belong to one core. Balance the number of
    key heads across the compute grid.
    """
    g = device.compute_with_storage_grid_size()
    units = BH // G
    ncores = min(units, g.x * g.y)
    base, extra = divmod(units, ncores)
    out, h = [], 0
    for i in range(ncores):
        nh = (base + (1 if i < extra else 0)) * G
        out.append((ttnn.CoreCoord(i // g.y, i % g.y), h, nh))
        h += nh
    return out


def _fold_chans(plan, H, HV, kt, vt, cols):
    """Per core, the conv state's channel tile of each of its channel tiles, in the kernel's order: per key head on the
    core q's kt tiles, then k's kt; then each head's v vt. The conv's channel 0 is x's tile column 0."""
    G = HV // H
    out = []
    for _, h0, nh in plan:
        if h0 % G or nh % G:
            raise ValueError(
                f"gdn_verify conv fold: heads {h0}..{h0 + nh - 1} split a key head ({G} value heads)"
            )
        ch = []
        for ik in range(nh // G):
            hk = ((h0 + ik * G) % HV) // G
            ch += [cols[0] + hk * kt + t for t in range(kt)] + [
                cols[1] + hk * kt + t for t in range(kt)
            ]
        for hl in range(nh):
            hv = (h0 + hl) % HV
            ch += [cols[2] + hv * vt + t for t in range(vt)]
        if len(ch) > TILE:
            raise ValueError(
                f"gdn_verify conv fold: {len(ch)} channel tiles on a core, at most {TILE}"
            )
        out.append(ch)
    return out


# KDA dt_bias: one fp32 page per value head. Rows 0..Kt-1 hold its K values;
# row t holds channels 32 t .. 32 t + 31. Cache by tensor.
_KDA_DB = {}


def _kda_db(device, dt_bias, HV, K):
    key = (id(device), id(dt_bias), HV, K)
    if key not in _KDA_DB:
        import torch

        db = torch.as_tensor(dt_bias).float().flatten()
        assert db.numel() == HV * K, f"KDA dt_bias: {db.numel()} values, want {HV * K}"
        kt = K // TILE
        # After HV dt_bias pages: E_i (row i ones) and G_i (column i ones), i < kt.
        pages = torch.zeros(HV + 2 * kt, TILE, TILE)
        for h in range(HV):
            pages[h, :kt] = db[h * K : (h + 1) * K].view(kt, TILE)
        for i in range(kt):
            pages[HV + i, i, :] = 1.0
            pages[HV + kt + i, :, i] = 1.0
        t = ttnn.from_torch(
            pages,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _KDA_DB[key] = (dt_bias, t)
    return _KDA_DB[key][1]


def fold_fits(device, BH, H, HV, kt, vt):
    """Whether the fused convolution takes this call's heads (whole key heads and at most 32 channel tiles a core)."""
    try:
        _fold_chans(_fold_cores(device, BH, HV // H), H, HV, kt, vt, (0, H * kt, 2 * H * kt))
    except ValueError:
        return False
    return True


def _fold_taps(device, taps, bias, chans):
    key = (id(device), *(id(t) for t in taps), id(bias), tuple(map(tuple, chans)))
    if key not in _FOLD_TAPS:
        import torch

        w = [ttnn.to_torch(t).float().flatten() for t in taps]
        bv = None if bias is None else ttnn.to_torch(bias).float().flatten()
        pages = []
        for ch in chans:
            blk = torch.zeros(5, TILE, TILE)
            for c, ct in enumerate(ch):
                sl = slice(ct * TILE, (ct + 1) * TILE)
                for k in range(4):
                    blk[k, c] = w[k][sl]
                if bv is not None:
                    blk[4, c] = bv[sl]
            pages.append(blk)
        packed = ttnn.from_torch(
            torch.cat(pages),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _FOLD_TAPS[key] = (
            taps,
            bias,
            packed,
        )  # the tensors held with it, so their ids are not reused
    return _FOLD_TAPS[key][2]


def _fold_plan(x, conv, plan, H, HV, kt, vt, R, cols, ssm, tiles, fmt, rbuf, out_dtype, gnorm):
    """Validate fused convolution, update its CB plan in place, and build kernel arguments.

    cb_qg first holds the packed window (WP tiles, a multiple of Kt so gathers start
    at the CB base). cb_kg holds WT packed tap tiles. CB 28 carries conv output to
    the reader; writer staging moves to cb_rbuf after the reader's scratch."""
    state, taps, bias, window = conv
    B = math.prod(list(x.shape)[:-2])
    C = list(state.shape)[-1]
    assert not ssm, "gdn_verify conv fold: the DeltaNet's mode only"
    assert x.dtype == ttnn.bfloat16, (
        f"gdn_verify conv fold: x (the mega output) in bf16, got {x.dtype}"
    )
    assert R <= TILE - 3, f"gdn_verify conv fold: at most {TILE - 3} rows, got {R}"
    assert len(taps) == 4, f"gdn_verify conv fold: kernel size 4, got {len(taps)} taps"
    assert C % TILE == 0 and list(state.shape) == [B, 3, C], f"conv state {state.shape}"
    if window is not None:
        assert list(window.shape) == [B, 3 + R, C], f"conv window {window.shape}"
    for t in (
        (state, *taps) + (() if bias is None else (bias,)) + (() if window is None else (window,))
    ):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT, (t.shape, t.dtype)
    chans = _fold_chans(plan, H, HV, kt, vt, cols)
    assert max(max(ch) for ch in chans) < C // TILE, (
        "gdn_verify conv fold: q, k and v inside the conv's channels"
    )
    packed = _fold_taps(x.device(), taps, bias, chans)
    ncmax = max(len(ch) for ch in chans)
    wp = max(kt, -(-(3 + R) // kt) * kt)
    wt = max(kt, -(-5 // kt) * kt)
    tiles["qg"], tiles["kg"] = wp, wt
    fmt["stage"], tiles["stage"] = ttnn.bfloat16, R  # CB 28: the conv's output (bf16, R tiles)
    tx = tile_bytes(ttnn.bfloat16)
    win = window is not None
    # cb_rbuf holds reader scratch, fx and fs (input and conv-state staging),
    # head-output staging, shifted conv-state pages ws, and optional window pages.
    fx = rbuf
    nhmax = max(nh for _, _, nh in plan)
    gt_bytes = (
        tile_bytes(fmt["gg"]) if gnorm else 0
    )  # every head's gate tiles after fs in reader scratch
    stage = fx + 2 * ncmax * tx + nhmax * vt * gt_bytes
    ws = stage + vt * tile_bytes(out_dtype)
    total = ws + ncmax * tx * (2 if win else 1)
    tiles["rbuf"] = -(-total // tile_bytes(ttnn.float32))
    # GDN_FOLD_SKIP (a timing test only, wrong outputs): 1 no shift or window, 2 no packing, 4 no conv ops, 8 no gather
    skip = f"#define GDN_FOLD_SKIP {FOLD_SKIP}\n" if FOLD_SKIP else ""
    return {
        "state": state,
        "packed": packed,
        "window": window,
        "rt": lambda ci: [
            0,
            C // TILE,
            state.buffer_address(),
            packed.buffer_address(),
            window.buffer_address() if win else 0,
            5 * ci,
        ],
        "reader": (
            f"#define GDN_FOLD 1\n#define GDN_FOLD_WP {wp}\n#define GDN_FOLD_WT {wt}\n"
            f"#define GDN_FOLD_WINDOW {int(win)}\n#define GDN_FOLD_NCMAX {ncmax}\n" + skip
        ),
        "writer": (
            f"#define GDN_FOLD 1\n#define GDN_FOLD_STAGE {stage}\n#define GDN_FOLD_FX {fx}\n#define GDN_FOLD_WS {ws}\n"
            f"#define GDN_FOLD_NCMAX {ncmax}\n#define GDN_FOLD_WINDOW {int(win)}\n"
        ),
        "wrt": [
            cols[0],
            cols[1],
            cols[2],
            0,
            C // TILE,
            state.buffer_address(),
            window.buffer_address() if win else 0,
        ],
        # the reader tells the writer that its staging is complete (the writer resets it)
        "semaphores": lambda cores: [
            ttnn.SemaphoreDescriptor(
                id=0, core_type=ttnn.CoreType.WORKER, core_ranges=cores, initial_value=0
            )
        ],
        "compute": (
            f"#define GDN_FOLD 1\n#define GDN_FOLD_WP {wp}\n#define GDN_FOLD_WT {wt}\n"
            f"#define GDN_FOLD_BIAS {int(bias is not None)}\n" + skip
        ),
    }


def gated_delta_verify(
    x,
    a,
    b,
    state,
    slots,
    *,
    num_heads,
    num_v_heads,
    head_k_dim=128,
    head_v_dim=128,
    cols=(0, 0, 0),
    a_col=0,
    b_col=0,
    raw_gates=None,
    gate=None,
    gate_col=0,
    norm_weight=None,
    norm_eps=1e-6,
    scale=None,
    out_dtype=ttnn.bfloat16,
    memory_config=None,
    clip=None,
    ssm_d=None,
    ssm_floor=None,
    conv=None,
    kda=None,
):
    """Run DeltaNet, KDA, or Mamba-2 updates with optional fused convolution.

    ssm_d supplies D per value head (HV floats) and enables SSM mode. x holds [x | B | C];
    cols = (C's column, B's column, x's column). head_k_dim is state size N and head_v_dim
    is head dimension P; K and V may differ. raw_gates = (A, dt_bias), A = -exp(A_log).
    a holds raw dt; b is unused, so pass a again. Gate and norm weight are absent.
    o = C h_dec + (C . B) dt x + D x. ssm_floor sets the minimum dt after softplus;
    None disables the floor.

    conv = (conv_state, taps, bias, window) fuses gdn_conv.gated_conv_step. x is the fused
    projection output, with convolution channel 0 at tile column 0. Each core computes
    its heads' channels, shifts conv_state in place, and writes window if supplied.
    The q, k, and v results match gdn_conv bit for bit: the same operands, operations,
    order, and formats. Each key head's G value heads must share a core, with at most
    32 channel tiles per core, so each conv-state page has a single writer. Otherwise
    the call raises ValueError and the caller must run gdn_conv separately.
    """
    device = x.device()
    H, HV, K, V = num_heads, num_v_heads, head_k_dim, head_v_dim
    kt, vt = K // TILE, V // TILE
    kv = kt * vt
    ssm = ssm_d is not None
    assert ssm or kt == vt, (
        "gdn_verify: the scratch CBs take K and V blocks alike (K == V outside the SSM mode)"
    )
    if scale is None:
        scale = K**-0.5
    shape = list(x.shape)
    R = shape[-2]
    assert 1 <= R <= TILE, f"gdn_verify: 1 to {TILE} rows, got {x.shape}"
    # slots=None: the states of rows 0 .. R - 2 are not written (a prefill wants only the last)
    assert slots is None or len(slots) == R - 1, (
        f"gdn_verify: {R} rows need {R - 1} slots, got {len(slots)}"
    )
    B = math.prod(shape[:-2])
    BH = B * HV
    for t in (a, b) + (() if gate is None else (gate,)):
        assert list(t.shape)[-2] == R, f"gdn_verify: {R} rows in x, {t.shape}"
    raw = raw_gates is not None
    gnorm = gate is not None
    if ssm:
        assert raw and not gnorm, "gdn_verify SSM: raw gates (A, dt_bias), no gated norm"
        d_vals = list(map(float, ssm_d))
        assert len(d_vals) == HV, f"gdn_verify SSM: {HV} D values, got {len(d_vals)}"
    assert ssm_floor is None or (ssm and ssm_floor > 0), f"gdn_verify: ssm_floor {ssm_floor}"
    if raw:
        a_neg, dt_bias = (list(map(float, v)) for v in raw_gates)
        assert len(a_neg) == HV and len(dt_bias) == HV
    if gnorm:
        wp = list(norm_weight.padded_shape)
        assert wp[-1] == V and math.prod(wp[:-1]) == TILE, f"norm weight {norm_weight.shape}"
    assert math.prod(state.shape) == BH * K * V, (state.shape, BH, K, V)
    for s in slots or ():
        assert (
            list(s.shape) == list(state.shape)
            and s.dtype == state.dtype
            and s.layout == state.layout
            and s.memory_config() == state.memory_config()
        ), f"gdn_verify: a slot {s.shape} {s.dtype} unlike the state {state.shape} {state.dtype}"
    # The row copies and the packed rows move 16-element face rows: 2- or 4-byte elements only.
    flat = (ttnn.bfloat16, ttnn.float32)
    assert x.dtype in flat and out_dtype in flat and (not gnorm or gate.dtype in flat), (
        f"gdn_verify: x, gate and o in bf16 or fp32, got {x.dtype}, {out_dtype}"
    )
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if clip is not None:
        assert math.isfinite(clip) and clip > 0, f"gdn_verify: clip {clip}"
        # bit for bit ttnn.clip after the kernel only while c rounds to the same bf16 by truncation
        # and to nearest (the kernel's pack and ttnn.clip's store of c need not round alike)
        assert out_dtype != ttnn.bfloat16 or f32_bits(clip) & 0xFFFF < 0x8000, (
            f"gdn_verify: clip {clip} rounds differently to bf16 by truncation and to nearest"
        )
    fmt = {n: ttnn.float32 for n in CB}
    fmt.update(qg=x.dtype, kg=x.dtype, v=x.dtype, S0=state.dtype, Sout=state.dtype, Sc=state.dtype)
    fmt.update(out=out_dtype, stage=out_dtype)
    fmt["gg"] = gate.dtype if gnorm else ttnn.bfloat16
    tiles = {n: 1 for n in CB}
    tiles.update(qg=kt, kg=kt, v=vt, gab=2, oasm=vt, gg=vt if gnorm else 1, S0=kv, ones=2 + vt)
    tiles.update(q=kt, k=kt, sd=kv, dec=2, bt=2, dl=vt, kt=kt, Sout=kv, out=2 * vt)
    tiles.update(t4a=vt, t4b=vt, t4c=vt, t16=kv, qn=kt, kn=kt, stage=vt, Sc=kv, orow=vt)
    gbytes = tile_bytes(fmt["gg"])
    # rbuf: the gate scratch, a head's q, k, v (and gate) tiles, the group's v rows; in fp32 pages
    rbuf = SCR + (2 * kt + 2 * vt) * tile_bytes(x.dtype) + vt * gbytes
    tiles["rbuf"] = -(-rbuf // tile_bytes(ttnn.float32))
    f32s = F32S if (not ssm and state.dtype == ttnn.float32) else 0  # fp32 states only
    if kda is not None:
        # KDA: dict(f_col=x's element column of head 0's f, dt_bias=[HV * K] values, a=[HV] exp(A_log) values); the
        # state fp32 on F32S 2's frame, the gated norm (sigmoid), raw b (beta = sigmoid(b)), q, k, v and f bf16
        assert not ssm and conv is None, "gdn_verify KDA: not with the SSM mode or the conv fold"
        assert state.dtype == ttnn.float32, (
            "gdn_verify KDA: an fp32 state (the bf16 one drifts 5-10 % at 256 tokens)"
        )
        assert H == HV and gnorm and raw and x.dtype == ttnn.bfloat16, (
            "gdn_verify KDA: G 1, gnorm, raw gates, bf16 x"
        )
        assert kda["f_col"] % TILE == 0 and len(kda["a"]) == HV, (
            "gdn_verify KDA: f_col a tile column; a per head"
        )
        f32s = 2
        tiles["dec"], tiles["qk"] = max(2, kt), max(1, kt)  # D_i and d's row form
        tiles["ones"] += 2 * kt  # E_i and G_i
        kda_rbuf = (
            rbuf + 2 * kt * tile_bytes(x.dtype) + 2 * tile_bytes(ttnn.float32)
        )  # f's tiles and dt_bias, 2 heads
        tiles["rbuf"] = -(-kda_rbuf // tile_bytes(ttnn.float32))
        kda_db = _kda_db(device, kda["dt_bias"], HV, K)
    if f32s:  # S_dec at fp32 for pass 2 (F32S 1), or the state's FPU copy (F32S 2) in t16
        assert DST >= 2, "gdn_verify: F32S needs GDN_DEC_DST >= 2"
    if f32s == 2:  # S' is packed to sd with Sout and Sc in one format
        fmt["sd"] = state.dtype
        fmt["t16"] = state.dtype
    if ssm:
        tiles["t1b"] = 2  # D per item, moved to [0, 0] as the gates are
        # S' is packed to sd with Sout and Sc in one format; qk holds exp(g) over a tile
        fmt["sd"] = state.dtype
        assert DST >= 2, (
            "gdn_verify: SSM needs GDN_DEC_DST >= 2 (the decay's tile and a state tile)"
        )
    if gnorm:  # the reader lands the weight's tiles in rbuf together, before the gather
        assert vt * tile_bytes(norm_weight.dtype) <= rbuf, "gdn_verify: rbuf too small for w"
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, R, HV * V]), out_dtype, ttnn.TILE_LAYOUT, device, mc
    )
    ones = _ones(device)
    fold = conv is not None
    plan = _fold_cores(device, BH, HV // H) if fold else head_cores(device, BH)
    if fold:
        fx = _fold_plan(
            x, conv, plan, H, HV, kt, vt, R, cols, ssm, tiles, fmt, rbuf, out_dtype, gnorm
        )
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
    gate_t, w_t = (gate, norm_weight) if gnorm else (ones, ones)
    reader_ct = [kt, vt, R, int(POISON), int(ssm), int(f32s == 2)]
    for t in (x, a, b, gate_t, state, ones, w_t):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    if kda is not None:
        reader_ct.extend(ttnn.TensorAccessorArgs(kda_db).get_compile_time_args())
    if fold:
        for t in (fx["state"], fx["packed"], fx["window"] or fx["state"]):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct = [kt, vt, R, int(POISON), int(ssm)]
    for t in (o, state):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    if fold:
        for t in (fx["state"], fx["window"] or fx["state"]):
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # Some ttnn builds need equal runtime-argument lengths for cache-hit refreshes.
    nh_max = max(nh for _, _, nh in plan)

    def per_head(h0, nh):
        """The reader's dt_bias and a_neg per head (RAW; fp32 bits), padded to nh_max heads."""
        out = []
        per = 3 if ssm else 2
        if raw:
            for bh in range(h0, h0 + nh):
                out += [f32_bits(dt_bias[bh % HV]), f32_bits(a_neg[bh % HV])]
                if ssm:
                    out.append(f32_bits(d_vals[bh % HV]))
        return out + [0] * per * (nh_max - nh)

    for ci, (c, h0, nh) in enumerate(plan):
        rd[c.x][c.y] = [
            h0,
            nh,
            H,
            HV,
            _pitch(x),
            *cols,
            _pitch(a),
            a_col,
            _pitch(b),
            b_col,
            tile_bytes(a.dtype),
            tile_bytes(b.dtype),
            int(gnorm),
            _pitch(gate_t) if gnorm else 0,
            gate_col,
            tile_bytes(w_t.dtype) if gnorm else 0,
            x.buffer_address(),
            a.buffer_address(),
            b.buffer_address(),
            gate_t.buffer_address(),
            state.buffer_address(),
            ones.buffer_address(),
            w_t.buffer_address(),
            int(raw),
            *(fx["rt"](ci) if fold else ()),
            *((kda["f_col"] // TILE, kda_db.buffer_address()) if kda is not None else ()),
            *per_head(h0, nh),
        ]
        wr[c.x][c.y] = [
            h0,
            nh,
            HV,
            o.buffer_address(),
            state.buffer_address(),
            int(gnorm),
            *((s.buffer_address() for s in slots) if slots is not None else [0] * (R - 1)),
            *([H, *fx["wrt"]] if fold else []),
        ]
        cp[c.x][c.y] = [
            nh,
            f32_bits(scale),
            f32_bits(gdn_scan.EPS),
            f32_bits(norm_eps),
            f32_bits(1.0 / V),
            *(
                [f32_bits(float(kda["a"][(h0 + i) % HV])) for i in range(nh)] + [0] * (nh_max - nh)
                if kda is not None
                else []
            ),
        ]

    def compute_config():
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        utd = list(UNPACK_TO_DEST)
        if f32s:  # S_dec (F32S 1) and an fp32 state reach Dst as stored, not at tf32
            if f32s == 1:
                utd[CB["t16"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
            if state.dtype == ttnn.float32:
                utd[CB["S0"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
                utd[CB["Sc"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if kda is not None:  # KDA: d and D_i reach Dst as stored (exact), not at tf32
            utd[CB["t1a"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
            utd[CB["dec"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if ssm:  # D per item reaches Dst in fp32, as dt_bias and A do
            utd[CB["t1a"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
            # Decay tiles and fp32 state reach Dst as stored, without tf32 truncation.
            utd[CB["qk"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
            if state.dtype == ttnn.float32:
                utd[CB["S0"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
                utd[CB["Sc"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = utd  # a field, not a constructor argument
        return cfg

    def kernel(name, ct_args, rt, config, defines=""):
        return ttnn.KernelDescriptor(
            kernel_source=defines + source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ct_args,
            runtime_args=rt,
            config=config,
        )

    program = ttnn.ProgramDescriptor(
        kernels=[
            kernel(
                "gdn_verify_reader.cpp",
                reader_ct,
                rd,
                ttnn.ReaderConfigDescriptor(),
                (fx["reader"] if fold else "") + ("#define GDN_KDA 1\n" if kda is not None else ""),
            ),
            kernel(
                "gdn_verify_writer.cpp",
                writer_ct,
                wr,
                ttnn.WriterConfigDescriptor(),
                fx["writer"] if fold else "",
            ),
            kernel(
                "gdn_verify_compute.cpp",
                # Include nh_max in the descriptor so program caches distinguish
                # per-core runtime-argument lengths.
                [
                    kt,
                    vt,
                    int(raw),
                    R,
                    kv,
                    int(gnorm),
                    DST,
                    nh_max,
                    f32_bits(clip) if clip else 0,
                    item_iter(min(32 // R, nh_max) * R) if NARROW else 0,
                    int(ssm),
                    f32_bits(ssm_floor) if ssm_floor else 0,
                    f32s,
                ],
                cp,
                compute_config(),
                (fx["compute"] if fold else "")
                + ("#define GDN_KDA 1\n" if kda is not None else "")
                + (SSM_DEFINES if (ssm or (f32s and DN_INP32)) else "")
                + (F32S_DEFINES if f32s else ""),
            ),
        ],
        semaphores=fx["semaphores"](cores) if fold else [],
        cbs=cbs,
    )
    io = [x, a, b, ones] + ([gate, norm_weight] if gnorm else [])
    if kda is not None:
        io.append(kda_db)
    if fold:
        io += [fx["packed"], fx["state"]] + ([fx["window"]] if fx["window"] is not None else [])
    seen = set()
    io = [t for t in io if not (id(t) in seen or seen.add(id(t)))]
    ttnn.generic_op([*io, *(slots or ()), o, state], program)
    return o
