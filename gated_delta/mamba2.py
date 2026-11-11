# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Run a Nemotron-H Mamba-2 mixer in TT-NN.

This module uses the dimensions of Nemotron 3 Nano 30B-A3B and Nemotron 3.5 Lightning 30B-A3B.
Each call processes T tokens for one user with this directory's kernels:

- in_proj as two matmuls: z [T, 4096] and [x | B | C | dt_raw] [T, 6208] (the checkpoint's rows 0-4095 and
  4096-10303; one matmul's output would put dt_raw at element column 10240, the same kernels either way);
- gdn_conv.gated_conv_step: the causal conv over x, B and C (4 taps, the bias, SiLU), its state in place;
- gdn_verify.gated_delta_verify's SSM mode: the recurrence with dt = softplus(dt_raw + dt_bias) (floored at
  time_step_min), exp(A dt), D x, the state [HV, N, P] in float32 in place;
- the gated RMSNorm as the checkpoint's mixer (norm_before_gate=False): g = y silu(z), then each group of 512
  channels divided by its RMS (eps 1e-5) and times the weight. The group means come from one matmul with a
  [4096, 32] matrix of 1/512 (group c // 512), and rsqrt(mean + eps) goes back to the channels, times the
   weight, through a [32, 4096] matrix (row g: the weight on group g's channels).
   These use stock ops without reshaping a tiled tensor;
- out_proj.

The mixer's input is the block's RMSNorm output, its output goes into the block's residual add: both are the
caller's (the assembled model's). A prompt goes through in calls of up to 29 tokens (gdn_conv keeps the 3 state
rows in the tile of its tokens); decode is T = 1.

  mx = Mamba2Mixer(device, w)          # w: host weight tensors keyed by checkpoint names
  # Set weight_dtype for both projections, or in_proj_dtype and out_proj_dtype separately.
  # For example: in_proj_dtype=ttnn.bfloat4_b, out_proj_dtype=ttnn.bfloat8_b.
  cache = mx.new_cache()               # [conv state [1, 3, 6144] bf16, SSM state [1, 64, 128, 64] fp32]
  out = mx(h, cache)                   # h [1, T, 2688] bf16 TILE on the device; cache updated in place
  out, parts = mx(h, cache, keep=True) # parts: z, xd, conv, y, yz, yn on the device (for checks; caller frees)

"""

import sys
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import gdn_conv
import gdn_verify

TILE = 32
HIDDEN, H, HV, N, P = 2688, 8, 64, 128, 64  # hidden, groups, heads, state size, head dim
INTER = HV * P  # 4096: z, x and y
C = INTER + 2 * H * N  # 6144: the conv's channels [x | B | C]
XD = C + HV  # 6208: [x | B | C | dt_raw]
GROUP = INTER // H  # 512: the gated norm's group
COLS = ((INTER + H * N) // TILE, INTER // TILE, 0)  # C, B, x by tile column in the conv's output
MAX_T = TILE - 3  # gdn_conv: 29 tokens a call


def hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,  # the packer accumulates the K blocks' partial sums in L1
    )


class Mamba2Mixer:
    def __init__(
        self,
        device,
        w,
        *,
        weight_dtype=ttnn.bfloat16,
        in_proj_dtype=None,
        out_proj_dtype=None,
        state_dtype=ttnn.float32,
        out_dtype=ttnn.bfloat16,
        floor=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        cfg = w["config"]
        assert cfg["n_groups"] == H, cfg
        self.device, self.mc, self.sd, self.od = device, memory_config, state_dtype, out_dtype
        self.eps = float(cfg["eps"])
        self.floor = cfg["time_step_min"] if floor is None else floor
        self.cfg = hifi4()
        win = w["mixer.in_proj.weight"].float()
        assert list(win.shape) == [INTER + XD, HIDDEN], win.shape

        def put(t, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Prequantized weights convert exactly when they use the target format's grid
        # in 16-element blocks along outputs: rows of [in, out] within each tile face.
        ind = in_proj_dtype or weight_dtype
        self.w_z = put(win[:INTER].t(), ind)  # [2688, 4096]
        self.w_xd = put(win[INTER:].t(), ind)  # [2688, 6208]
        self.w_out = put(
            w["mixer.out_proj.weight"].float().t(), out_proj_dtype or weight_dtype
        )  # [4096, 2688]
        cw = w["mixer.conv1d.weight"].float()  # [6144, 1, 4]
        self.taps = [put(cw[:, 0, k].reshape(1, 1, C), ttnn.bfloat16) for k in range(4)]
        self.bias = put(w["mixer.conv1d.bias"].float().reshape(1, 1, C), ttnn.bfloat16)
        gm = torch.zeros(INTER, TILE)
        gw = torch.zeros(TILE, INTER)
        nw = w["mixer.norm.weight"].float()
        for g in range(H):
            gm[g * GROUP : (g + 1) * GROUP, g] = 1.0 / GROUP
            gw[g, g * GROUP : (g + 1) * GROUP] = nw[g * GROUP : (g + 1) * GROUP]
        self.gmean = put(gm, ttnn.bfloat16)  # 1/512 and 0: exact in bf16
        self.gweight = put(gw, ttnn.bfloat16)  # the weight as stored (bf16)
        self.gates = (
            (-torch.exp(w["mixer.A_log"].float())).tolist(),
            w["mixer.dt_bias"].float().tolist(),
        )
        self.D = w["mixer.D"].float().tolist()

    def free(self):
        for t in (self.w_z, self.w_xd, self.w_out, *self.taps, self.bias, self.gmean, self.gweight):
            ttnn.deallocate(t)

    def new_cache(self):
        z = {"layout": ttnn.TILE_LAYOUT, "device": self.device, "memory_config": self.mc}
        return [
            ttnn.zeros([1, 3, C], dtype=ttnn.bfloat16, **z),
            ttnn.zeros([1, HV, N, P], dtype=self.sd, **z),
        ]

    def __call__(self, h, cache, keep=False):
        T = h.shape[-2]
        assert 1 <= T <= MAX_T and list(h.shape) == [1, T, HIDDEN], h.shape
        conv_state, state = cache
        mm = {"compute_kernel_config": self.cfg, "memory_config": self.mc}
        z = ttnn.linear(h, self.w_z, dtype=ttnn.bfloat16, **mm)
        xd = ttnn.linear(h, self.w_xd, dtype=ttnn.bfloat16, **mm)
        co = gdn_conv.gated_conv_step(
            xd, conv_state, self.taps, self.bias, channels=C, x_col=0, memory_config=self.mc
        )
        y = gdn_verify.gated_delta_verify(
            co,
            xd,
            xd,
            state,
            None,
            num_heads=H,
            num_v_heads=HV,
            head_k_dim=N,
            head_v_dim=P,
            cols=COLS,
            a_col=C,
            b_col=C,
            raw_gates=self.gates,
            out_dtype=self.od,
            memory_config=self.mc,
            ssm_d=self.D,
            ssm_floor=self.floor or None,
        )
        sz = ttnn.silu(z, memory_config=self.mc)
        yz = ttnn.multiply(y, sz, dtype=ttnn.bfloat16, memory_config=self.mc)
        sq = ttnn.multiply(yz, yz, dtype=ttnn.float32, memory_config=self.mc)
        ms = ttnn.matmul(
            sq, self.gmean, dtype=ttnn.float32, **mm
        )  # [1, T, 32]: each group's mean square
        mse = ttnn.add(ms, self.eps, memory_config=self.mc)
        rs = ttnn.rsqrt(mse, memory_config=self.mc)
        scale = ttnn.matmul(
            rs, self.gweight, dtype=ttnn.float32, **mm
        )  # [1, T, 4096]: rsqrt times the weight
        yn = ttnn.multiply(yz, scale, dtype=ttnn.bfloat16, memory_config=self.mc)
        out = ttnn.linear(yn, self.w_out, dtype=ttnn.bfloat16, **mm)
        for t in (sz, sq, ms, mse, rs, scale):
            ttnn.deallocate(t)
        if keep:
            return out, {"z": z, "xd": xd, "conv": co, "y": y, "yz": yz, "yn": yn}
        for t in (z, xd, co, y, yz, yn):
            ttnn.deallocate(t)
        return out
