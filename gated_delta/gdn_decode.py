# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""Run one Gated DeltaNet decode step in a single device launch.

The kernel computes q/k L2 norms, gates, decay, the state update in place, and o = q S'.
It optionally applies gated RMSNorm. It replaces recurrent_gated_delta_rule_decode_inplace_ttnn
and the surrounding reshapes, repeat_interleave, gate math, and normalization.

  o = gated_delta_decode(x, a, b, state, num_heads=16, num_v_heads=48, cols=(0, 64, 128), ...)

x: the conv output, flat [B, 1, W] (q, k, v by tile column cols). a, b: flat [B, 1, W'] with head
hv's raw a (b) at element column a_col + hv (b_col + hv), turned into g and beta in the kernel
with raw_gates = (a_neg, dt_bias); without raw_gates, a and b hold g and beta themselves. state:
[B, HV, K, V], updated in place. gate (flat [B, 1, W'']) and norm_weight ([1, 1, V]): the output's
gated RMSNorm. Returns o, flat [B, 1, HV*V].

The multi-row kernel in gdn_verify.py handles decode with R = 1. It computes norms once per
packed group of up to 32 heads. The multi-row and single-row paths use the same arithmetic.
"""

import gdn_verify
import ttnn
from gdn_verify import CB, DST, UNPACK_TO_DEST, _ones, _pitch  # noqa: F401 (re-exported helpers)


def gated_delta_decode(
    x,
    a,
    b,
    state,
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
    conv=None,
):
    assert list(x.shape)[-2] == 1, f"gdn_decode: one token, got {x.shape}"
    return gdn_verify.gated_delta_verify(
        x,
        a,
        b,
        state,
        [],
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        cols=cols,
        a_col=a_col,
        b_col=b_col,
        raw_gates=raw_gates,
        gate=gate,
        gate_col=gate_col,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        scale=scale,
        out_dtype=out_dtype,
        memory_config=memory_config,
        clip=clip,
        conv=conv,
    )
