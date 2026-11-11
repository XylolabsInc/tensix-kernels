# int4: an int4-weight matmul with group scales along K

A matmul for one token (decode) with symmetric int4 weights and one bf16 scale per output channel and 32 inputs
(G = 32), on Tenstorrent Tensix devices through `ttnn.generic_op`. bf16 activations, fp32 accumulation, bf16 output.
A second op turns the same int4 weight into a `bfloat8_b` weight for an ordinary block-float matmul over many rows.

## API (`int4/w4a16.py`)

| Function | What |
|---|---|
| `int4_weight(device, codes, banks=None, max_banks=None)` | codes `[N_out, K]` int8 in -7..7 as the kernel's weight (a dict; `"t"` is the device tensor) |
| `group_scales(device, scales)` | scales `[N_out, K/32]` as `[K/32, N]` bf16 TILE in DRAM |
| `w4a16_linear(x, w, scale, memory_config=None, fidelity="hifi2", **opts)` | y = x @ (q * s).T for one token; x `[1, 1, 1, K]` bf16 TILE interleaved (L1 or DRAM); opts: `cores_per_bank`, `kb`, `kw`, `sbw`, `full_sync`, `split`, `budget` |
| `dequantize(w, scale, out=None)` | q * s as a `bfloat8_b` `[1, 1, K, N]` weight, DRAM width-sharded over the same banks |
| `reference(x, codes, scales)`, `dequantized(codes, scales)` | host references |

Module attributes: `L1_BUDGET` (None: the device's free L1 span less `L1_MARGIN`), `PACK_PRECISE` (the dequantize's
precise `bfloat8_b` packing).

## Formats and limits

- Codes in -7..7 (no -8), stored as sign-magnitude nibbles, 512 bytes per 32 x 32 tile.
- One token per call (x has one row); K a multiple of 1024.
- The weight is DRAM width-sharded over banks 0 .. banks - 1 (`max_banks` caps the count); N is padded to a multiple of
  banks x 32.
- tt-metal's compute API of v0.81 or later (`common/kernels/dense_ops.h`).

## Test

```
python -s int4/tests/test_int4.py
```

The test's docstring lists what is checked and its bars. The matmul runs at HiFi2 by default; pass
`fidelity="hifi4"` for the tighter result.
