# dense_reader: a matmul for decode

`linear.py` is a matmul for Tenstorrent Tensix devices, launched through `ttnn.generic_op` with kernels of its own
(`kernels/`): y = x @ w (\* scale), with bf16 activations, block-float (`bfloat8_b`, `bfloat4_b`) or `bfloat16`
weights, fp32 accumulation, and a bf16 (or float32) output. After the matmul it can apply per-output-column scales,
`act(gate) * up` with SiLU (SwiGLU) or GELU-tanh (GeGLU), a residual add, or keep the fp32 result; it can also keep a
top-k of each output row (`topk=`), and run several weights as one call (block lists).

## API

| Function | What |
|---|---|
| `matmul(x, w, scale, *, memory_config=None, fidelity="hifi2", full_sync=False, topk=None, out=None, col0=0, w_up=None, max_banks=None, **opts)` | the op; x `[1, 1, M, K]` bf16 TILE interleaved (DRAM or L1); w a weight, or a block list; scale `[1, 1, 1, N]` bf16 or None |
| `linear(x, w, scale, memory_config=None, activation=None, out_dtype=None, **opts)` | the same for x of any rank; activation None, "silu", "swiglu" or "geglu"; w may be `(gate, up)` or a block list |
| `native_weight(device, t, dtype, col_scale=False, sharded=True, max_banks=None)` | a float `[out, in]` matrix as a weight `[1, 1, in, out]` and its scale tensor |
| `interleave_glu(gate, up)` | `[gate \| up]` with alternating output tiles, for the fused GLU |
| `scale_tensor(device, s)` | per-column scales as `[1, 1, 1, N]` bf16 TILE |
| `topk_result(device, records, rows, k)`, `topk_records(r, rows, k)`, `records_host(t, rows, k)` | the top-k records' tensor and their decode |
| `block_float(t, dtype)`, `reference(x, w, scale=None)` | host references |
| `plan`, `choose`, `choose_blocks` | the work split (for inspection) |
| `BlockListMismatch` | a block list refused because its chunk depth would differ from one call per block |

Options in `**opts`: `cores_per_bank`, `split`, `rows`, `xs`, `kb`, `kw`, `sbw`, `glu`, `glu_act` (0 silu, 1 gelu_tanh),
`residual`, `residual_mode` ("epilogue" or "preload"), `out32`. Module attributes (defaults in the file): `L1_BUDGET`,
`X_BUDGET`, `XS_DEPTH`, `MIN_CHUNKS`, `MIN_CHUNKS_SHAPES`, `NARROW_N`, `SPLIT_SHARE`, `REQ_TILES`, `GLU_ROWS`,
`RECORDS_BF16`, `CONFIGS`, and `common.device.MAX_BANKS` and `common.device.PLACE`.

## Formats and limits

- x: bf16, TILE layout, interleaved in DRAM or L1. Top-k and block lists take up to 32 rows of x (top-k keeps at most
  16 rows and `rows * k <= 64` per core).
- Weights: `bfloat8_b`, `bfloat4_b` or `bfloat16`, TILE layout, DRAM width-sharded or DRAM interleaved. The GLU stage
  and block lists need DRAM width-sharded weights with whole gate/up pairs per bank.
- K and N multiples of 32. The planner refuses a shape that does not fit a core's L1 and says why.
- The fused stages that need unit scales (GLU, residual, `out32`, block lists) take `scale=None`.
- tt-metal's compute API of v0.81 or later (`common/kernels/dense_ops.h`). Tested on ttsim (tt-metal / ttnn 0.81.1).

## Test

```
python -s dense_reader/tests/test_dense_reader.py          # or: pytest -s dense_reader/tests
```

The test's docstring lists what is checked and its bars. The bf16 outputs run at HiFi2, the op's default fidelity;
pass `fidelity="hifi4"` for the tighter result.
