# dense_reader: a weight-streaming matmul for decode

`linear.py` is a matmul for Tenstorrent Tensix devices, launched through `ttnn.generic_op` with kernels of its own
(`kernels/`): y = x @ w (\* scale), with bf16 activations, block-float (`bfloat8_b`, `bfloat4_b`) or `bfloat16`
weights, fp32 accumulation, and a bf16 (or float32) output. It is built for decode, where the weights are read once per
token and the matmul is bound by how fast they stream from DRAM.

## How it works

- **Weights DRAM width-sharded.** A weight `[1, 1, K, N]` is width-sharded over the DRAM banks, one column block per
  bank. Each bank's columns are split over `cores_per_bank` worker cores, placed next to the device's preferred worker
  for that bank where the device can say. Each K row of a core's columns is one contiguous read straight into its
  circular buffer (a buffer page is a weight tile; the block-float tiles carry their own exponents).
- **x streamed with the weight chunks.** For up to 32 rows, a core does not wait for a block of x before the first
  weight chunk: the reader reads the x tiles a chunk needs together with that chunk, under the same read barrier, and
  keeps only a few chunks' worth of x in L1. For more rows, each core holds a block of x and the rows go in slices of
  up to 4 tile rows (8 with `full_sync`), each slice reading the weights once.
- **Split loaders.** The writer RISC of each core loads some of the weight chunks into a second circular buffer: two
  DRAM request streams per core, one on each NoC. The reader's share is 0.5 (every other chunk) by default and can be
  set from 0.5 to 0.75 (`SPLIT_SHARE`; `common/kernels/split.h`). The output is the same bit for bit whatever the
  share.
- **Full-row chunks and L1 accumulation.** A chunk is a few full K rows of the core's columns; the compute
  accumulates each chunk in dest and adds it to the core's fp32 accumulator in L1 with the packer's L1 accumulation.
- **Fused output stages.** After the K loop, one of: per-output-column scales (a bf16 tile row, broadcast); none
  (unit scales); `act(gate) * up` with act silu (SwiGLU) or gelu_tanh (GeGLU), from a weight whose output tiles
  alternate gate and up (`interleave_glu`) or from two weights; a residual added in fp32 before the one bf16 rounding;
  the fp32 accumulator as it is (`out32`).
- **Top-k records in the writer.** With `topk=`, each core's writer keeps a running top-k of each output row (up to 16
  rows) over its columns and writes it as one small record (keys as order-preserving 16-bit codes of the bf16 values,
  then the global column ids). The output itself need not be written. `lm_head_topk/` builds an LM head on this.
- **Block lists.** `matmul(x, [(w0, col0_0), (w1, col0_1), ...], None)` runs two to four weights, each
  DRAM-sharded on its own, as one call: each core runs its slice of every block in turn in one dispatch, and each
  block's output lands at its own column offset of one output tensor. Pairs `(gate, up)` give the GLU per block.

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
`residual`, `residual_mode` ("epilogue" or "preload"), `out32`. Without them the first configuration of `CONFIGS`
whose circular buffers fit the core's L1 is used.

Module attributes. The defaults are starting points chosen from measurements on one device (for example `CONFIGS`
trying 2 cores a bank first); retune them per device:

| Name | Default | What |
|---|---|---|
| `L1_BUDGET` | None | circular-buffer bytes per core; None: the device's free L1 span at the first plan, less `L1_MARGIN` |
| `X_BUDGET` | 600000 | the most bytes of x a core holds in L1 for prefill |
| `XS_DEPTH` | 3 | x groups the reader may run ahead of the compute when x streams |
| `MIN_CHUNKS`, `MIN_CHUNKS_SHAPES` | 0, None | when x streams, halve the chunk depth until a core has at least this many chunks (0: off), for these `(K, N)` shapes only (None: all) |
| `NARROW_N` | 0 | try 1 core per bank first for decode shapes with N at most this (0: off) |
| `SPLIT_SHARE` | 0.5 | the reader RISC's share of the weight chunks with split loaders (0.5 to 0.75) |
| `REQ_TILES` | 0 | cut each weight row's read into requests of at most this many tiles (0: one request a row) |
| `GLU_ROWS` | True | the GLU stage's SFPU work on the valid rows only (one tile row of x, at most 16 rows) |
| `RECORDS_BF16` | False | the top-k result tensor as bfloat16 instead of uint32 (the same bytes) |
| `CONFIGS` | see the file | the configurations tried in order |
| `common.device.MAX_BANKS` | None | the most DRAM banks a weight is sharded over (None: every bank the device reports) |
| `common.device.PLACE` | "optimal" | core placement: next to each bank's preferred worker, or "rows" |

## Limits and assumptions

- x: bf16, TILE layout, interleaved in DRAM or L1. Up to 32 rows stream with the weights; more rows go in row slices.
  Top-k and block lists take up to 32 rows of x (top-k keeps at most 16 rows and `rows * k <= 64` per core).
- Weights: `bfloat8_b`, `bfloat4_b` or `bfloat16`, TILE layout, DRAM width-sharded (any bank count up to the device's)
  or DRAM interleaved. The GLU stage and block lists need DRAM width-sharded weights with whole gate/up pairs per bank.
- K and N multiples of 32. A weight's columns per core must fit the core's L1 together with its accumulator; the
  planner refuses a shape that does not fit and says why.
- The fused stages that need unit scales (GLU, residual, `out32`, block lists) take `scale=None`.
- tt-metal's compute API of v0.81 or later (`common/kernels/dense_ops.h`). Tested on ttsim (tt-metal / ttnn 0.81.1)
  on the default 11 x 10 worker grid and on a reduced worker grid, both with 8 DRAM banks.

## Test

```
python -s dense_reader/tests/test_dense_reader.py          # or: pytest -s dense_reader/tests
```

The test's docstring lists what is checked: the three weight formats at 1, 7 and 32 rows, per-column scales, prefill in
one and two row slices, an interleaved weight, SwiGLU and GeGLU from one and from two weights, the residual in both
modes, the fp32 output, a weight over fewer banks than the device has, and `linear` on a rank-3 input, against x times
the weight's own values in float64; and bit for bit: split shares and request sizes against the defaults, one loader
against two, the top-k records against the op's own output, and block lists against one call per block. The bars
against the reference are a relative error below 1 % and a PCC above 0.9999 for the bf16 outputs, which run at HiFi2,
the op's default fidelity (measured 0.42 to 0.80 %), and 0.1 % for the fp32 output at HiFi4 (measured 0.03 %). HiFi2
makes two of the matrix unit's four multiply passes over the operands' mantissas, so its products leave out low-order
bits that HiFi4 keeps; pass `fidelity="hifi4"` for the tighter result.

## Left out

- An int8 weight path is not included: this package takes the block-float and bf16 formats only.
- A staged-copy loader and a two-chunks-in-flight loader for that int8 path, a ring of transaction ids for the weight
  reads (an experimental option, off by default), and the writer's whole-row reject for the top-k scan (an alternative
  to the per-word reject kept here).
- Timing and bandwidth probes, and defaults chosen from measurements on a particular board (chunk-depth tables per
  model shape, a split share chosen by weight size).
