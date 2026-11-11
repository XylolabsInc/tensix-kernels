# moe: a mixture-of-experts layer that reads only the chosen experts

`moe8.py` runs one MoE layer on a Tenstorrent Tensix device through `ttnn.generic_op`, with kernels of its own
(`kernels/`). Only the k experts each token routed to are read from DRAM. The expert ids are read on the device,
never on the host, so every op can be captured in a trace.

## What it does

| Op | In | Out |
|---|---|---|
| `route(logits, ids, w, k, kind=...)` | the router's float32 logits `[1, 1, R, E]` | the top-k ids `[1, 1, R, k]` (uint16 or uint32) and their weights (bf16), written in place |
| `gate_up(x, P, ids, w, k, act=...)` | one token `x [1, 1, 1, H]` | `h [1, 1, 1, k * I]`: per slot s, `w_s * act(x G_e) * (x U_e)` |
| `down(h, P, ids, k)` | `h` | `y [1, 1, 1, H]`: the sum over the slots of `h_s @ D_e` |
| `gate_up_rows(x, P, ids, w, k, rows)` | up to 32 tokens, each with its own k experts | `h [1, 1, 32, u_max * I]`, by expert of the rows' union |
| `down_rows(h, P, ids, k, rows)` | `h` | `y [1, 1, 32, H]` (rows past `rows` are 0) |

- **route** runs on one core in integer fixed point: the top k by fp32 logit (lower id first among equal keys) with
  softmax weights, or sigmoid scores with an optional score bias, DeepSeek-V3's group-limited routing (`groups=`), a
  routed scaling factor (`scale=`), unnormalized sigmoid weights (`norm=False`), a fixed shared-expert slot
  (`slot=`) and a shared expert's sigmoid gate folded into the router's matmul (`shared=`).
- **gate_up** activations: `"silu"` (SwiGLU), `"gelu_tanh"` (GeGLU) and `"swiglu_oai"` (gpt-oss's clamped SwiGLU,
  `min(g, 7) * sigmoid(1.702 min(g, 7)) * (clamp(u, -7, 7) + 1)`); an optional per-expert gate/up bias.
- **The R-row ops** build the union of the rows' expert ids on every core from the ids tile, and read each expert in
  the union once for all rows.
- **Two loaders** (`loaders=2`): the writer RISC of each core loads some of the weight chunks into a second circular
  buffer, a second DRAM request stream next to the reader's. The outputs are the same bit for bit. The reader's share
  is `SPLIT_SHARE` (0.5 to 0.75; `common/kernels/split.h`).

## The packed weights

`pack(device, gate_up, down)` converts one layer's experts at load time: `gate_up [E, H, 2I]` as `[up | gate]`
columns, `down [E, I, H]`. It returns the dict `P` the ops take. The experts are stored as whole BFP4_b tiles
(`bfp4_tiles`, the same bytes as tt-metal's host conversion), so values already on the BFP4 grid stay exact. The
weights are DRAM width-sharded over the banks:

- gate/up: each expert's gate/up tile pairs are spread over the banks, rotating by expert so the banks fill evenly.
  A table gives each expert's offset in each bank. An expert with more tiles in a bank than dest holds is stored in
  column groups (`gu_groups`).
- down: H's column tiles are split over the banks and, in a bank, over its cores.

`kblock=True` stores each block transposed in its tile, so the shared exponents run along K (a GPTQ-along-K export);
the matmuls then take their weight tiles transposed. `cache=` keeps the host result in a directory per layer, with a
fingerprint of the weights and sha256 sums. `prepack` writes such a cache without a device, `pack_cached` uploads one
without the weights, and `repack_down` moves a cache to another split of down's columns. A cache that does not match
the call raises `PackCacheMismatch` and is left unchanged.

## Settings

These are module attributes, and the ops' keyword arguments override them.

| Name | Default | What |
|---|---|---|
| `LOADERS` | 2 | decode ops' loaders (column groups and a bias need 2) |
| `ROWS_LOADERS` | 1 | the R-row ops' loaders |
| `SPLIT_SHARE` | 0.5 | the reader's share of the chunks with two loaders |
| `GU_CPB`, `GU_ROWS_CPB` | 0 | gate_up's cores per bank (0: the pack's `cores_per_bank`, default 3, capped at the grid's share) |
| `ROWS_SFPU` | True | decode gate_up's SFPU stages on the valid row only |
| `L1_MARGIN` | 32 KiB | L1 kept free next to an op's circular buffers |
| `common.device.MAX_BANKS` | None | the most DRAM banks weights are sharded over (None: the device's count) |

## Limits and assumptions

- Experts: k <= 16 per token; E <= 1024 for the R-row ops (the union's scratch); H and I multiples of 32; at most 16
  DRAM banks (a table row).
- Weights: BFP4_b only, as packed by `pack` (uint8 DRAM width-sharded tensors). Activations bf16, fp32 accumulation,
  bf16 outputs.
- Decode ops take one row of x. The ids and weights are `[1, 1, 1, k]` in TILE or ROW_MAJOR layout, and the kernels
  read row 0 of the first page. The R-row ops take up to 32 rows, with ids and weights as TILE tensors. The ids of a
  row must be distinct, and rows past R must be finite.
- `route`: R <= 32, E a multiple of 32, logits float32 TILE; the weights are rounded to bf16 at the end.
- Every op checks its circular buffers against the device's free L1 before it launches.
- Tested on ttsim (tt-metal / ttnn 0.81.1) on the default 11 x 10 worker grid and on a reduced worker grid, both
  with 8 DRAM banks.

## Tests

```
python -s moe/tests/test_moe.py            # or: pytest -s moe/tests
```

What is checked and with which bars is in the test's docstring. Against a float64 reference: at HiFi4, a relative
error below 1 % and a PCC above 0.9999; at HiFi2, the ops' default fidelity, below 1.6 %, HiFi2's measured error (up to
1.40 %) with a margin. HiFi2 makes two of the matrix unit's four multiply passes over the operands' mantissas, so its
products leave out low-order bits that HiFi4 keeps: a property of the fidelity, the same on any device. Pass
`fidelity="hifi4"` for the tighter result. Configurations that must agree are checked bit for bit.

## Left out

- One read per chunk of full K rows (an alternative decode loader): it reads the same bytes in fewer requests and
  was dropped to keep one code path.
- The R-row ops' table-row prefetch, the partial coefficient tile and SFPU stages on the used rows only. These were
  optional variants with the same results.
- Timing probes and benchmarks.
