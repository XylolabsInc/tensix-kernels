# moe: a mixture-of-experts layer that reads only the chosen experts

`moe8.py` runs one MoE layer on a Tenstorrent Tensix device through `ttnn.generic_op`, with kernels of its own
(`kernels/`). Only the k experts each token routed to are read from DRAM. The expert ids are read on the device,
never on the host, so every op can be captured in a trace.

## API

| Op | In | Out |
|---|---|---|
| `route(logits, ids, w, k, kind=...)` | the router's float32 logits `[1, 1, R, E]` | the top-k ids `[1, 1, R, k]` (uint16 or uint32) and their weights (bf16), written in place |
| `gate_up(x, P, ids, w, k, act=...)` | one token `x [1, 1, 1, H]` | `h [1, 1, 1, k * I]`: per slot s, `w_s * act(x G_e) * (x U_e)` |
| `down(h, P, ids, k)` | `h` | `y [1, 1, 1, H]`: the sum over the slots of `h_s @ D_e` |
| `gate_up_rows(x, P, ids, w, k, rows)` | up to 32 tokens, each with its own k experts | `h [1, 1, 32, u_max * I]`, by expert of the rows' union |
| `down_rows(h, P, ids, k, rows)` | `h` | `y [1, 1, 32, H]` (rows past `rows` are 0) |

- **route**: the top k by fp32 logit (lower id first among equal keys) with softmax weights, or sigmoid scores with
  an optional score bias, DeepSeek-V3's group-limited routing (`groups=`), a routed scaling factor (`scale=`),
  unnormalized sigmoid weights (`norm=False`), a fixed shared-expert slot (`slot=`) and a shared expert's sigmoid
  gate folded into the router's matmul (`shared=`).
- **gate_up** activations: `"silu"` (SwiGLU), `"gelu_tanh"` (GeGLU) and `"swiglu_oai"` (gpt-oss's clamped SwiGLU,
  `min(g, 7) * sigmoid(1.702 min(g, 7)) * (clamp(u, -7, 7) + 1)`); an optional per-expert gate/up bias.

`pack(device, gate_up, down)` converts one layer's experts at load time (`gate_up [E, H, 2I]` as `[up | gate]`
columns, `down [E, I, H]`) and returns the dict `P` the ops take. The experts are stored as whole BFP4_b tiles
(`bfp4_tiles`, the same bytes as tt-metal's host conversion), so values already on the BFP4 grid stay exact.
`kblock=True` stores each block transposed in its tile (shared exponents along K, for a GPTQ-along-K export).
`cache=` keeps the host result in a directory per layer, with a fingerprint of the weights and sha256 sums;
`prepack` writes such a cache without a device, `pack_cached` uploads one without the weights, and `repack_down`
moves a cache to another split of down's columns. A cache that does not match the call raises `PackCacheMismatch`.

Module attributes, which the ops' keyword arguments override (defaults in the file): `LOADERS`, `ROWS_LOADERS`,
`SPLIT_SHARE`, `GU_CPB`, `GU_ROWS_CPB`, `ROWS_SFPU`, `L1_MARGIN`, and `common.device.MAX_BANKS`.

## Formats and limits

- Experts: k <= 16 per token; E <= 1024 for the R-row ops; H and I multiples of 32; at most 16 DRAM banks.
- Weights: BFP4_b only, as packed by `pack`. Activations bf16, fp32 accumulation, bf16 outputs.
- Decode ops take one row of x. The ids and weights are `[1, 1, 1, k]` in TILE or ROW_MAJOR layout. The R-row ops take
  up to 32 rows, with ids and weights as TILE tensors. The ids of a row must be distinct, and rows past R must be
  finite.
- `route`: R <= 32, E a multiple of 32, logits float32 TILE; the weights are rounded to bf16 at the end.
- Tested on ttsim (tt-metal / ttnn 0.81.1).

## Test

```
python -s moe/tests/test_moe.py            # or: pytest -s moe/tests
```

The test's docstring lists what is checked and its bars. The ops run at HiFi2 by default; pass `fidelity="hifi4"`
for the tighter result.
