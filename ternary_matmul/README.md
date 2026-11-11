# ternary_matmul: a ternary-weight matmul with group scales along K

Matmuls with ternary weights (trits -1, 0, +1) and one bf16 scale per output channel and G inputs (G a multiple of 32),
on Tenstorrent Tensix devices through `ttnn.generic_op`: one token (`t2_linear`), many rows (`t2m_linear`), and
`linear` choosing by rows. bf16 activations, fp32 accumulation, bf16 output. An optional block Hadamard rotation of the
input (for checkpoints whose ternary weights live in a rotated basis) is fused into the one-token kernel or runs as its
own op.

## How it works

- **Weight formats.** The trits as hardware block-float tiles whose shared exponents read every value exactly:
  - `bfp2`: BFP2_b, 320 bytes a tile (64 exponent bytes of 127, then 2-bit codes: bit 1 the sign, bit 0 the
    magnitude), 2.5 bits per weight read;
  - `bfp2c`: the same tiles with only their 256 code bytes in DRAM (2 bits per weight); the kernel keeps the exponent
    bytes in L1 and copies the codes behind them;
  - `bfp4`: BFP4_b, 576 bytes a tile, 4.5 bits per weight.

  ttnn has no BFP2 data type, so the weight is a `uint8` ROW_MAJOR tensor DRAM width-sharded over the banks, each
  core's tiles contiguous in its bank; the weight circular buffers are declared `bfloat4_b` with 320-byte pages and the
  compute kernel switches srcA's unpacker to BFP2_b.
- **One token.** As in `int4/`: the token is spread over the rows of the activation tile, so the partial sums of the
  32 groups of a pass land in 32 rows of D; D times the scale tensor's own tile, then its column sum, once per 32
  groups. Weight chunks are full K rows of a core's columns; the writer RISC loads every other chunk.
- **Many rows.** Each weight tile is dequantized in L1 by its group's scale row (`mul_tiles_bcast_rows` with a row
  index; exact, since the trits are -1, 0, +1 and the scale is bf16), then an ordinary matmul with fp32 accumulation;
  row slices of up to `MT_MAX` tile rows, each reading the weights once.
- **Hadamard.** x' = H_1024 (s * x) per block of 1024 inputs (the normalized Sylvester Hadamard, +-1 signs s per input
  width), computed as H_32 X H_32 on 32 x 32 tiles (two tile matmuls a block), unnormalized: the scales carry the
  factor 1/32 (`group_scales(..., factor=HADAMARD_SCALE)`). Fused (`hadamard=` on `t2_linear`), as its own op
  (`hadamard`), or for many rows in ttnn ops (`rotate_rows`).

## API (`ternary_matmul/t2.py`)

| Function | What |
|---|---|
| `ternary_weight(device, trits, fmt="bfp2", cores_per_bank=3, sbw=None, banks=None, max_banks=None)` | trits `[N_out, K]` as the kernel's weight (a dict; `"t"` is the device tensor) |
| `group_scales(device, scales, factor=1.0)` | scales `[N_out, K/G]` as `[K/G, N]` bf16 TILE in DRAM |
| `hadamard_signs(device, signs)` | the rotation's +-1 signs `[K]` as the kernels' sign masks |
| `t2_linear(x, w, scale, g=128, hadamard=None, memory_config=None, fidelity="hifi2", kw=None, split=True, topk=None, out=None, col0=0, gap=None)` | one token; `topk`: the writers keep each core's top k (the dense reader's record format, `dense_reader.linear.topk_result` / `topk_records`); `out`, `col0`, `gap`: write into a wider output at a column offset, optionally in two runs |
| `t2m_linear(x, w, scale, g=128, ...)` | many rows (x already in the weights' basis) |
| `linear(x, w, scale=None, g=None, activation=None, rows_kernel=True, **opts)` | any rank; one row runs `t2_linear`, more rows `rotate_rows` and `t2m_linear` |
| `hadamard(x, mask)`, `rotate_rows(x, mask)` | the rotation alone (one token on the device; many rows in ttnn ops) |
| `reference(x, trits, scales, g)`, `rotate(x, signs)`, `dequantize(trits, scales, g)` | host references |

Module attributes: `L1_BUDGET` (None: the device's free L1 span less `L1_MARGIN`), `DEPTH`, `CHUNK_TILES` (the default
chunk size), `XDMA` and `RING_FIXED` (how x's rows reach the one-token kernel), `MT_MAX`.

## Limits and assumptions

- K a multiple of G and of 32; with the Hadamard, K a multiple of 1024 and at most one block per core for `hadamard`.
- The weight is DRAM width-sharded over banks 0 .. banks - 1 (`max_banks` caps the count).
- `topk` keeps at most 64 candidates a core and one row (one token).
- The one-token kernel's scale stage and fused rotation multiply fp32 sums at the call's `fidelity` (HiFi2 by default),
  which drops low bits of those sums; `fidelity="hifi4"` gives the tighter result.
- The compute kernels switch srcA's unpacker to BFP2_b with tt-llk's internal
  `_llk_unpack_reconfig_data_format_srca_impl_` (the compute API has no BFP2_b circular-buffer format): an internal
  API that can change between tt-metal versions. Written for tt-metal v0.81 (`common/kernels/dense_ops.h`).

## Test

```
python -s ternary_matmul/tests/test_ternary.py
```

Random trits and fp16 group scales, x with outlier channels, against x @ (q * bf16(s)) in float32 (x rotated in float64
with the Hadamard), PCC above 0.9999 and a relative error below 0.5 % for every kernel at HiFi4, the rotation included;
at HiFi2, the default fidelity, below 1.5 % for the one-token kernel (measured 0.78 to 1.26 %, the most with the fused
rotation) and 0.8 % for the multi-row kernel (measured 0.69 %), HiFi2's measured accuracy with a margin (HiFi2 makes two
of the matrix unit's four multiply passes, so its products leave out low-order bits that HiFi4 keeps); bit for bit
across the three weight formats, one loader against two, the fused Hadamard against the Hadamard op (both at HiFi4; the
fused rotation runs at the call's fidelity, the op at HiFi4), the offset output against the plain one, and the top-k
records against the op's own output.

## Left out

- A LoFi variant of the main matmul: whether bf16 activations multiply exactly at LoFi depends on the device, so it is
  not offered here (it also needed tt-llk's internal matmul calls).
- An LM head built on the top-k records with a host rescore: `lm_head_topk/` covers that pattern for block-float heads.
- Loaders for particular checkpoint formats and the timing probes.
