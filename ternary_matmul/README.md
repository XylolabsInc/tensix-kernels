# ternary_matmul: a ternary-weight matmul with group scales along K

Matmuls with ternary weights (trits -1, 0, +1) and one bf16 scale per output channel and G inputs (G a multiple of 32),
on Tenstorrent Tensix devices through `ttnn.generic_op`: one token (`t2_linear`), many rows (`t2m_linear`), and
`linear` choosing by rows. bf16 activations, fp32 accumulation, bf16 output. An optional block Hadamard rotation of the
input (for checkpoints whose ternary weights live in a rotated basis) is fused into the one-token kernel or runs as its
own op.

## Weight formats

- `bfp2`: BFP2_b tiles, 320 bytes a tile, 2.5 bits per weight read;
- `bfp2c`: the same tiles with only their 256 code bytes in DRAM (2 bits per weight);
- `bfp4`: BFP4_b, 576 bytes a tile, 4.5 bits per weight.

ttnn has no BFP2 data type, so the weight is a `uint8` ROW_MAJOR tensor DRAM width-sharded over the banks.

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

The rotation is x' = H_1024 (s * x) per block of 1024 inputs (the normalized Sylvester Hadamard, +-1 signs s per
input), unnormalized on the device: the scales carry the factor 1/32 (`group_scales(..., factor=HADAMARD_SCALE)`).
Module attributes: `L1_BUDGET` (None: the device's free L1 span less `L1_MARGIN`), `DEPTH`, `CHUNK_TILES`, `XDMA`,
`RING_FIXED`, `MT_MAX`.

## Limits

- K a multiple of G and of 32; with the Hadamard, K a multiple of 1024 and at most one block per core for `hadamard`.
- The weight is DRAM width-sharded over banks 0 .. banks - 1 (`max_banks` caps the count).
- `topk` keeps at most 64 candidates a core and one row (one token).
- Written for tt-metal v0.81 (`common/kernels/dense_ops.h`); the BFP2_b unpack uses a tt-llk internal (listed in
  `AGENTS.md`) that can change between tt-metal versions.

## Test

```
python -s ternary_matmul/tests/test_ternary.py
```

The test's docstring lists what is checked and its bars. The kernels run at HiFi2 by default; pass
`fidelity="hifi4"` for the tighter result.
