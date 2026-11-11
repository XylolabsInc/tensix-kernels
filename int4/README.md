# int4: an int4-weight matmul with group scales along K

A matmul for one token (decode) with symmetric int4 weights and one bf16 scale per output channel and 32 inputs
(G = 32), on Tenstorrent Tensix devices through `ttnn.generic_op`. bf16 activations, fp32 accumulation, bf16 output.
A second op turns the same int4 weight into a `bfloat8_b` weight for an ordinary block-float matmul over many rows.

## How it works

- **Weight format.** Codes q in -7..7 as sign-magnitude nibbles (sign << 3 | magnitude, the first element of a pair in
  the low nibble), 512 bytes per 32 x 32 tile in the tile's face order: exactly the mantissa section of a `bfloat4_b`
  tile. The tensor is `uint8` ROW_MAJOR, DRAM width-sharded over the banks. The reader reads each K row of a core's
  columns in one request into a staging buffer and copies each 512-byte tile behind 64 exponent bytes of 129, so the
  unpacker reads every value as the integer q.
- **Group scales without dequantizing.** The scale changes every 32 rows of K, so it cannot be applied after the K sum.
  The token is spread over the rows of the activation tile instead (K tile t in row t % 32), so a pass of 32 K tiles
  accumulates in dest with row r holding K tile 32p + r's partial product. That tile is multiplied by the scale
  tensor's own tile, then summed over its rows (a matmul with a tile of ones in row 0) into the output, with the
  packer's L1 accumulation.
- **Work split.** Each DRAM bank's columns are read by up to `cores_per_bank` cores placed next to the bank's preferred
  worker; the writer RISC loads every other weight chunk (a second request stream on the other NoC).

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

## Limits and assumptions

- One token per call (x has one row); K a multiple of 1024 (whole passes of 32 K tiles); codes in -7..7 (no -8).
- The weight is DRAM width-sharded over banks 0 .. banks - 1 (`max_banks` caps the count); N is padded to a multiple of
  banks x 32.
- tt-metal's compute API of v0.81 or later (`common/kernels/dense_ops.h`).

## Test

```
python -s int4/tests/test_int4.py
```

Random codes and scales, x with outlier channels, against x @ (q * s) in float32: relative error below 1 % and PCC above
0.9999, at HiFi4 and at HiFi2, the default fidelity (measured 0.84 to 0.96 % at HiFi2: HiFi2 makes two of the matrix
unit's four multiply passes, so its products leave out low-order bits that HiFi4 keeps); the dequantized weight within
2 % (its `bfloat8_b` rounding); one loader against two bit for bit; a weight over fewer banks than the device has.

## Left out

- Timing and bandwidth probes and comparisons against other matmuls.
- Loading checkpoints in a particular quantizer's file layout: the API takes plain code and scale tensors.
