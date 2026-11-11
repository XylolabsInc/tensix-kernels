# lm_head_topk: an LM head that returns token ids, not logits

`lmhead.py` runs an LM head on a Tenstorrent Tensix device without writing the logits. The head `[vocab, dim]` is cut
into vocabulary blocks, each a call of the dense reader (`dense_reader/linear.py`) with block-float weights, and each
core's writer keeps a running top-k of every row over its columns (the dense reader's top-k records). What comes out
of the matmuls is a small tensor of candidates, not `rows x vocab` logits. The candidates are then rescored exactly
from the bf16 head rows and the best is picked, on the host or, with `head_pick.py`, on the device.

## How it works

1. **Top-k blocks.** For each of `chunks` vocabulary blocks, `matmul(x, block, None, topk=...)`: every core of
   the block's plan writes one record, its best `k` columns for each of the first `rows` rows. The records of all blocks
   go into one result tensor (`slots` records a block: the device's worker count).
2. **Rescore.** Of all records, each row's best `top` candidates by the device's (block-float) logit are rescored with
   the bf16 head rows: bf16 x times bf16 rows, summed in float32, rounded to bf16 (or kept in float32 with
   `logits="float32"`).
3. **Pick.** The largest rescored logit, the lowest token id on ties, a NaN counting as the largest (`pick_token`). The
   token equals a bf16 host head's argmax whenever the host's top-1 is among the device's candidates.

`head_pick.pick` does steps 2 and 3 on the device, one core a row: the reader scans the records for the row's best
`top`, gathers their bf16 rows from a row-major head table in DRAM (one page a vocabulary row, so a tied embedding
table can serve), the compute tilizes them and takes the dot products with fp32 accumulation, and the writer picks.
Only the token ids come back. `head_pick.pick_records` is the pick without a rescore, for a draft head: each row's
best record by the device's own logit, mapped through a token map (a sub-head of selected tokens), on one core. Both
read nothing back and keep their buffers from call to call, so a trace can capture the whole head.

## API

| Function | What |
|---|---|
| `DeviceHead(device, weight, fmt="bfp4", chunks=16, k=8, rescore=True, top=16, blocks=None, logits="bf16", device_pick=False, rows=None, valid=None, token_map=None, l1x=True, **opts)` | the head; `weight` `[vocab, dim]` torch |
| `head(x, rows=1)` | token ids `[rows]` for the first `rows` rows of x `[1, 1, >= rows, dim]` bf16 TILE |
| `head.logits_topk(x, rows, ids_out=None)`, `head.collect(x, rows, xh=None)` | the device part (traceable) and the host part apart |
| `head.candidates(x, rows)` | the device's candidates: values and global ids |
| `pick_token(exact, ids, vocab)` | the pick rule on the host |
| `head_pick.head_rows(device, weight)` | the bf16 head as a row-major DRAM table |
| `head_pick.pick(records, x, rows_t, nrows, k, top=16, logits="bf16", out=None, valid=None, ids_out=None)` | rescore and pick on the device |
| `head_pick.token_map(device, ids)`, `head_pick.pick_records(records, nrows, k, vocab, tmap=None, out=None, ids_out=None)` | the records-only pick and its token map |
| `head_pick.tokens(out, nrows)` | the host's read of a pick's output: tokens, the device's own best, valid counts |

`fmt`: the blocks' weight format, "bfp4", "bfp8" or "bf16". `blocks`: device weights already made (one per block,
DRAM width-sharded, in vocabulary order). `valid`: a padded head's real vocabulary (pad rows are never candidates).
`ids_out`: a one-element uint32 DRAM tensor that also receives row 0's token (an embedding's index for a draft's next
step).

## Limits and assumptions

- Up to 16 rows a call (the dense reader's top-k keeps at most 16 rows, and `rows * k <= 64`); x up to 32 rows.
- `vocab / chunks` a multiple of 32, and a block's columns per core must fit the core's L1 with the dense reader's
  accumulator: raise `chunks` for a large vocabulary.
- The device pick rescores at most 32 candidates a row (`top <= 32`), one core a row (`rows` at most the worker count).
- The pick equals a host rescore's whenever both sum the same products to the same float32; the device's sum order
  differs from torch's, so a near tie in the last bits of the float32 sums can part them.
- Tested on ttsim (tt-metal / ttnn 0.81.1) on the default 11 x 10 worker grid and on a reduced worker grid, both
  with 8 DRAM banks.

## Test

```
python -s lm_head_topk/tests/test_lm_head.py          # or: pytest -s lm_head_topk/tests
```

All checks are exact (token for token, value for value), on a random bf16 head `[8192, 1024]` on a `bfloat4_b` device
head: the records against the blocks' own logits; the host rescore's tokens against a bf16 host head; the device pick
against the host rescore in bf16 and float32 logits, also on a head whose rows repeat in pairs (exact ties: the lowest
id must win) and on a padded head; the records-only pick with and without a token map against the host's argmax over
the same records.

## Left out

- An int8 weight format for the blocks (the dense reader's int8 path is not part of this release).
- Host-side conveniences tied to one deployment: one joint readback of the records and x, paging in a memory-mapped
  host head, a resident host copy.
