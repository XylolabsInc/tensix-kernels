# lm_head_topk: an LM head that returns token ids, not logits

`lmhead.py` runs an LM head on a Tenstorrent Tensix device without writing the logits. The head `[vocab, dim]` is cut
into vocabulary blocks, each a call of the dense reader (`dense_reader/linear.py`) that keeps a top-k of every row.
The candidates are then rescored exactly from the bf16 head rows and the best is picked, on the host or, with
`head_pick.py`, on the device. The pick is the largest rescored logit, the lowest token id on ties, a NaN counting as
the largest (`pick_token`).

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

## Formats and limits

- Up to 16 rows a call (`rows * k <= 64`); x up to 32 rows.
- `vocab / chunks` a multiple of 32; raise `chunks` for a large vocabulary if a block does not fit.
- The device pick rescores at most 32 candidates a row (`top <= 32`), one core a row (`rows` at most the worker count).
- The pick equals a host rescore's whenever both sum the same products to the same float32; the device's sum order
  differs from torch's, so a near tie in the last bits of the float32 sums can part them.
- Tested on ttsim (tt-metal / ttnn 0.81.1).

## Test

```
python -s lm_head_topk/tests/test_lm_head.py          # or: pytest -s lm_head_topk/tests
```

All checks are exact (token for token, value for value); the test's docstring lists them.
