# Changelog

## 0.1.3
### Fixed
- `head_pick.pick` and `pick_records` take at most 32 rows, the rows of the output page, and refuse an `out` that is
  not a `[1, 1, 32, 32]` uint32 row-major tensor interleaved in DRAM.
- `gdn_commit.commit` refuses pairs whose destination is another pair's source or destination, as the copies are
  unordered.
- The pack cache of `moe8` fingerprints every byte of the weights, not every 4093rd value; a cache made before this
  release is taken again with `rebuild=True`.
- `int4_weight`, `ternary_weight` and `moe8.pack` and `pack_cached` check an explicit `banks` against the device's
  banks and against `max_banks` or `MAX_BANKS` (ValueError).

## 0.1.2
### Fixed
- `dense_reader.linear` and `ternary_matmul.t2.linear` treat every leading dimension of x as rows.
- `rotate_rows` no longer reuses another sign mask's rotation after a buffer address is reused.
- The top-k candidate writers rank -0 with +0 and every NaN above all values, as lm_head_topk's pick does.
- The dense_reader planner counts the residual and top-k record CBs in its L1 fit.
