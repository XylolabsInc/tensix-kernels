# Changelog

## 0.1.2
### Fixed
- `dense_reader.linear` and `ternary_matmul.t2.linear` treat every leading dimension of x as rows.
- `rotate_rows` no longer reuses another sign mask's rotation after a buffer address is reused.
- The top-k candidate writers rank -0 with +0 and every NaN above all values, as lm_head_topk's pick does.
- The dense_reader planner counts the residual and top-k record CBs in its L1 fit.
