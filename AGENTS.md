# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code and others) working in this repository.
`CLAUDE.md` is a symlink to this file.

## What this is

Device kernels for LLM inference on Tenstorrent Tensix NPUs. They are written against stock tt-metal / ttnn 0.81.1,
launched from Python through `ttnn.generic_op`, and compiled by tt-metal's JIT at their first call; nothing in
tt-metal is rebuilt or patched. The README covers each package; each package's README gives its API, limits and
what it leaves out.

## Layout

| Path | Content |
|---|---|
| `common/` | `source.py` (kernel loader, header inlining), `device.py` (banks, sharding, placement, read VCs, the split pattern), `testing.py`, `kernels/dense_ops.h`, `kernels/split.h` |
| `dense_reader/` | `linear.py`: `matmul`, `linear`, `native_weight`, top-k records, block lists |
| `lm_head_topk/` | `lmhead.DeviceHead`, `head_pick` |
| `moe/` | `moe8`: `pack`, `pack_cached`, `route`, `gate_up`, `down` and the `*_rows` forms |
| `int4/` | `w4a16` |
| `ternary_matmul/` | `t2` |
| `gated_delta/` | the gated delta rule, KDA and Mamba-2: `gdn_scan` (prefill), `gdn_verify` and `gdn_decode`, `gdn_conv`, `gdn_pconv`, `gdn_commit`, `mamba2` |
| `examples/`, `scripts/` | one example per package outside `gated_delta/` (which has its own); `setup_ttsim.sh` and `run_checks.sh` |
| `NOTICE`, `ORIGINS.md`, `gated_delta/ORIGINS.md` | where every file comes from |

Each package has `kernels/`, its tests (`tests/test_*.py`; `gated_delta/tests/check_*.py`) and a README.

## Setup and checks

Without a card, on ttsim, as [RUNNING.md](RUNNING.md) gives step by step:

```
scripts/setup_ttsim.sh DIR        # a Python 3.12 venv with ttnn 0.81.1 and torch, ttsim, the soc descriptor, SFPI 7.89.0
. DIR/env.sh
scripts/run_checks.sh             # 5 tests and 5 examples, ending "run_checks.sh: all passed"
gated_delta/run_tests.sh          # gated_delta's 10 checks and 4 examples, ending "run_tests: 14 passed, 0 failed, 0 skipped"
```

- One package: `python -s <package>/tests/test_<name>.py` (dense_reader 9, lm_head_topk 5, moe 7, int4 4,
  ternary_matmul 6), or `pytest -s -p no:cacheprovider <tree>` for all five (31 passed); one `gated_delta` check:
  `python gated_delta/tests/check_<name>.py`.
- Run from outside the tree: tt-metal writes `generated/` in the working directory. Unset `PYTHONPATH` and
  `LD_LIBRARY_PATH` if they point at another torch.
- On the simulator each check takes from under a minute to several minutes. A change to a kernel or to its Python
  launcher runs that package's checks and example before it is committed, on ttsim's default worker grid and on a
  reduced one. Before a release, run everything on both grids, and `gated_delta/run_tests.sh` with `TT_METAL_MODELS`
  set (a tt-metal checkout whose `models/` imports), so that `check_pconv` and `check_pgates` run instead of being
  skipped.

## Rules for every package

- **Never widen a bar or drop a comparison to make a change pass**, and never model the simulator's rounding in a
  reference. A change that moves a bit-for-bit pair apart is a bug until explained.
- **Stock tt-metal only.** No patched tt-metal, and no API or tensor format that exists only outside upstream. The
  internals used beyond the documented kernel API are listed in the README ("Not documented kernel API") and in
  `gated_delta/README.md` ("Requirements"), validated on tt-metal v0.81.x; re-check them on any version bump, and a
  new one needs its line there.
- **Origins and headers.** Nothing is copied from tt-metal or any other code base. Every source file starts with
  SPDX `Apache-2.0` and Xylolabs' copyright. A file whose structure follows a tt-metal file keeps that file's
  Tenstorrent SPDX lines verbatim above ours, says what was modified and from which tt-metal v0.81.1 path, and is
  named in `NOTICE` and in the `ORIGINS.md` that covers it. `NOTICE` keeps tt-metal's attribution block.
- **Nothing device-specific is fixed.** DRAM bank count, worker grid and L1 come from the device. The other defaults
  (loader counts, cores per bank) are starting points measured on one device, to be retuned per device; never present
  them as tuned for a named board.
- **Correctness only.** No timing claims, and no device or board names, speeds, bandwidth, clocks, power or
  temperatures in code or docs. Tests and docs say what was checked against what.
- **Seed every random tensor.**
- **Public material only.** Use public sources (tt-metal, ttnn, ttsim, Tenstorrent's documentation) and nothing you
  can't publish under Apache-2.0.

## dense_reader, lm_head_topk, moe, int4, ternary_matmul

Bars (do not change without a measured reason):

- References are float64: x rounded to bf16 times the weight's own device values.
- HiFi4 cases: 1 % relative error with PCC > 0.9999 (ternary_matmul 0.5 %; the dense fp32 output 0.1 %).
- HiFi2, the ops' default: its measured error plus a margin, stated as HiFi2's accuracy (moe 1.6 %, ternary_matmul
  1.5 % and 0.8 %, int4 and dense_reader 1 %). HiFi2 does 2 of the matrix unit's 4 multiply passes.
- Bit for bit: split shares and one against two loaders, block lists against one call per block, top-k records
  against the op's own output, the device pick against the host rescore, the pack cache against a fresh pack.

Limits an agent must keep:

- `max_banks` / `common.device.MAX_BANKS` caps the DRAM banks a weight is sharded over, and a weight sharded over
  more is refused. Keep the cap and the refusal.
- Decode only: rows ≤ 32 (top-k ≤ 16 rows, rows × k ≤ 64). Weights are DRAM width-sharded: `bfloat8_b`,
  `bfloat4_b` or `bfloat16` for dense_reader; BFP4 packs for moe; int4 and BFP2 for int4 and ternary_matmul.

## gated_delta

Bars (`gated_delta/tests/_verdict.py`; every check exits 1 on a miss; `run_tests.sh` exits 0 only when all pass):

- Against the float64 reference (`gdn_scan.torch_recurrent` and each test's own): relative L2 error ≤ 2 % and
  PCC ≥ 0.9999 on every compared tensor, outputs and final states. Tighter where a test says so: `check_kda.py` 1 %;
  `check_mixer.py` 1.5 % on the output, 1 % on the state.
- Bit for bit: R rows in one verify call against R chained one-row calls; each state slot against the state after its
  row; T conv tokens in one call against T one-token calls; the conv state's shift; the prompt conv and gates against
  upstream's stock ops; the commit against its sources.
- A comparison line saying `DIFFERENT`, or a NaN or inf anywhere, fails.

State layouts (all TILE_LAYOUT; every state is preallocated and updated in place at the same address, so traces stay
valid):

- Gated delta rule and KDA recurrent state: `[B, HV, K, V]`, key dim K as rows, value dim V as columns;
  o = q·S, S' = exp(g)·S + β·kᵀ(v − k·exp(g)·S). The same order as flash-linear-attention's and Hugging Face
  Qwen3-Next's `recurrent_state`. float32; the gated delta rule also takes bf16; KDA and Mamba-2 are float32 only.
  Prefill (`gdn_scan.gated_delta_rule`) takes `initial_state` in any shape with that tile order (`[B·HV, K, V]`
  works) or `None` for zeros, and returns `final_state` `[B, HV, K, V]` or writes it into `out_state` (which may be
  `initial_state` itself).
- Verify and decode (`gdn_verify.gated_delta_verify`, `gdn_decode.gated_delta_decode` at R = 1): the state is read
  once and the state after the last row is written back into it. `slots` are R − 1 tensors laid out exactly as the
  state; `slots[r]` is the state after row r. Rejecting draft rows means committing the right slot into the state
  (`gdn_commit.commit`). R is 1 to 32 a call.
- Conv state: `[B, 3, C]` bf16, rows 0 to 2 the last three inputs, oldest first. Kernel size 4: tap k goes with state
  row k, tap 3 with the new token. Shifted in place; T is 1 to 29 tokens a call, with an optional window
  `[B, 3 + T, C]` for T > 1.
- Mamba-2 SSM state: `[B, HV, N, P]`, d_state N as rows, head dim P as columns, float32. That is the transpose of
  Hugging Face Mamba2's cache (`[B, H, P, N]`): loading or saving one transposes the last two dims.
- Gates: raw a and b at element columns `a_col + hv` and `b_col + hv` of their tensors (g and β computed in the kernel
  from `raw_gates = (a_neg, dt_bias)`), or g and β directly without `raw_gates`.
- Shapes: head dims and channel counts are multiples of 32, and value heads a multiple of key heads.

## Style

- Follow the surrounding code: Python as the packages write it, C++ kernels in the form of the existing kernels.
- American spelling in code, comments and docs.
- Plain, short docs: say what a thing does, what it was checked against, and how to run it.
- Commits: Conventional Commits with a gitmoji, imperative mood, header under 72 characters, for example
  `fix(moe): 🐛 keep the router's top-k order for ties`. One change per commit, signed.
