# Origins

Where every file of this repository comes from, and how that was checked.

## Classes

- **ours**: written for this repository.
- **ours, follows tt-metal**: its structure follows a named tt-metal file (https://github.com/tenstorrent/tt-metal, Apache-2.0), so it is treated as a
  modified version of that file: it carries that file's Tenstorrent SPDX lines above ours, and [NOTICE](NOTICE) names
  the file. No text is copied beyond calls of tt-metal's public APIs.
- **adapted**: taken from a public repository under its declared license (the source, license and changes are in the
  "Source" and "Notes" columns and in the last section); it keeps that source's copyright line beside ours.

9 files follow a tt-metal file and are named in [NOTICE](NOTICE); 5 kernel files of `gated_delta/` are
adapted from tt-metal's `chunk_gated_delta_rule` op, and [gated_delta/ORIGINS.md](gated_delta/ORIGINS.md) lists the
changes to them. Code that needed anything beyond stock tt-metal v0.81.1 was left out.

## Files

One row per tracked file, except `LICENSE` (the text of the Apache License, Version 2.0), `NOTICE`, this file and
`DEVELOPMENT.md`; `tests/test_origins.py` checks that each appears exactly once. "Source" names the file a row follows
or is adapted from. "Lines checked" counts the lines of 30 or more characters (whitespace-normalized), every line
counted, duplicates included; "Lines in tt-metal" counts those also found in the tt-metal tree described below.

| File | Class | Source | Lines checked | Lines in tt-metal | Notes |
|---|---|---|---|---|---|
| `.gitignore` | ours |  | 0 | 0 | ignore patterns |
| `AGENTS.md` | ours |  | 94 | 0 | instructions for coding agents |
| `CHANGELOG.md` | ours |  | 4 | 0 | release notes |
| `CLAUDE.md` | ours |  | 0 | 0 | symbolic link to AGENTS.md |
| `README.md` | ours |  | 54 | 1 | documentation |
| `RUNNING.md` | ours |  | 57 | 0 | documentation |
| `common/__init__.py` | ours |  | 2 | 1 |  |
| `common/device.py` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` | 70 | 9 | `read_vcs` follows the read-VC assignment of tt-metal's `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`; the rest is ours |
| `common/kernels/dense_ops.h` | ours, follows tt-metal | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`; `models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_compute.cpp` | 106 | 35 | the SFPU functions take the form of tt-metal's `calculate_silu` (`ckernel_sfpu_silu.h`), and the "rows" helpers (`silu_rows`, `mul_rows`, ...) that of the helpers of the same names in tt-metal's `models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_compute.cpp`; its shared lines are that form (loop header, `dst_reg` access, the bf16 convert) and include lines |
| `common/kernels/split.h` | ours |  | 19 | 1 | the split loaders' chunk pattern |
| `common/source.py` | ours |  | 18 | 1 | kernel source loader (header inlining) |
| `common/testing.py` | ours |  | 21 | 2 |  |
| `conftest.py` | ours |  | 5 | 4 |  |
| `dense_reader/README.md` | ours |  | 28 | 0 | documentation |
| `dense_reader/__init__.py` | ours |  | 1 | 1 | package marker (license header only) |
| `dense_reader/kernels/dense_compute.cpp` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | 190 | 10 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `dense_reader/kernels/dense_reader.cpp` | ours |  | 119 | 5 |  |
| `dense_reader/kernels/dense_wload.h` | ours |  | 54 | 2 |  |
| `dense_reader/kernels/dense_writer.cpp` | ours |  | 153 | 3 |  |
| `dense_reader/linear.py` | ours |  | 494 | 8 |  |
| `dense_reader/tests/test_dense_reader.py` | ours |  | 146 | 6 |  |
| `examples/dense_reader_example.py` | ours |  | 16 | 3 | a runnable example |
| `examples/int4_example.py` | ours |  | 17 | 3 | a runnable example |
| `examples/lm_head_topk_example.py` | ours |  | 15 | 3 | a runnable example |
| `examples/moe_example.py` | ours |  | 28 | 3 | a runnable example |
| `examples/ternary_matmul_example.py` | ours |  | 18 | 3 | a runnable example |
| `gated_delta/ORIGINS.md` | ours |  | — | — | documentation added to the inventory after the original comparison |
| `gated_delta/README.md` | ours |  | 68 | 0 | documentation |
| `gated_delta/examples/conv.py` | ours |  | 50 | 5 | a runnable example |
| `gated_delta/examples/gated_deltanet.py` | ours |  | 47 | 4 | a runnable example |
| `gated_delta/examples/mamba2_mixer.py` | ours |  | 46 | 3 | a runnable example |
| `gated_delta/examples/verify.py` | ours |  | 66 | 4 | a runnable example |
| `gated_delta/gdn_commit.py` | ours |  | 53 | 5 | Commit speculative state with page-for-page copies in one device launch |
| `gated_delta/gdn_conv.py` | ours |  | 93 | 7 | Run causal conv1d with four taps and SiLU for T Gated DeltaNet tokens |
| `gated_delta/gdn_decode.py` | ours |  | 17 | 1 | Run one Gated DeltaNet decode step in a single device launch |
| `gated_delta/gdn_pconv.py` | ours |  | 151 | 19 | `gdn_pconv.stock` calls upstream's model code, which is not included here |
| `gated_delta/gdn_scan.py` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_mono_program_factory.cpp` | 224 | 11 | its program layout follows that file; `chunk_gated_delta_rule_seq_adapter` has the interface of upstream's adapter of the same name |
| `gated_delta/gdn_verify.py` | ours |  | 302 | 9 | Run R Gated DeltaNet decode steps in order in one device launch |
| `gated_delta/kernels/gdn_commit_reader.cpp` | ours |  | 29 | 3 | the commit kernels |
| `gated_delta/kernels/gdn_commit_writer.cpp` | ours |  | 25 | 3 | the commit kernels |
| `gated_delta/kernels/gdn_compat.h` | ours |  | 67 | 15 | one name for each compute-API call the kernels make, and partial-tile SFPU helpers on tt-metal's LLK entry points |
| `gated_delta/kernels/gdn_compute.cpp` | adapted | tt-metal v0.81.1, `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gated_delta_rule.cpp` (the "mono" program), Apache-2.0 | 344 | 35 | several heads per core; the q/k L2 norm and the q scale in the kernel; the WY inverse per 32 x 32 diagonal block; the gated RMSNorm of the output; the state update at float32 |
| `gated_delta/kernels/gdn_conv_compute.cpp` | ours |  | 38 | 4 | computes what upstream's `_causal_conv1d_decode_t1` and `_causal_conv1d_fir` compute (tt-metal v0.81.1, `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`) |
| `gated_delta/kernels/gdn_conv_reader.cpp` | ours |  | 136 | 9 | computes what upstream's `_causal_conv1d_decode_t1` and `_causal_conv1d_fir` compute (tt-metal v0.81.1, `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`) |
| `gated_delta/kernels/gdn_conv_writer.cpp` | ours |  | 43 | 6 | computes what upstream's `_causal_conv1d_decode_t1` and `_causal_conv1d_fir` compute (tt-metal v0.81.1, `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`) |
| `gated_delta/kernels/gdn_ops.h` | adapted | the tile helpers of `gdn_compute.cpp` (above), shared with the verify kernel; tt-metal v0.81.1, Apache-2.0 | 126 | 16 |  |
| `gated_delta/kernels/gdn_pconv_compute.cpp` | ours |  | 59 | 5 | bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order |
| `gated_delta/kernels/gdn_pconv_reader.cpp` | ours |  | 43 | 5 | bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order |
| `gated_delta/kernels/gdn_pconv_writer.cpp` | ours |  | 38 | 9 | bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order |
| `gated_delta/kernels/gdn_pgates_compute.cpp` | ours |  | 36 | 5 | bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order |
| `gated_delta/kernels/gdn_pgates_reader.cpp` | ours |  | 33 | 3 | bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order |
| `gated_delta/kernels/gdn_pgates_writer.cpp` | ours |  | 21 | 5 | bit for bit with upstream's `_causal_conv1d_fir` and the gate ops of its prefill, by using the same SFPU operations in the same order |
| `gated_delta/kernels/gdn_reader.cpp` | adapted | tt-metal v0.81.1, `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/dataflow/reader_chunk_gated_delta_rule.cpp` and `chunk_gdn_gate_read.hpp`, Apache-2.0 | 187 | 27 | flat token-major or head-major inputs with a column offset; several heads per core |
| `gated_delta/kernels/gdn_verify_compute.cpp` | ours |  | 555 | 14 | the verify kernels; the compute kernel uses `gdn_ops.h` |
| `gated_delta/kernels/gdn_verify_reader.cpp` | ours |  | 399 | 19 | the verify kernels |
| `gated_delta/kernels/gdn_verify_writer.cpp` | adapted | tt-metal v0.81.1, `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/dataflow/writer_chunk_gated_delta_rule.cpp`, Apache-2.0 | 157 | 13 | multi-row recurrent states and slots, output row gathering, gated normalization and folded convolution |
| `gated_delta/kernels/gdn_writer.cpp` | adapted | tt-metal v0.81.1, `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/dataflow/writer_chunk_gated_delta_rule.cpp`, Apache-2.0 | 40 | 11 | flat or head-major output; several heads per core |
| `gated_delta/mamba2.py` | ours |  | 86 | 6 | Run a Nemotron-H Mamba-2 mixer in TT-NN |
| `gated_delta/run_tests.sh` | ours |  | 26 | 1 | every check of `gated_delta/tests/`, then the examples |
| `gated_delta/tests/_verdict.py` | ours |  | 25 | 1 | The pass or fail verdict of the checks that print their comparisons (check.py, check_rows.py, check_decode.py, check_conv.py, check_ssm.py) |
| `gated_delta/tests/check.py` | ours |  | 156 | 11 | Compare chunked gated delta-rule prefill with an fp64 recurrence on ttsim or a device |
| `gated_delta/tests/check_commit.py` | ours |  | 31 | 7 | Check batched page copies against their sources, bit for bit, on ttsim or a device |
| `gated_delta/tests/check_conv.py` | ours |  | 87 | 9 | Compare causal conv1d with upstream device ops and an fp64 reference |
| `gated_delta/tests/check_decode.py` | ours |  | 59 | 8 | Compare one Gated DeltaNet decode step with an fp64 recurrence from random state |
| `gated_delta/tests/check_kda.py` | ours |  | 80 | 6 | Compare the verify kernel's KDA mode with an fp64 recurrence using synthetic inputs |
| `gated_delta/tests/check_mixer.py` | ours |  | 103 | 4 | Compare the Mamba-2 mixer with a host fp64 reference using the device's stored weights |
| `gated_delta/tests/check_pconv.py` | ours |  | 65 | 10 | Check prompt convolution and state outputs against upstream ops, bit for bit |
| `gated_delta/tests/check_pgates.py` | ours |  | 39 | 7 | Check prompt gates against stock ttnn ops, bit for bit over logical columns |
| `gated_delta/tests/check_rows.py` | ours |  | 106 | 5 | Compare multi-row verification with chained decode steps and an fp64 recurrence |
| `gated_delta/tests/check_ssm.py` | ours |  | 85 | 5 | Compare Mamba-2 decode and verification with an fp64 SSM recurrence |
| `int4/README.md` | ours |  | 16 | 0 | documentation |
| `int4/__init__.py` | ours |  | 1 | 1 | package marker (license header only) |
| `int4/kernels/w4a16_compute.cpp` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | 68 | 6 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `int4/kernels/w4a16_reader.cpp` | ours |  | 85 | 2 |  |
| `int4/kernels/w4a16_wload.h` | ours |  | 25 | 5 |  |
| `int4/kernels/w4a16_writer.cpp` | ours |  | 43 | 2 |  |
| `int4/kernels/w4dq_compute.cpp` | ours |  | 19 | 1 |  |
| `int4/kernels/w4dq_reader.cpp` | ours |  | 32 | 2 |  |
| `int4/kernels/w4dq_writer.cpp` | ours |  | 20 | 2 |  |
| `int4/tests/test_int4.py` | ours |  | 49 | 4 |  |
| `int4/w4a16.py` | ours |  | 150 | 13 |  |
| `lm_head_topk/README.md` | ours |  | 24 | 0 | documentation |
| `lm_head_topk/__init__.py` | ours |  | 1 | 1 | package marker (license header only) |
| `lm_head_topk/head_pick.py` | ours |  | 142 | 8 |  |
| `lm_head_topk/kernels/head_pick_compute.cpp` | ours |  | 27 | 3 |  |
| `lm_head_topk/kernels/head_pick_reader.cpp` | ours |  | 76 | 4 |  |
| `lm_head_topk/kernels/head_pick_records.cpp` | ours |  | 75 | 4 |  |
| `lm_head_topk/kernels/head_pick_writer.cpp` | ours |  | 44 | 3 |  |
| `lm_head_topk/lmhead.py` | ours |  | 124 | 2 |  |
| `lm_head_topk/tests/test_lm_head.py` | ours |  | 91 | 4 |  |
| `moe/README.md` | ours |  | 31 | 0 | documentation |
| `moe/__init__.py` | ours |  | 1 | 1 | package marker (license header only) |
| `moe/kernels/moe8_rows.h` | ours |  | 30 | 2 |  |
| `moe/kernels/moe_dn2_reader.cpp` | ours |  | 52 | 5 |  |
| `moe/kernels/moe_dn2_writer.cpp` | ours |  | 51 | 6 |  |
| `moe/kernels/moe_dn_compute.cpp` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | 30 | 5 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `moe/kernels/moe_dn_reader.cpp` | ours |  | 47 | 5 |  |
| `moe/kernels/moe_dn_writer.cpp` | ours |  | 14 | 3 |  |
| `moe/kernels/moe_dnr2_writer.cpp` | ours |  | 57 | 5 |  |
| `moe/kernels/moe_dnr_reader.cpp` | ours |  | 57 | 5 |  |
| `moe/kernels/moe_gu2_reader.cpp` | ours |  | 100 | 8 |  |
| `moe/kernels/moe_gu2_writer.cpp` | ours |  | 76 | 5 |  |
| `moe/kernels/moe_gu_compute.cpp` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | 72 | 4 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `moe/kernels/moe_gu_reader.cpp` | ours |  | 69 | 7 |  |
| `moe/kernels/moe_gu_writer.cpp` | ours |  | 34 | 3 |  |
| `moe/kernels/moe_gur2_writer.cpp` | ours |  | 83 | 5 |  |
| `moe/kernels/moe_gur_reader.cpp` | ours |  | 126 | 9 |  |
| `moe/kernels/moe_gur_writer.cpp` | ours |  | 42 | 3 |  |
| `moe/kernels/moe_route.cpp` | ours |  | 231 | 19 |  |
| `moe/kernels/moe_sigmoid_q30.h` | ours |  | 69 | 1 | a fixed-point sigmoid table; `moe8.sigmoid_table()` regenerates it exactly (the test checks it) |
| `moe/moe8.py` | ours |  | 667 | 18 |  |
| `moe/tests/test_moe.py` | ours |  | 189 | 7 |  |
| `scripts/run_checks.sh` | ours |  | 18 | 1 | every test and example (RUNNING.md) |
| `scripts/setup_ttsim.sh` | ours |  | 34 | 1 | the install for running on ttsim (RUNNING.md) |
| `ternary_matmul/README.md` | ours |  | 27 | 0 | documentation |
| `ternary_matmul/__init__.py` | ours |  | 1 | 1 | package marker (license header only) |
| `ternary_matmul/kernels/had_compute.cpp` | ours |  | 14 | 2 |  |
| `ternary_matmul/kernels/had_reader.cpp` | ours |  | 41 | 4 |  |
| `ternary_matmul/kernels/had_writer.cpp` | ours |  | 25 | 3 |  |
| `ternary_matmul/kernels/t2_compute.cpp` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | 102 | 6 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp`; uses tt-llk's internal srcA unpacker reconfiguration for BFP2_b |
| `ternary_matmul/kernels/t2_reader.cpp` | ours |  | 140 | 8 |  |
| `ternary_matmul/kernels/t2_wload.h` | ours |  | 36 | 5 |  |
| `ternary_matmul/kernels/t2_writer.cpp` | ours |  | 76 | 3 |  |
| `ternary_matmul/kernels/t2m_compute.cpp` | ours, follows tt-metal | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | 76 | 6 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp`; uses tt-llk's internal srcA unpacker reconfiguration for BFP2_b |
| `ternary_matmul/kernels/t2m_reader.cpp` | ours |  | 50 | 4 |  |
| `ternary_matmul/kernels/t2m_writer.cpp` | ours |  | 40 | 2 |  |
| `ternary_matmul/t2.py` | ours |  | 347 | 18 |  |
| `ternary_matmul/tests/test_ternary.py` | ours |  | 127 | 8 |  |
| `tests/test_origins.py` | ours |  | 21 | 4 | every tracked file is listed in ORIGINS.md exactly once |

## How it was checked

All counts are from a run on 2026-11-24 over the files of this tree (on `d600eca` with this change), against the version
this repository pins, tt-metal v0.81.1. The files the 0.1.2 fixes and release touched (`CHANGELOG.md`,
`dense_reader/kernels/dense_writer.cpp`, `dense_reader/linear.py`, `dense_reader/tests/test_dense_reader.py`,
`ternary_matmul/kernels/t2_writer.cpp`, `ternary_matmul/t2.py`, `ternary_matmul/tests/test_ternary.py`) were counted
again on 2026-11-28 against the same trees.

1. **Line comparison.** Every line of 30 or more characters (whitespace-normalized), every line counted and duplicates
   included, of each file in the table was looked up in tt-metal v0.81.1 (its source checkout at tag `v0.81.1`,
   24,522 text files, plus the runtime tree of the `ttnn` 0.81.1 wheel, 6,665) and in
   transformers v5.13.0 (5,909), the GPTQ reference implementation
   [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (commit `2d65066`, 41),
   [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) (commit `9f7d370`, 172) and
   [llama.cpp](https://github.com/ggml-org/llama.cpp) (commit `d3c3e24`, 4,152). Of 10,585 lines
   checked, 728 are found in tt-metal's tree and 1 only in one of the other four trees. The lines found
   are license headers, import lines, calls of public APIs and common idioms.
2. **Tokens.** Each of the 116 `.py`, `.cpp`, `.h` and `.sh` files was compared with every Python, C, C++, CUDA and
   shell file of the same trees as a set of distinct 10-token sequences (comments dropped, and docstrings for Python):
   at most 36.84 % of a file's distinct sequences occur in any one upstream file (the highest is `conftest.py`, closest
   to `tt-metal/.github/actions/scripts/test_release_failure_tickets.py`; a file of 5 lines checked, so few sequences); 58 files reach 2.5 % or more. Among
   the files of class "ours" with 50 or more lines checked, the highest is 16.18 % (`gated_delta/kernels/gdn_compat.h`).
3. **Build.** Every kernel is compiled by tt-metal's JIT from stock tt-metal v0.81.1 headers when the tests run (on
   ttsim), so none uses a header, function or constant that stock tt-metal lacks.

## Third-party model code and configs

| File | Upstream | Revision | License |
|---|---|---|---|
| `gated_delta/kernels/gdn_compute.cpp` | [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) | v0.81.1 | Apache-2.0 |
| `gated_delta/kernels/gdn_reader.cpp` | tenstorrent/tt-metal | v0.81.1 | Apache-2.0 |
| `gated_delta/kernels/gdn_writer.cpp` | tenstorrent/tt-metal | v0.81.1 | Apache-2.0 |
| `gated_delta/kernels/gdn_ops.h` | tenstorrent/tt-metal | v0.81.1 | Apache-2.0 |

tt-metal's `chunk_gated_delta_rule` op is itself derived from flash-linear-attention's `naive_chunk_gated_delta_rule`
(https://github.com/fla-org/flash-linear-attention, MIT); [NOTICE](NOTICE) reproduces that license. No model code or
model configuration is included.
