# Origins

Where every file of this repository comes from, and how that was checked.

## Classes

- **ours**: written for this repository.
- **ours, follows tt-metal**: written for this repository; its structure follows a named file of tt-metal
  (https://github.com/tenstorrent/tt-metal, Apache-2.0), so it is treated as a modified version of that file: it
  carries that file's Tenstorrent copyright notice above ours, and [NOTICE](NOTICE) names the file. No tt-metal text
  is copied beyond calls of its public APIs.

No file is copied from tt-metal or from any other code base. Code that needed anything beyond stock tt-metal
v0.81.1 was left out.

## How it was checked

1. **Line comparison.** Every source line of 30 or more characters (whitespace-normalized) of each `.py`, `.cpp`, `.h`
   and `.sh` file was looked up in tt-metal v0.81.1: its source checkout at tag `v0.81.1` and the runtime tree of the
   `ttnn` 0.81.1 wheel (kernel APIs, LLK and the ops' kernels), 11,894 source files in all. The column "lines in
   tt-metal" counts the lines found there. They are calls of tt-metal's APIs (for example
   `pack_reconfig_data_format(cb_out);`, `source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,`), include lines and
   loop headers, except in `common/kernels/dense_ops.h`, noted below.
2. **Structure.** The files marked "follows tt-metal" were compared with the tt-metal file named: the structure is the
   same idea, the code is not copied. Every kernel was also compared with each of tt-metal's 2,690 op and model
   kernels as a sequence of code tokens (comments dropped; Python's difflib ratio). Every kernel here scored at most
   0.49 against any of them, except `moe/kernels/moe_dn_writer.cpp` (176 tokens), 0.60 against an unrelated
   layer-norm writer; the tt-metal kernels closest in subject (the gpt_oss fused-decode demo's streamed linear,
   routed-expert and LM-rescore kernels) scored no higher than unrelated collective-communication readers, that is,
   at the level of the shared dataflow and compute API boilerplate.
3. **Build.** Every kernel is compiled by tt-metal's JIT from stock tt-metal v0.81.1 headers when the tests run (on
   ttsim), so none uses a header, function or constant that stock tt-metal lacks.

## Files

| File | Class | Lines checked | Lines in tt-metal | Notes |
|---|---|---|---|---|
| `conftest.py` | ours | 4 | 2 |  |
| `common/__init__.py` | ours | 1 | 0 |  |
| `common/device.py` | ours, follows tt-metal | 68 | 7 | `read_vcs` follows the read-VC assignment of tt-metal's `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`; the rest is ours |
| `common/kernels/dense_ops.h` | ours, follows tt-metal | 103 | 32 | the SFPU functions take the form of tt-metal's `calculate_silu` (`ckernel_sfpu_silu.h`), and the "rows" helpers (`silu_rows`, `mul_rows`, ...) that of the helpers of the same names in tt-metal's `models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_compute.cpp`; its shared lines are that form (loop header, `dst_reg` access, the bf16 convert) and include lines |
| `common/kernels/split.h` | ours | 19 | 0 | the split loaders' chunk pattern |
| `common/source.py` | ours | 17 | 0 | kernel source loader (header inlining) |
| `common/testing.py` | ours | 20 | 1 |  |
| `dense_reader/__init__.py` | ours | 0 | 0 | package marker (license header only) |
| `dense_reader/kernels/dense_compute.cpp` | ours, follows tt-metal | 188 | 5 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `dense_reader/kernels/dense_reader.cpp` | ours | 118 | 2 |  |
| `dense_reader/kernels/dense_wload.h` | ours | 53 | 1 |  |
| `dense_reader/kernels/dense_writer.cpp` | ours | 157 | 2 |  |
| `dense_reader/tests/test_dense_reader.py` | ours | 125 | 3 |  |
| `dense_reader/linear.py` | ours | 487 | 7 |  |
| `lm_head_topk/__init__.py` | ours | 0 | 0 | package marker (license header only) |
| `lm_head_topk/head_pick.py` | ours | 139 | 7 |  |
| `lm_head_topk/kernels/head_pick_compute.cpp` | ours | 26 | 2 |  |
| `lm_head_topk/kernels/head_pick_reader.cpp` | ours | 71 | 2 |  |
| `lm_head_topk/kernels/head_pick_records.cpp` | ours | 70 | 3 |  |
| `lm_head_topk/kernels/head_pick_writer.cpp` | ours | 43 | 1 |  |
| `lm_head_topk/lmhead.py` | ours | 123 | 1 |  |
| `lm_head_topk/tests/test_lm_head.py` | ours | 71 | 2 |  |
| `moe/__init__.py` | ours | 0 | 0 | package marker (license header only) |
| `moe/kernels/moe8_rows.h` | ours | 29 | 1 |  |
| `moe/kernels/moe_dn2_reader.cpp` | ours | 51 | 2 |  |
| `moe/kernels/moe_dn2_writer.cpp` | ours | 50 | 4 |  |
| `moe/kernels/moe_dn_compute.cpp` | ours, follows tt-metal | 28 | 2 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `moe/kernels/moe_dn_reader.cpp` | ours | 46 | 2 |  |
| `moe/kernels/moe_dn_writer.cpp` | ours | 13 | 2 |  |
| `moe/kernels/moe_dnr2_writer.cpp` | ours | 56 | 3 |  |
| `moe/kernels/moe_dnr_reader.cpp` | ours | 56 | 2 |  |
| `moe/kernels/moe_gu2_reader.cpp` | ours | 99 | 6 |  |
| `moe/kernels/moe_gu2_writer.cpp` | ours | 75 | 3 |  |
| `moe/kernels/moe_gu_compute.cpp` | ours, follows tt-metal | 70 | 1 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `moe/kernels/moe_gu_reader.cpp` | ours | 68 | 5 |  |
| `moe/kernels/moe_gu_writer.cpp` | ours | 33 | 2 |  |
| `moe/kernels/moe_gur2_writer.cpp` | ours | 82 | 3 |  |
| `moe/kernels/moe_gur_reader.cpp` | ours | 125 | 7 |  |
| `moe/kernels/moe_gur_writer.cpp` | ours | 41 | 2 |  |
| `moe/kernels/moe_route.cpp` | ours | 225 | 12 |  |
| `moe/kernels/moe_sigmoid_q30.h` | ours | 4 | 0 | a fixed-point sigmoid table; `moe8.sigmoid_table()` regenerates it exactly (the test checks it) |
| `moe/moe8.py` | ours | 664 | 12 |  |
| `moe/tests/test_moe.py` | ours | 190 | 5 |  |
| `int4/__init__.py` | ours | 0 | 0 | package marker (license header only) |
| `int4/kernels/w4a16_compute.cpp` | ours, follows tt-metal | 66 | 3 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp` |
| `int4/kernels/w4a16_reader.cpp` | ours | 84 | 1 |  |
| `int4/kernels/w4a16_wload.h` | ours | 24 | 1 |  |
| `int4/kernels/w4a16_writer.cpp` | ours | 42 | 1 |  |
| `int4/kernels/w4dq_compute.cpp` | ours | 17 | 0 |  |
| `int4/kernels/w4dq_reader.cpp` | ours | 31 | 1 |  |
| `int4/kernels/w4dq_writer.cpp` | ours | 19 | 1 |  |
| `int4/tests/test_int4.py` | ours | 51 | 2 |  |
| `int4/w4a16.py` | ours | 149 | 12 |  |
| `ternary_matmul/__init__.py` | ours | 0 | 0 | package marker (license header only) |
| `ternary_matmul/kernels/had_compute.cpp` | ours | 13 | 1 |  |
| `ternary_matmul/kernels/had_reader.cpp` | ours | 40 | 2 |  |
| `ternary_matmul/kernels/had_writer.cpp` | ours | 24 | 2 |  |
| `ternary_matmul/kernels/t2_compute.cpp` | ours, follows tt-metal | 100 | 3 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp`; uses tt-llk's internal srcA unpacker reconfiguration for BFP2_b |
| `ternary_matmul/kernels/t2_reader.cpp` | ours | 139 | 4 |  |
| `ternary_matmul/kernels/t2_wload.h` | ours | 35 | 1 |  |
| `ternary_matmul/kernels/t2_writer.cpp` | ours | 72 | 2 |  |
| `ternary_matmul/kernels/t2m_compute.cpp` | ours, follows tt-metal | 74 | 2 | the matmul loop follows tt-metal's `bmm_large_block_zm_fused_bias_activation.cpp`; uses tt-llk's internal srcA unpacker reconfiguration for BFP2_b |
| `ternary_matmul/kernels/t2m_reader.cpp` | ours | 49 | 2 |  |
| `ternary_matmul/kernels/t2m_writer.cpp` | ours | 39 | 1 |  |
| `ternary_matmul/t2.py` | ours | 345 | 16 |  |
| `ternary_matmul/tests/test_ternary.py` | ours | 111 | 5 |  |
| `examples/dense_reader_example.py` | ours | 15 | 2 | a runnable example |
| `examples/int4_example.py` | ours | 16 | 2 | a runnable example |
| `examples/lm_head_topk_example.py` | ours | 14 | 2 | a runnable example |
| `examples/moe_example.py` | ours | 27 | 2 | a runnable example |
| `examples/ternary_matmul_example.py` | ours | 17 | 2 | a runnable example |
| `scripts/run_checks.sh` | ours | 17 | 0 | every test and example (RUNNING.md) |
| `scripts/setup_ttsim.sh` | ours | 33 | 0 | the install for running on ttsim (RUNNING.md) |

Documentation and other files: `README.md` (each), `RUNNING.md`, `NOTICE`, `.gitignore` and this file are ours;
`LICENSE` is the text of the Apache License, Version 2.0.
