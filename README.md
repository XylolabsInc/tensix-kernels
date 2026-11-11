<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg">
  <img src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" height="40">
</picture>

# tensix-kernels

**Device kernels for LLM inference on Tenstorrent Tensix NPUs, launched from Python through ttnn**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![ttnn](https://img.shields.io/badge/ttnn-0.81.1-5B2C83)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.81.1)
[![ttsim](https://img.shields.io/badge/tested%20on-ttsim-5B2C83)](https://github.com/tenstorrent/ttsim)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-models-FFD21E)](https://huggingface.co/Xylolabs)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `kernels` · `llm-inference` · `gated-deltanet` · `mixture-of-experts`

</div>

Device kernels for LLM decode on Tenstorrent Tensix devices. They use stock tt-metal and ttnn: Python launches them
through `ttnn.generic_op`, tt-metal's JIT compiles them at the first call, and nothing in tt-metal is rebuilt or
patched. They target the DRAM-bound parts of decode: each weight is read once, in large contiguous requests, by the
cores next to its DRAM bank, with the output stage fused in. `gated_delta/` adds the recurrent layers of hybrid models
(Gated DeltaNet, Kimi Delta Attention, Mamba-2) for prefill and decode.

| Package | What |
|---|---|
| [`dense_reader/`](dense_reader/README.md) | A weight-streaming matmul for decode: `bfloat8_b`, `bfloat4_b` or `bfloat16` weights DRAM width-sharded, x streamed with the weight chunks, split loaders on both data-movement RISCs, fused output stages (per-column scales, SwiGLU / GeGLU, a residual add, fp32 output), top-k records kept by the writers, and block lists (one call over several separately sharded weights) |
| [`lm_head_topk/`](lm_head_topk/README.md) | An LM head that returns token ids: the dense reader's top-k records over vocabulary blocks, then an exact bf16 rescore and pick on the host or on the device, and a records-only pick for draft heads |
| [`moe/`](moe/README.md) | A mixture-of-experts layer that reads only the chosen experts: routing on the device (softmax or sigmoid, top-k), gate/up and down for one token and for up to 32 rows (each expert of the rows' union read once), `bfloat4_b` expert packing |
| [`int4/`](int4/README.md) | An int4-weight matmul for one token with a scale per 32 inputs, and the dequantize to `bfloat8_b` for many rows |
| [`ternary_matmul/`](ternary_matmul/README.md) | A ternary-weight matmul (BFP2_b tiles) with group scales, an optional fused block Hadamard rotation, for one token and for many rows |
| [`gated_delta/`](gated_delta/README.md) | The linear-attention layers of hybrid models: the gated delta rule (Gated DeltaNet), Kimi Delta Attention and the Mamba-2 state update, as a chunked prefill, a recurrent step of 1 to 32 rows a call with state slots for speculative verify, the causal conv with its rolling state, and a whole Mamba-2 mixer |
| `common/` | Shared pieces: the kernel source loader, DRAM bank and core placement helpers, the compute helpers (`kernels/dense_ops.h`), the split loaders' chunk pattern (`kernels/split.h`), the test runner |
| `examples/` | One small runnable example per package: build inputs, call the public API, compare with torch |
| `scripts/` | `setup_ttsim.sh` (the install for running on ttsim) and `run_checks.sh` (every test and example); see [RUNNING.md](RUNNING.md) |

## Requirements

- Python 3.10 or 3.12 with `torch`, and `pip install ttnn==0.81.1` (ttnn 0.81.1 ships wheels for Linux x86_64 with
  glibc 2.34 or later). The compute kernels use the compute API of tt-metal v0.81 (`compute_kernel_hw_startup`,
  `matmul_block_init`), which older releases name differently.
- A Tenstorrent card, or [ttsim](https://github.com/tenstorrent/ttsim), Tenstorrent's simulator (tested with
  v1.15.1).
- SFPI 7.89.0, the version tt-metal v0.81.1 uses. The wheel does not include it and `python -m ttnn.download_sfpi`
  does not work in 0.81.1, so unpack it into ttnn's `runtime` directory by hand:

```
curl -LO https://github.com/tenstorrent/ttsim/releases/download/v1.15.1/libttsim_bh.so
curl -L -o soc_descriptor.yaml https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.81.1/tt_metal/soc_descriptors/blackhole_140_arch.yaml
export TT_METAL_SIMULATOR=$PWD/libttsim_bh.so TT_METAL_SLOW_DISPATCH_MODE=1
curl -LO https://github.com/tenstorrent/sfpi/releases/download/7.89.0/sfpi_7.89.0_x86_64_debian.txz
tar -xJf sfpi_7.89.0_x86_64_debian.txz -C "$(python -c 'import os, ttnn; print(os.path.dirname(ttnn.__file__))')/runtime"
```

`soc_descriptor.yaml` must sit beside the simulator library for tt-metal older than v0.83.

[RUNNING.md](RUNNING.md) gives this setup step by step (`scripts/setup_ttsim.sh`) and runs every check
(`scripts/run_checks.sh`). Both were tested from scratch with ttsim v1.16.0, Python 3.12.3, ttnn 0.81.1 and torch
2.15.0 from PyPI, setting only `TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE=1` (ttsim runs slow dispatch only)
and cache paths; the checks also passed with ttsim v1.15.1. Nothing was tested on hardware.

**Not documented kernel API.** The kernels also use these tt-metal internals, which may move between releases (all
validated on tt-metal v0.81.x):

- `_llk_unpack_reconfig_data_format_srca_impl_` with `get_local_cb_interface(get_operand_id(cb)).fifo_page_size`,
  `p_dim_stride_target` and `DataFormat::Bfp2_b` (the BFP2_b srcA switch of `ternary_matmul/kernels/t2_compute.cpp`
  and `t2m_compute.cpp`);
- `MEM_ZEROS_BASE` and `MEM_ZEROS_SIZE` (the device memory map's zero area; `lm_head_topk`, `moe` and
  `ternary_matmul` readers and writers);
- the `SFPU_UNARY_CALL` and `SFPU_BINARY_CALL` macros with `ckernel::sfpu::calculate_*` functions
  (`calculate_silu`, `calculate_sfpu_binary_mul`) and `_sfpu_sigmoid_` in the row helpers and SFPU functions of
  `common/kernels/dense_ops.h`.

**Defaults.** The DRAM bank count, the worker grid and the L1 the circular buffers may use come from the device
(`common.device.MAX_BANKS` caps the banks weights are sharded over; the `L1_BUDGET` attributes cap the L1). The other
defaults, such as the dense reader's `CONFIGS` trying 2 cores a bank first or the MoE ops' `LOADERS = 2`, are starting
points chosen from measurements on one device; retune them per device.

## Use

```python
import sys; sys.path.insert(0, "path/to/tensix-kernels")
import torch, ttnn
from dense_reader import linear

device = ttnn.open_device(device_id=0)
w, _ = linear.native_weight(device, torch.randn(4096, 2048) * 0.02, ttnn.bfloat8_b)   # [out, in] -> [1, 1, in, out]
x = ttnn.from_torch(torch.randn(1, 1, 1, 2048, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
y = linear.matmul(x, w, None)                                                     # [1, 1, 1, 4096] bf16
```

Each package's README gives its API, its limits and assumptions, and what it leaves out.

## Tests

Each package has a correctness test against torch (`<package>/tests/`). They run as plain scripts or under pytest
(the root `conftest.py` opens one device for the session):

```
python -s dense_reader/tests/test_dense_reader.py
python -s lm_head_topk/tests/test_lm_head.py
python -s moe/tests/test_moe.py
python -s int4/tests/test_int4.py
python -s ternary_matmul/tests/test_ternary.py
pytest -s .                       # the same, with pytest installed
scripts/run_checks.sh             # every test, then every example (RUNNING.md)
gated_delta/run_tests.sh          # gated_delta's checks and its four examples (gated_delta/README.md)
```

They were run on ttsim with tt-metal / ttnn 0.81.1, on the default 11 x 10 worker grid and on a reduced worker grid,
both with 8 DRAM banks. The bars are in each test's docstring: bit for bit where two configurations of an op must agree
(split loaders, block lists, top-k records against the op's own output, the device pick against the host's), relative
error and PCC against a float64 reference otherwise.

## Related work

- tt-metal's own gpt_oss fused decode (`models/demos/gpt_oss/tt/fused_decode` in v0.81.1) is parallel work with the same
  ideas: a streamed linear, MoE layers that read only the routed experts, the router's top-k in the writer, and a
  streamed LM head with a top-k per writer and a terminal pick.
- tt-metal's DRAM-sharded matmul and its tech report "Saturating DRAM bandwidth"
  (`tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md`) describe the reader next to each DRAM bank
  that the dense reader builds on.

## License

Apache License, Version 2.0 ([LICENSE](LICENSE)). Parts follow the structure of tt-metal files, credited in
[NOTICE](NOTICE); [ORIGINS.md](ORIGINS.md) lists where every file comes from and how that was checked, and
[gated_delta/ORIGINS.md](gated_delta/ORIGINS.md) does the same for `gated_delta/`. This is an independent project, not
affiliated with Tenstorrent.
