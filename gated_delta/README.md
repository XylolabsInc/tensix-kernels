# Gated delta-rule kernels for TT-NN

Device kernels for the linear-attention layers of hybrid language models on Tenstorrent hardware, written against
stock upstream tt-metal / ttnn and launched through `ttnn.generic_op`. They cover:

- the gated delta rule (Gated DeltaNet: Qwen3.5 and Qwen3.6, and the DeltaNet layers of the 35B-A3B MoE models);
- Kimi Delta Attention (KDA, per-channel decay);
- the Mamba-2 selective state-space update (Nemotron-H).

## Contents

| Module | Kernels | What it does |
|---|---|---|
| `gdn_scan.py` | `kernels/gdn_compute.cpp`, `gdn_reader.cpp`, `gdn_writer.cpp` | The chunked prefill: `gated_delta_rule(q, k, v, g, beta, initial_state, ...)` over a whole prompt, with the q/k L2 norm, the q scale and, optionally, the gated RMSNorm of the output in the kernel. `chunk_gated_delta_rule_seq_adapter(...)` has upstream's adapter interface. `torch_recurrent(...)` is the exact recurrence in float64. |
| `gdn_verify.py` | `kernels/gdn_verify_compute.cpp`, `gdn_verify_reader.cpp`, `gdn_verify_writer.cpp` | The recurrent update of 1 to 32 rows a call (the decode step, or the verify step of speculative decoding), the state kept on the core between rows: `gated_delta_verify(x, a, b, state, slots, ...)`. Each row's state can go to a slot, so rejected draft rows can be rolled back. Modes: the gated delta rule (raw gates in, `g` and `beta` computed in the kernel; optional gated RMSNorm); KDA (`kda=...`); the Mamba-2 update (`ssm_d=...`, with `dt = softplus(dt_raw + dt_bias)` and an optional floor); and the causal conv folded in (`conv=...`). |
| `gdn_decode.py` | (the verify kernels) | `gated_delta_decode(...)`: the verify kernel at one row. |
| `gdn_conv.py` | `kernels/gdn_conv_compute.cpp`, `gdn_conv_reader.cpp`, `gdn_conv_writer.cpp` | `gated_conv_step(x, state, taps, bias, ...)`: the causal depthwise conv (4 taps, optional bias, SiLU) of 1 to 29 tokens with its rolling state, in place; optional per-row state slots. |
| `gdn_pconv.py` | `kernels/gdn_pconv_*.cpp`, `kernels/gdn_pgates_*.cpp` | The prompt's conv over the fused q, k and v channels in one launch, with the new conv state whole and split (`prefill_conv`), and the prompt's gates `g` and `beta` from the raw projections (`prefill_gates`). Both match upstream's stock op sequence bit for bit; `stock` and `stock_gates` run that sequence for comparison. |
| `gdn_commit.py` | `kernels/gdn_commit_reader.cpp`, `gdn_commit_writer.cpp` | `commit(pairs)`: many page-for-page copies in one launch, for example every layer's state slot into its state after a verify. |
| `mamba2.py` | (the conv and verify kernels) | `Mamba2Mixer(device, weights)`: a whole Mamba-2 mixer (in_proj, the conv, the selective-state update, the gated RMSNorm in groups, out_proj) for Nemotron-H's shapes, with a per-layer cache of the conv and SSM states. |
| `kernels/gdn_ops.h`, `kernels/gdn_compat.h` | | The tile helpers the compute kernels share, and wrappers over the compute API. |
| `examples/`, `run_tests.sh` | | Four examples, each checked against float64: a prompt and decode steps (`gated_deltanet.py`), the verify step with a rollback (`verify.py`), the conv (`conv.py`) and the Mamba-2 mixer (`mamba2_mixer.py`); every check and example in one run. |

## Requirements

- Upstream tt-metal and ttnn 0.81.1 from PyPI (wheels for Python 3.10 and 3.12, Linux x86_64 with glibc 2.34 or
  later), and PyTorch.
- The kernels are compiled from source by tt-metal's JIT at the first call; nothing is prebuilt.
- The tests run on ttsim, Tenstorrent's simulator, or on a device. `check_pconv.py`, `check_pgates.py` and
  `check.py --native` also run upstream's stock path for comparison. They need a tt-metal checkout's `models/`
  importable: set `TT_METAL_MODELS` (or `TT_METAL_HOME`) to the checkout, or put it on `PYTHONPATH`.
- Besides the documented compute API, the compute kernels call tt-llk internals, validated on v0.81.x and not a
  stable API: `_llk_math_eltwise_unary_sfpu_params_` and `_llk_math_eltwise_binary_sfpu_params_` (the partial-tile
  SFPU helpers in `kernels/gdn_compat.h`), `ckernel::sfpu::calculate_softplus`, `calculate_sigmoid`,
  `calculate_silu`, `calculate_sfpu_binary`, `calculate_sfpu_binary_mul`, `calculate_rsqrt` and `calculate_clamp`,
  `TTI_STALLWAIT`, and `MEM_ZEROS_BASE`.

## Parameters of the hardware

Nothing is fixed to one device:
- The compute grid comes from `device.compute_with_storage_grid_size()`, and work is spread over it.
- Tensors are interleaved over the device's DRAM banks.
- `gdn_scan.L1_BUDGET` caps the circular buffers a core may use; set it to your device's worker L1.
- A program larger than the device's kernel config ring fails at enqueue. With `gdn_scan.KERNEL_CONFIG_RING` set to the
  device's default ring size, `gdn_scan.worker_l1_kw(less)` gives
  `open_device` arguments for a smaller worker L1, which makes the ring larger.

## Tests

Each test compares the device with a reference and exits 1 on a miss. Against float64 the bar is a relative L2 error of
at most 2 % with a PCC of at least 0.9999 (`check_kda.py`: 1 %; `check_mixer.py`: 1.5 % on the output, 1 % on the
state). Wherever two device paths must agree, the bar is bit for bit: R rows in one verify call against R one-row
calls, each state slot against its step, the conv's T tokens against T one-token calls, the prompt conv and gates
against upstream's stock ops, and the commit against its sources. `tests/_verdict.py` applies the shared bars.

| Test | Against |
|---|---|
| `tests/check.py` | the prefill scan against the float64 recurrence |
| `tests/check_rows.py` | R rows in one verify call against R chained one-row calls (bit for bit) and float64 |
| `tests/check_decode.py` | the one-row step against float64 |
| `tests/check_conv.py` | the conv step against float64 and against T one-token calls |
| `tests/check_pconv.py`, `tests/check_pgates.py` | the prompt conv and gates against upstream's stock ops, bit for bit |
| `tests/check_ssm.py` | the Mamba-2 mode against the float64 recurrence |
| `tests/check_kda.py` | the KDA mode against float64 |
| `tests/check_commit.py` | the commit op against its sources, bit for bit |
| `tests/check_mixer.py` | the Mamba-2 mixer against the same mixer in float64 on the device's weights |

### Running them on ttsim

```
pip install ttnn==0.81.1 torch
# ttsim, the Blackhole simulator; soc_descriptor.yaml must sit beside the .so for tt-metal before v0.83
curl -LO https://github.com/tenstorrent/ttsim/releases/download/v1.15.1/libttsim_bh.so
curl -L -o soc_descriptor.yaml https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.81.1/tt_metal/soc_descriptors/blackhole_140_arch.yaml
export TT_METAL_SIMULATOR=$PWD/libttsim_bh.so TT_METAL_SLOW_DISPATCH_MODE=1
# the SFPI compiler, which the ttnn wheel leaves out
curl -LO https://github.com/tenstorrent/sfpi/releases/download/7.89.0/sfpi_7.89.0_x86_64_debian.txz
tar -xJf sfpi_7.89.0_x86_64_debian.txz -C "$(python -c 'import os, ttnn; print(os.path.dirname(ttnn.__file__))')/runtime"
cd gated_delta                   # this package
./run_tests.sh                   # every check, then the examples; TT_METAL_MODELS=/path/to/tt-metal adds the two stock-path checks
```

`run_tests.sh` prints one line per run and a verdict (`run_tests: N passed, 0 failed, ...`), and keeps each run's
output in `logs/`. Single checks run on their own, with their default cases or the cases you name:

```
python tests/check.py            # cases: smoke chunks gqa small qwen qwen20 qwen1k decay
python tests/check_rows.py       # small qwen q9b
python tests/check_decode.py     # small qwen q9b
python tests/check_conv.py       # small, plus --rows T and --slots
python tests/check_ssm.py small nemo
python tests/check_kda.py        # --rows 1,2,4,20,32 --state zero,random
python tests/check_commit.py
python tests/check_mixer.py      # --weights bf16|bfp8|bfp4|mixed
TT_METAL_MODELS=/path/to/tt-metal python tests/check_pconv.py
TT_METAL_MODELS=/path/to/tt-metal python tests/check_pgates.py
```

On the simulator a check takes from under a minute to several minutes. Each prints one line per comparison and a
final verdict line, and exits 1 on a miss.

## Examples

Each example prints its comparisons and exits 1 when one misses the bars of "Tests" above. The outputs shown are from
ttsim.

`examples/gated_deltanet.py` runs a Gated DeltaNet layer's recurrence the way a model does: a 64-token prompt through
`gdn_scan.gated_delta_rule`, then decode steps through `gdn_decode.gated_delta_decode` on the state the prompt left,
each checked against the float64 recurrence:

```
python examples/gated_deltanet.py
prompt of 64: o 0.483 % from the float64 recurrence
decode 0: o 0.351 %
...
state after the last step: 0.234 %
```

`examples/verify.py` runs the verify step of speculative decoding: 4 rows (the last accepted token, then 3 draft
tokens) in one `gdn_verify.gated_delta_verify` call, which leaves the state after each of the first 3 rows in a slot.
Each row's o, each slot and the final state are checked against the float64 recurrence run row by row. Then only 2
rows are accepted: `gdn_commit.commit` copies slot 1 into the state, bit for bit, and decode goes on from there:

```
python examples/verify.py
4 rows in one verify call from a random state, against float64 row by row:
row 0: o 0.192 % (PCC 0.9999993); slot 0: 0.061 % (PCC 0.9999998)
row 1: o 0.201 % (PCC 0.9999995); slot 1: 0.075 % (PCC 0.9999997)
row 2: o 0.178 % (PCC 0.9999996); slot 2: 0.092 % (PCC 0.9999996)
row 3: o 0.277 % (PCC 0.9999989); the final state: 0.126 % (PCC 0.9999994)
2 of 4 rows accepted: slot 1 into the state, bit for bit
the next row from there: o 0.302 % (PCC 0.9999990); the state: 0.104 % (PCC 0.9999995)
verify: 11 comparisons, all within the bars
```

`examples/conv.py` runs the causal conv in front of the recurrence, `gdn_conv.gated_conv_step` (4 taps, a bias,
SiLU): an 8-token chunk from a zero state, which also writes the window it read, then one-token steps on the state the
chunk left. The outputs are checked against a float64 conv over the whole sequence; the window and the state, copies
of the inputs, bit for bit:

```
python examples/conv.py
chunk of 8: out 0.179 % (PCC 0.9999984)
  the window: bit for bit
  the state = the last three inputs: bit for bit
step 0: out 0.191 % (PCC 0.9999980)
step 1: out 0.176 % (PCC 0.9999983)
step 2: out 0.159 % (PCC 0.9999986)
the state after the last step = the last three inputs: bit for bit
conv: 7 comparisons, all within the bars
```

`examples/mamba2_mixer.py` runs `mamba2.Mamba2Mixer` with random bf16 weights at its dimensions (generated as in
`tests/check_mixer.py`): a 16-token prompt from `new_cache()`, then 2 decode steps on that cache. Each output and the
SSM state are checked against the same mixer in float64, on the weights as the device holds them:

```
python examples/mamba2_mixer.py
prompt of 16: out 0.491 % (PCC 0.9999880)
SSM state after the prompt: 0.483 % (PCC 0.9999909)
decode 0: out 0.462 % (PCC 0.9999894)
decode 1: out 0.497 % (PCC 0.9999877)
SSM state after the last step: 0.469 % (PCC 0.9999910)
mixer: 5 comparisons, all within the bars
```

## Limits

- Tiles are 32 x 32. Head dimensions and channel counts are multiples of 32, and value heads are a multiple of key heads.
- `gated_delta_verify`: at most 32 rows a call. `gated_conv_step`: at most 29 tokens a call.
- `prefill_conv`: kernel size 4, no bias.
- The KDA and Mamba-2 modes keep the state in float32. The gated delta rule takes a bf16 or float32 state.
- Accuracy is that of the device's arithmetic: the FPU reads float32 operands at TF32. The tests' bars say how close
  each op comes to float64.

## Origins and license

See `ORIGINS.md` for each file's origin. Files adapted from tenstorrent/tt-metal keep its Apache-2.0 notice. The
rest is new code under the same license.
