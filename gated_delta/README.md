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
- The compute kernels also call a few tt-llk internals, validated on v0.81.x and listed in `AGENTS.md`.
- `gdn_scan.L1_BUDGET` caps the circular buffers a core may use; set it to your device's worker L1.

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

Each check prints one line per comparison and a final verdict line, and exits 1 on a miss.

## Examples

Each example checks itself against float64 with the bars of "Tests" above and exits 1 on a miss:

```
python examples/gated_deltanet.py    # a 64-token prompt through the scan, then decode steps
python examples/verify.py            # 4 rows in one verify call, the state slots, and a 2-row rollback through gdn_commit
python examples/conv.py              # an 8-token conv chunk, then one-token steps
python examples/mamba2_mixer.py      # a 16-token prompt and 2 decode steps through the Mamba-2 mixer
```

## Limits

- Tiles are 32 x 32. Head dimensions and channel counts are multiples of 32, and value heads are a multiple of key heads.
- `gated_delta_verify`: at most 32 rows a call. `gated_conv_step`: at most 29 tokens a call.
- `prefill_conv`: kernel size 4, no bias.
- The KDA and Mamba-2 modes keep the state in float32. The gated delta rule takes a bf16 or float32 state.

## Origins and license

See `ORIGINS.md` for each file's origin. Files adapted from tenstorrent/tt-metal keep its Apache-2.0 notice. The
rest is new code under the same license.
