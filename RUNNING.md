# Running the checks

How to install what the packages need and run every package's correctness test and example on ttsim, without a
Tenstorrent card. These steps were run from scratch, in this order, on a Linux x86_64 host with nothing of
Tenstorrent's installed beforehand. Nothing here was tested on hardware.

## Versions

| Step | Ran on |
|---|---|
| Python venv | Python 3.12.3 (`python3 -m venv`) |
| `ttnn` | 0.81.1, the PyPI wheel |
| `torch` | 2.15.0, the PyPI wheel |
| Simulator | ttsim v1.16.0 (`libttsim_bh.so` from tenstorrent/ttsim's GitHub release) |
| Soc descriptor | tt-metal v0.81.1's `tt_metal/soc_descriptors/blackhole_140_arch.yaml`, beside the simulator library |
| SFPI | 7.89.0 (`sfpi_7.89.0_x86_64_debian.txz` from tenstorrent/sfpi's GitHub release), its sha256 checked against the one the ttnn wheel names |
| Build | none: the kernels are compiled by tt-metal's JIT at their first call |

## 1. Install

```
scripts/setup_ttsim.sh DIR        # PYTHON=python3.12 and TTSIM_VERSION=v1.16.0 by default
```

`PYTHON` must be a Python 3.12 whose `venv` module works (on Debian and Ubuntu, the `python3.12-venv` package). A
uv-managed CPython 3.12 first on `PATH` fails in `ensurepip`; then give `PYTHON=/usr/bin/python3.12`.

`DIR` gets a venv with `ttnn==0.81.1` and `torch==2.15.0` from PyPI, ttsim and the soc descriptor in `DIR/sim`, SFPI
7.89.0 unpacked into the venv's `ttnn/runtime` (the wheel does not include it, and `python -m ttnn.download_sfpi`
does not work in 0.81.1), and `DIR/env.sh`. The script runs these steps:

```
python3.12 -m venv DIR/venv
DIR/venv/bin/python -m pip install ttnn==0.81.1 torch==2.15.0
curl -LO https://github.com/tenstorrent/ttsim/releases/download/v1.16.0/libttsim_bh.so
curl -L -o soc_descriptor.yaml https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.81.1/tt_metal/soc_descriptors/blackhole_140_arch.yaml
curl -LO https://github.com/tenstorrent/sfpi/releases/download/7.89.0/sfpi_7.89.0_x86_64_debian.txz
tar -xJf sfpi_7.89.0_x86_64_debian.txz -C DIR/venv/lib/python3.12/site-packages/ttnn/runtime
```

`soc_descriptor.yaml` must sit beside the simulator library for tt-metal older than v0.83.

## 2. Environment

```
. DIR/env.sh
```

It puts the venv first on `PATH` and sets `TT_METAL_SIMULATOR` (the ttsim library), `TT_METAL_SLOW_DISPATCH_MODE=1`
(ttsim runs slow dispatch only), `TT_METAL_CACHE` and `TTNN_CONFIG_OVERRIDES` (tt-metal's JIT cache and ttnn's cache
and temporary directories, all inside `DIR`). The checks ran in a shell without `PYTHONPATH`, `PYTHONHOME` or
`LD_LIBRARY_PATH` set: a `PYTHONPATH` or `LD_LIBRARY_PATH` that points at another Python or torch installation makes
`import ttnn` load the wrong torch.

## 3. Checks

```
scripts/run_checks.sh             # or: scripts/run_checks.sh DIR/env.sh
```

It runs each package's test, then each example, one Python process each on device 0, and ends with
`run_checks.sh: all passed` or the list of what failed (exit 1). tt-metal writes its logs under `generated/` in the
current directory; `WORKDIR=...` runs the processes elsewhere. `RUN="..."` puts a command in front of each process (a
job wrapper, for example). Each test also runs on its own (`python -s <package>/tests/test_<name>.py [test names]`)
and under pytest: `python -m pip install pytest`, then `pytest -s -p no:cacheprovider <path to the tree>` (the root
`conftest.py` opens one device for the session) printed `32 passed` in the same environment.

| Check | What it compares |
|---|---|
| `dense_reader/tests/test_dense_reader.py` | the matmul against float64 references (three weight formats, scales, prefill, GLU, residual, fp32 output, fewer banks), and bit for bit across configurations (split loaders, block lists, top-k records) |
| `lm_head_topk/tests/test_lm_head.py` | the top-k records and the host and device picks, token for token |
| `moe/tests/test_moe.py` | route, gate_up and down (one row and R rows) against float64 references at HiFi4 and HiFi2, and bit for bit across loaders, cores per bank, split shares and the pack cache |
| `int4/tests/test_int4.py` | the int4 matmul at HiFi4 and HiFi2 and the dequantize against torch, and one loader against two |
| `ternary_matmul/tests/test_ternary.py` | the ternary kernels and the Hadamard at HiFi4 and HiFi2 against torch, and bit for bit across weight formats, loaders and output layouts |
| `examples/dense_reader_example.py` | one token through a `bfloat8_b` matmul against torch |
| `examples/lm_head_topk_example.py` | token ids from a `bfloat4_b` device head with the device pick against a bf16 head in torch |
| `examples/moe_example.py` | route, gate_up and down for one token of a 32-expert top-4 layer against a float64 reference |
| `examples/int4_example.py` | one token through the int4 matmul against torch |
| `examples/ternary_matmul_example.py` | one token through the ternary matmul against torch |

The run ends with `run_checks.sh: all passed`; each test's docstring gives its bars.

## gated_delta

`gated_delta/` has its own runner, `gated_delta/run_tests.sh`: every check in `gated_delta/tests/`, then its four
examples, each in its own Python process, with each run's output kept in `gated_delta/logs/`. It runs in the same
kind of environment as above; its own run used these versions:

| Step | Ran on |
|---|---|
| Python | 3.12 |
| `ttnn` | 0.81.1, the PyPI wheel, with `torch` from PyPI |
| Simulator | ttsim v1.15.1 and v1.16.0, `soc_descriptor.yaml` beside the library |
| SFPI | 7.89.0, unpacked into `ttnn/runtime` |
| `TT_METAL_MODELS` | a plain tt-metal v0.81.1 checkout, for `check_pconv` and `check_pgates` (skipped without it) |
| Build | none: the kernels are compiled by tt-metal's JIT at their first call |

```
export TT_METAL_SIMULATOR=/path/to/libttsim_bh.so TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MODELS=/path/to/tt-metal       # optional: adds check_pconv and check_pgates
gated_delta/run_tests.sh                       # or gated_delta/run_tests.sh /path/to/python
```

The examples also run on their own (from `gated_delta/`): `python examples/gated_deltanet.py` (a prompt through the
scan, then decode steps), `python examples/verify.py` (4 rows in one verify call, the state slots and a 2-row
rollback through `gdn_commit`), `python examples/conv.py` (an 8-token chunk, then one-token steps) and
`python examples/mamba2_mixer.py` (a 16-token prompt and 2 decode steps). Each checks itself against float64 and
exits 1 on a miss.

The run ends with `run_tests: 14 passed, 0 failed, 0 skipped` (with `TT_METAL_MODELS` set).
