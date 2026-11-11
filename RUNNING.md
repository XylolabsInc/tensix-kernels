# Running the checks

How to install what the packages need and run every package's correctness test and example, without a Tenstorrent
card, on ttsim. Every step below was run from scratch, in this order, on a Linux x86_64 host (glibc 2.39) with
nothing of Tenstorrent's installed beforehand, and passed. Nothing here was tested on hardware.

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
`conftest.py` opens one device for the session) printed `31 passed` in the same environment.

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

## Results

With the steps above (ttsim v1.16.0, its default 11 x 10 worker grid and 8 DRAM banks), `scripts/run_checks.sh`
printed, apart from each check's own lines:

```
### dense_reader/tests/test_dense_reader.py
9 passed, 0 failed
### lm_head_topk/tests/test_lm_head.py
5 passed, 0 failed
### moe/tests/test_moe.py
7 passed, 0 failed
### int4/tests/test_int4.py
4 passed, 0 failed
### ternary_matmul/tests/test_ternary.py
6 passed, 0 failed
### examples/dense_reader_example.py
dense_reader example: y = x @ w for one token, relative error 0.169 % against torch
### examples/lm_head_topk_example.py
lm_head_topk example: tokens [3204, 2419], torch's bf16 head [3204, 2419]
### examples/moe_example.py
moe example: experts [27, 17, 30, 3], relative error 0.249 % against the float64 reference
### examples/int4_example.py
int4 example: y = x @ (q * s).T for one token, relative error 0.231 % against torch
### examples/ternary_matmul_example.py
ternary_matmul example: one token, relative error 0.230 % against torch
run_checks.sh: all passed
```

The same `scripts/run_checks.sh` also passed, with the same output, on ttsim v1.15.1 (Python 3.12.11, ttnn 0.81.1,
torch 2.15.0's CPU build, SFPI 7.89.0), on the default grid and on a reduced worker grid. The bars behind each test's
lines are in its docstring. The examples' bars: a relative error below 1 % (0.5 % for
the ternary matmul), the tokens equal, the experts equal to torch's top 4.

## On a card

Not tested on hardware. With a Tenstorrent card the simulator steps (the ttsim library, the soc descriptor,
`TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE`) do not apply; nothing here was checked on a card.
