#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
#
# setup_ttsim.sh DIR: everything the checks need without a Tenstorrent card, in DIR (made if absent):
#   DIR/venv   a Python venv with ttnn 0.81.1 and torch 2.15.0 from PyPI
#   DIR/sim    ttsim's Blackhole library and tt-metal v0.81.1's soc descriptor beside it
#   SFPI 7.89.0 (tt-metal v0.81.1's SFPI compiler), unpacked into the venv's ttnn/runtime, its archive checked against
#              the sha256 the ttnn wheel names
#   DIR/env.sh the environment to source before running anything: the simulator, slow dispatch, and every cache and
#              temporary directory of tt-metal and ttnn inside DIR
# Settings: PYTHON (python3.12), TTSIM_VERSION (v1.16.0). Downloads come from PyPI and tenstorrent's GitHub only.
set -euo pipefail
DIR=${1:?usage: setup_ttsim.sh DIR}
PYTHON=${PYTHON:-python3.12}
TTSIM_VERSION=${TTSIM_VERSION:-v1.16.0}
SFPI_VERSION=7.89.0
mkdir -p "$DIR/sim" "$DIR/cache" "$DIR/tmp"
DIR=$(cd "$DIR" && pwd)
export PIP_CACHE_DIR="$DIR/pip-cache"

echo "== venv ($("$PYTHON" -V 2>&1)), ttnn 0.81.1 and torch 2.15.0 from PyPI"
[ -x "$DIR/venv/bin/python" ] || "$PYTHON" -m venv "$DIR/venv"
"$DIR/venv/bin/python" -m pip install --quiet ttnn==0.81.1 torch==2.15.0

echo "== ttsim $TTSIM_VERSION and tt-metal v0.81.1's soc descriptor"
cd "$DIR/sim"
curl -fsSL -o libttsim_bh.so "https://github.com/tenstorrent/ttsim/releases/download/$TTSIM_VERSION/libttsim_bh.so"
curl -fsSL -o soc_descriptor.yaml \
	https://raw.githubusercontent.com/tenstorrent/tt-metal/v0.81.1/tt_metal/soc_descriptors/blackhole_140_arch.yaml

echo "== SFPI $SFPI_VERSION into ttnn/runtime"
TTNN=$("$DIR/venv/bin/python" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')
# the wheel names its SFPI release and the archive's sha256 (tt_metal/sfpi-version, shell assignments)
# shellcheck disable=SC1091
. "$TTNN/tt_metal/sfpi-version"
[ "${sfpi_version:-}" = "$SFPI_VERSION" ] || { echo "the wheel names SFPI ${sfpi_version:-?}, not $SFPI_VERSION" >&2; exit 1; }
TXZ="sfpi_${SFPI_VERSION}_x86_64_debian.txz"
curl -fsSLO "https://github.com/tenstorrent/sfpi/releases/download/$SFPI_VERSION/$TXZ"
# shellcheck disable=SC2154
echo "$sfpi_x86_64_debian_txz_hash  $TXZ" | sha256sum -c --quiet
tar -xJf "$TXZ" -C "$TTNN/runtime"

cat >"$DIR/env.sh" <<EOF
# source this before running the checks (written by setup_ttsim.sh)
export PATH="$DIR/venv/bin:\$PATH"
export TT_METAL_SIMULATOR="$DIR/sim/libttsim_bh.so" TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_CACHE="$DIR/cache"
export TTNN_CONFIG_OVERRIDES='{"cache_path": "$DIR/cache/ttnn", "model_cache_path": "$DIR/cache/ttnn/models", "tmp_dir": "$DIR/tmp"}'
EOF
echo "== done: . $DIR/env.sh"
