#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
#
# run_checks.sh [ENV_SH]: every package's correctness test, then every example, one after the other, each in its own
# Python process on device 0 (a card, or ttsim with setup_ttsim.sh's env.sh, given here or sourced before).
# Settings: PYTHON (python), RUN (a command prefix for each process, e.g. a job wrapper; empty by default), WORKDIR
# (the directory the processes run in, where tt-metal writes its generated/ logs; default the current one).
# Exit 0 when everything passed; the last line lists what failed.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "${1:-}" ]; then
	# shellcheck disable=SC1090
	. "$1"
fi
PYTHON=${PYTHON:-python}
cd "${WORKDIR:-$PWD}" || exit 2
failed=()
for t in \
	dense_reader/tests/test_dense_reader.py \
	lm_head_topk/tests/test_lm_head.py \
	moe/tests/test_moe.py \
	int4/tests/test_int4.py \
	ternary_matmul/tests/test_ternary.py \
	examples/dense_reader_example.py \
	examples/lm_head_topk_example.py \
	examples/moe_example.py \
	examples/int4_example.py \
	examples/ternary_matmul_example.py; do
	echo "### $t"
	# shellcheck disable=SC2086
	if ! ${RUN:-} "$PYTHON" -s "$ROOT/$t"; then
		failed+=("$t")
	fi
done
if [ ${#failed[@]} -eq 0 ]; then
	echo "run_checks.sh: all passed"
else
	echo "run_checks.sh: failed: ${failed[*]}"
	exit 1
fi
