#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0
#
# run_tests.sh [PYTHON]: every check in tests/ with its default cases, then the example, one after another, on the
# device or simulator the environment selects (TT_METAL_SIMULATOR for ttsim). Each run's output goes to
# logs/NAME.log; one line per run and a final verdict go to stdout. Exit 0 when every run passed.
# check_pconv.py and check_pgates.py compare with upstream's stock ops: they need TT_METAL_MODELS (a tt-metal
# checkout whose models/ is importable) and are skipped without it.
set -u
cd "$(dirname "$0")" || exit 1
PY=${1:-python}
mkdir -p logs
pass=0 fail=0 skip=0
run() { # NAME ARGS...
	local name=$1
	shift
	if "$PY" "$@" >"logs/$name.log" 2>&1; then
		echo "pass  $name"
		pass=$((pass + 1))
	else
		echo "FAIL  $name (exit $?; logs/$name.log)"
		fail=$((fail + 1))
	fi
}
run check tests/check.py
run check_rows tests/check_rows.py
run check_decode tests/check_decode.py
run check_conv tests/check_conv.py
run check_ssm tests/check_ssm.py small nemo
run check_kda tests/check_kda.py
run check_commit tests/check_commit.py
run check_mixer tests/check_mixer.py
if [ -n "${TT_METAL_MODELS:-}" ]; then
	run check_pconv tests/check_pconv.py
	run check_pgates tests/check_pgates.py
else
	echo "skip  check_pconv, check_pgates (no TT_METAL_MODELS)"
	skip=2
fi
run example examples/gated_deltanet.py
run example_verify examples/verify.py
run example_conv examples/conv.py
run example_mixer examples/mamba2_mixer.py
echo "run_tests: $pass passed, $fail failed, $skip skipped"
[ "$fail" = 0 ]
