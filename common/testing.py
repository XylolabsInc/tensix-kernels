# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Shared pieces of the packages' tests.

Each test module's test_* functions take the device as their one argument. Under pytest the root conftest.py provides
it (one device for the session); without pytest, `python -s <test file> [test names]` runs them through main() below.
"""

import sys
import traceback

import torch
import ttnn


def check(name, y, r, rel=0.01, pcc=0.9999):
    """y against the reference r: relative Frobenius error below rel and PCC above pcc, every value finite."""
    y, r = y.float(), r.float()
    e = ((y - r).norm() / r.norm()).item()
    p = torch.corrcoef(torch.stack([y.flatten(), r.flatten()]))[0, 1].item()
    ok = bool(torch.isfinite(y).all()) and e < rel and p > pcc
    print(f"{'ok  ' if ok else 'FAIL'} {name}: rel {100 * e:.4f} % pcc {p:.6f}", flush=True)
    assert ok, (name, e, p)


def same(name, a, b):
    """a and b bit for bit."""
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: bit for bit {'equal' if ok else 'DIFFERENT'}", flush=True)
    assert ok, name


def raises(name, exc, fn):
    """fn() raises exc."""
    try:
        fn()
    except exc as e:
        print(f"ok   {name}: {type(e).__name__}", flush=True)
        return
    print(f"FAIL {name}: no {exc.__name__}", flush=True)
    raise AssertionError(name)


def main(globs):
    """Run the test_* functions of a module (all, or those named on the command line) on device 0."""
    tests = [(n, f) for n, f in globs.items() if n.startswith("test_") and callable(f)]
    only = sys.argv[1:]
    run = [(n, f) for n, f in tests if not only or n in only]
    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for n, f in run:
            try:
                f(device)
                print(f"PASS {n}", flush=True)
            except Exception:
                traceback.print_exc()
                failed.append(n)
                print(f"FAIL {n}", flush=True)
    finally:
        ttnn.close_device(device)
    print(f"{len(run) - len(failed)} passed, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0
