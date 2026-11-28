# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Argument and cache guards that need no device: the packages are imported against a stand-in ttnn and called with
stand-in tensors, so the checks that run before any device call can be tested on the host (a device, when ttnn has one,
is not opened). Run with `pytest -s -p no:cacheprovider tests/test_host_guards.py`."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
PACKAGES = (
    "common",
    "lm_head_topk",
    "moe",
    "int4",
    "ternary_matmul",
    "gated_delta",
    "gdn_scan",
    "gdn_commit",
)


@pytest.fixture
def load(monkeypatch):
    """load("pkg.module"): the module imported against a stand-in ttnn; the real modules are put back afterwards."""
    before = dict(sys.modules)
    stub = MagicMock(name="ttnn")
    monkeypatch.setitem(sys.modules, "ttnn", stub)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(ROOT / "gated_delta"))
    for name in list(sys.modules):
        if name.split(".")[0] in PACKAGES:
            monkeypatch.delitem(sys.modules, name)

    def get(name):
        mod = importlib.import_module(name)
        mod.ttnn = stub
        return mod

    yield get
    for name in list(sys.modules):
        if name.split(".")[0] in PACKAGES:
            del sys.modules[name]
    sys.modules.update({n: m for n, m in before.items() if n.split(".")[0] in PACKAGES})


def tensor(ttnn, shape, dtype=None, layout=None):
    mc = SimpleNamespace(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
        is_sharded=lambda: False,
    )
    return SimpleNamespace(
        shape=shape,
        dtype=dtype,
        layout=layout,
        memory_config=lambda: mc,
        buffer_address=lambda: 0x1000,
    )


# lm_head_topk.head_pick: the output page holds 32 rows ---------------------------------------------------------------


def pick_args(ttnn, nrows):
    """Stand-ins for a pick of nrows rows (k = 8, dim = 64) on a device with a large grid."""
    device = SimpleNamespace(
        compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=8)
    )
    x = tensor(ttnn, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    x.device = lambda: device
    rows_t = tensor(ttnn, (1, 1, 100, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    records = tensor(
        ttnn, (1, 1, 2, 2 * nrows * 8 * 4 // 4), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    records.device = lambda: device
    return records, x, rows_t


def test_pick_refuses_more_rows_than_the_output_page_holds(load):
    pick = load("lm_head_topk.head_pick")
    records, x, rows_t = pick_args(pick.ttnn, 33)
    with pytest.raises(AssertionError):
        pick.pick(records, x, rows_t, 33, 8)
    with pytest.raises(AssertionError):
        pick.pick_records(records, 33, 8, 100)


@pytest.mark.parametrize(
    "shape, dtype, layout",
    [
        ((1, 1, 16, 32), "uint32", "ROW_MAJOR_LAYOUT"),
        ((1, 1, 32, 32), "bfloat16", "ROW_MAJOR_LAYOUT"),
        ((1, 1, 32, 32), "uint32", "TILE_LAYOUT"),
    ],
)
def test_pick_refuses_a_malformed_out(load, shape, dtype, layout):
    pick = load("lm_head_topk.head_pick")
    ttnn = pick.ttnn
    records, x, rows_t = pick_args(ttnn, 4)
    out = tensor(ttnn, shape, getattr(ttnn, dtype), getattr(ttnn, layout))
    with pytest.raises(AssertionError):
        pick.pick(records, x, rows_t, 4, 8, out=out)
    with pytest.raises(AssertionError):
        pick.pick_records(records, 4, 8, 100, out=out)


def test_pick_takes_the_output_page_of_32_rows(load):
    pick = load("lm_head_topk.head_pick")
    ttnn = pick.ttnn
    records, x, rows_t = pick_args(ttnn, 32)
    out = tensor(ttnn, (1, 1, 32, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    pick.check_out(out)
    assert pick.pick(records, x, rows_t, 32, 8, out=out) is out
