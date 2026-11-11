# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""pytest: the repository root on sys.path and one device for the session (the tests' `device` argument)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def device():
    import ttnn

    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)
