# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Every file git tracks is listed in ORIGINS.md exactly once (LICENSE, NOTICE, ORIGINS.md and DEVELOPMENT.md
are not listed), and README.md links ORIGINS.md and DEVELOPMENT.md on its first screen."""

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOT_LISTED = {"LICENSE", "NOTICE", "ORIGINS.md", "DEVELOPMENT.md"}


def tracked():
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.splitlines() if Path(p).name not in NOT_LISTED]


def listed():
    """The paths in the first column of the "Files" section's table."""
    text = (ROOT / "ORIGINS.md").read_text()
    section = re.search(r"^## Files\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE).group(1)
    return re.findall(r"^\| `([^`]+)` \|", section, re.MULTILINE)


def test_every_tracked_file_is_listed_once():
    rows = Counter(listed())
    files = tracked()
    assert [p for p in files if p not in rows] == []
    assert sorted(p for p in rows if p not in files) == []
    assert sorted(p for p, n in rows.items() if n > 1) == []


def test_readme_first_screen_links_both_documents():
    first = "\n".join((ROOT / "README.md").read_text().splitlines()[:40])
    assert "(ORIGINS.md)" in first and "(DEVELOPMENT.md)" in first
