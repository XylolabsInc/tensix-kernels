# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Kernel sources for ttnn.generic_op: a kernel file with its local headers inlined.

Kernels are passed to ttnn as source text (KernelDescriptor.SourceType.SOURCE_CODE), which has no include path of its
own, so every `#include "name"` whose file is in one of the given directories is replaced by that file's text (each
header once per kernel, its own includes inlined first). System and tt-metal includes (`<...>`, "api/...") stay.
"""

import re
from pathlib import Path

COMMON = Path(__file__).resolve().parent / "kernels"
_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"([^"/]+)"[ \t]*$', re.M)
_CACHE = {}


def _inline(text, dirs, seen):
    def sub(m):
        name = m.group(1)
        for d in dirs:
            p = d / name
            if p.is_file():
                if name in seen:
                    return ""
                seen.add(name)
                return _inline(p.read_text(), dirs, seen)
        return m.group(0)  # not ours: left to the compiler

    return _INCLUDE.sub(sub, text)


def load(kernels_dir, name, defines=()):
    """The source of kernels_dir/name with its local headers (kernels_dir, then common/kernels) inlined, after a
    `#define` line for each entry of defines."""
    key = (str(kernels_dir), name, tuple(defines))
    if key not in _CACHE:
        dirs = [Path(kernels_dir), COMMON]
        text = _inline((Path(kernels_dir) / name).read_text(), dirs, set())
        head = "".join(f"#define {d}\n" for d in defines)
        _CACHE[key] = head + text
    return _CACHE[key]
