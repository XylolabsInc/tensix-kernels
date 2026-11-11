# SPDX-FileCopyrightText: © 2026 Xylolabs
# SPDX-License-Identifier: Apache-2.0

"""The pass or fail verdict of the checks that print their comparisons (check.py, check_rows.py, check_decode.py,
check_conv.py, check_ssm.py). It reads every printed comparison line:

- against a reference (a line with a PCC, and a relative error in % or as a fraction): it passes when the relative
  L2 error is at most REL_BAR and the PCC at least PCC_BAR;
- between two device paths that must agree exactly (R rows in one call against R single-row calls, a slot against
  the state of its step, T tokens in one call against T one-token calls, the conv state's shift): it passes only bit
  for bit;
- any line saying DIFFERENT, or holding a non-finite value, fails.

  watch()                # before the check runs
  sys.exit(finish(name))  # after it: prints "NAME: PASS (n comparisons, 0 outside the bars)" and returns 0, else 1"""

import builtins
import re

REL_BAR = 0.02  # 2 %
PCC_BAR = 0.9999
EXACT = ("vs decode", "vs the shift", "vs T steps", "vs the state after step", "rows in one call against")
_REL = re.compile(r"\brel ([0-9.]+(?:e[+-]?\d+)?)( %)?")
_PCC = re.compile(r"\bPCC (-?[0-9.]+(?:e[+-]?\d+)?|nan)")
_state = {"seen": 0, "bad": 0}


def judge(line):
    """'' when the line passes (or is not a comparison), else why it fails."""
    if "DIFFERENT" in line or " nan" in line or " inf" in line:
        return "FAIL"
    if any(tag in line for tag in EXACT):
        _state["seen"] += 1
        return "" if "bit-identical" in line else "FAIL (must be bit for bit)"
    p = _PCC.search(line)
    if not p:
        return ""
    _state["seen"] += 1
    if p.group(1) == "nan":
        return "FAIL"
    pcc = float(p.group(1))
    m = _REL.search(line)
    rel = (float(m.group(1)) / (100 if m.group(2) else 1)) if m else 0.0
    if pcc >= PCC_BAR and rel <= REL_BAR:
        return ""
    return f"FAIL (bars: rel {100 * REL_BAR:g} %, PCC {PCC_BAR})"


def watch():
    orig = builtins.print

    def printed(*args, **kwargs):
        line = kwargs.get("sep", " ").join(str(a) for a in args)
        why = judge(line)
        if why:
            _state["bad"] += 1
            kwargs.pop("sep", None)
            args = (f"{line}  {why}",)
        orig(*args, **kwargs)

    builtins.print = printed


def finish(name):
    ok = _state["bad"] == 0 and _state["seen"] > 0
    print(f"{name}: {'PASS' if ok else 'FAIL'} ({_state['seen']} comparisons, {_state['bad']} outside the bars)")
    return 0 if ok else 1
