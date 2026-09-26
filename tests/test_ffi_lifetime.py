"""Every buffer handed to the C ABI has to outlive the call.

`addr(x)` returns a plain `int`, so `addr(f64(y))` drops the last reference to
`y` before the FFI call runs and the kernel reads freed memory. The symptom is
order-dependent: a test passes alone and fails once another test has churned
the allocator. Nothing catches that at runtime, so this test reads the source
and rejects the pattern outright.

Binding the temporary to a local first -- `w = f64(x)` then `addr(w)` -- is
the fix, and that is what the code does everywhere.
"""

from __future__ import annotations

import os
import re

import pytest

PACKAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "python",
    "mojostellargraph",
)

# `addr(` applied to the result of a call: the temporary dies at the comma.
TEMPORARY = re.compile(r"addr\(\s*(?:f64|i32|np\.[a-z_]+)\(")


def _sources():
    for name in sorted(os.listdir(PACKAGE)):
        if name.endswith(".py") and name != "_lib.py":
            with open(os.path.join(PACKAGE, name)) as fh:
                yield name, fh.read().split("\n")


def test_no_temporary_is_handed_to_the_ffi():
    offenders = []
    for name, lines in _sources():
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue
            if TEMPORARY.search(line):
                offenders.append("{}:{}: {}".format(name, i, line.strip()))
    assert not offenders, "a temporary is passed to addr():\n" + "\n".join(offenders)


def test_every_ffi_call_site_binds_its_buffers():
    """A call into `lib().msg_*` has to reach `addr` through a name, so a
    reader can see the buffer is still referenced."""
    for name, lines in _sources():
        inside = False
        depth = 0
        for i, line in enumerate(lines, 1):
            if not inside and "lib().msg_" in line:
                inside, depth = True, line.count("(") - line.count(")")
                continue
            if inside:
                depth += line.count("(") - line.count(")")
                if "addr(" in line and TEMPORARY.search(line):
                    raise AssertionError(
                        "{}:{}: temporary inside an FFI call".format(name, i)
                    )
                if depth <= 0:
                    inside = False


@pytest.mark.parametrize("name", [n for n, _ in _sources()])
def test_module_imports_cleanly(name):
    import importlib

    importlib.import_module("mojostellargraph.{}".format(name[:-3]))
