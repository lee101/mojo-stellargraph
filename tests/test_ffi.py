"""The ctypes signature table must match `src/capi.mojo` exactly.

Every buffer crosses as an `Int` address and every size as an `Int`, so a
mismatch here is a silent memory bug: the kernel reads a length where a
pointer was expected, or writes past the end of a buffer. This test parses the
Mojo source and compares, so the table cannot drift.
"""

from __future__ import annotations

import ctypes
import os
import re

import pytest

from mojostellargraph import _lib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPI = os.path.join(ROOT, "src", "capi.mojo")

EXPORT = re.compile(
    r'@export\("(\w+)"\)\s*\ndef (\w+)\((.*?)\)\s*abi\("C"\)\s*(?:->\s*(\w+))?:',
    re.S,
)


def _exports():
    with open(CAPI) as fh:
        src = fh.read()
    out = {}
    for name, _fn, args, ret in EXPORT.findall(src):
        args = [a.strip() for a in " ".join(args.split()).split(",") if a.strip()]
        argtypes = [
            ctypes.c_double if a.split(":")[1].strip() == "Float64" else ctypes.c_int64
            for a in args
        ]
        restype = {"": None, "Float64": ctypes.c_double, "Int": ctypes.c_int64}[ret]
        out[name] = (argtypes, restype)
    return out


def test_signature_table_matches_capi():
    have = _lib._SIGNATURES
    assert set(have) == set(_exports()), "the ctypes table and src/capi.mojo differ"
    for name, (argtypes, restype) in _exports().items():
        assert have[name][0] == argtypes, "{}: argtypes".format(name)
        assert have[name][1] == restype, "{}: restype".format(name)


def test_every_export_resolves():
    lib = _lib.lib()
    for name in _exports():
        assert getattr(lib, name) is not None


def test_library_is_built_from_the_shipped_sources():
    # `_lib.build` is what makes `import mojostellargraph` work from a clean
    # checkout, so the artifact it loads has to exist and be no older than the
    # newest `.mojo` file.
    path = _lib.build()
    assert os.path.exists(path)
    newest = max(
        os.path.getmtime(os.path.join(d, f))
        for d, _, fs in os.walk(_lib.SRC)
        for f in fs
        if f.endswith(".mojo")
    )
    assert os.path.getmtime(path) >= newest, "the shared library is stale"


@pytest.mark.parametrize("name", sorted(_exports()))
def test_signature_arity_is_plausible(name):
    argtypes, _ = _exports()[name]
    assert len(argtypes) >= 1
