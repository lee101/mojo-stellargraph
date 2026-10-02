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

import numpy as np

import pytest

import mojostellargraph as sg

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
            {
                "Float64": ctypes.c_double,
                "UInt64": ctypes.c_uint64,
            }.get(a.split(":")[1].strip(), ctypes.c_int64)
            for a in args
        ]
        restype = {
            "": None,
            "Float64": ctypes.c_double,
            "Int": ctypes.c_int64,
        }[ret]
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


# ------------------------------------------- indices the kernels trust blindly
#
# The kernels index straight into the caller's buffers with whatever they are
# handed, so a bad index is an out-of-bounds access rather than a wrong number.
# These tests pin the checks that stand between the two.


def _gcn(n=8, d=4, units=3):
    rng = np.random.default_rng(7)
    layer = sg.GraphConvolution(units, activation="relu", final_layer=True)
    layer.build(d)
    layer.kernel[:] = rng.normal(size=layer.kernel.shape)
    a = np.eye(n)
    return layer, a, rng.normal(size=(n, d))


def test_graph_convolution_rejects_an_out_of_range_out_index():
    layer, a, x = _gcn()
    with pytest.raises(IndexError):
        layer(x, a, np.array([a.shape[0]]))


def test_graph_convolution_rejects_a_negative_out_index():
    layer, a, x = _gcn()
    with pytest.raises(IndexError):
        layer(x, a, np.array([-1]))


def test_graph_attention_rejects_an_out_of_range_out_index():
    rng = np.random.default_rng(11)
    n, d, heads, units = 8, 4, 2, 3
    a = np.eye(n)
    for layer in (sg.GraphAttention(units, attn_heads=heads, final_layer=True),
                  sg.GraphAttentionSparse(units, attn_heads=heads, final_layer=True)):
        layer.build(d)
        layer.kernels[:] = rng.normal(size=(heads, d, units))
        layer.attn_kernels[:] = rng.normal(size=(heads, 2 * units))
        layer.biases[:] = 0.0
        with pytest.raises(IndexError):
            layer(rng.normal(size=(n, d)), a, np.array([n + 3]))


def test_ppnp_and_appnp_reject_an_out_of_range_out_index():
    rng = np.random.default_rng(13)
    n, d = 8, 4
    a = np.eye(n)
    feats = rng.normal(size=(n, d))
    ppnp = sg.PPNPPropagationLayer(d, final_layer=True)
    appnp = sg.APPNPPropagationLayer(d, final_layer=True)
    with pytest.raises(IndexError):
        ppnp(feats, a, np.array([n]))
    with pytest.raises(IndexError):
        appnp(feats, feats, a, np.array([n]))


def test_sparse_tensor_rejects_an_index_past_the_dense_shape():
    with pytest.raises(IndexError):
        sg.SparseTensor(
            np.array([[0, 5]], dtype=np.int32), np.array([1.0]), (3, 3)
        )
    with pytest.raises(IndexError):
        sg.SparseTensor(
            np.array([[-1, 0]], dtype=np.int32), np.array([1.0]), (3, 3)
        )


def test_csr_from_edges_rejects_an_endpoint_past_the_node_count():
    with pytest.raises(IndexError):
        sg.explorer.csr_from_edges(np.array([[0, 4]]), 3)


def test_a_non_monotonic_indptr_is_a_dead_end_not_an_out_of_bounds_read():
    """A caller may pass its own CSR. `indptr = [0, 1, 0, 1]` gives node 1 a
    derived out-degree of -1, which is not a column range."""
    indptr = np.array([0, 1, 0, 1], dtype=np.int32)
    colind = np.array([1, 0], dtype=np.int32)
    walks = sg.UniformRandomWalk((indptr, colind), 3).run(
        nodes=[1], n=1, length=4, seed=0
    )
    assert walks == [[1]]


def test_naive_weighted_choices_of_an_empty_weight_vector_is_none():
    """Upstream's `probs[-1]` on an empty iterator would raise, so there is
    nothing to choose; the kernel returns `None` before touching `colind`."""
    assert sg.naive_weighted_choices(np.zeros(0), seed=0) is None
