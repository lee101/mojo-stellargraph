"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOSTELLARGRAPH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-stellargraph.so"
)

I = ctypes.c_int64
U = ctypes.c_uint64
F = ctypes.c_double

# Kept in step with `src/capi.mojo` by hand: every export is a flat list of
# `Int` addresses and sizes, so the table is just the argument counts.
_SIGNATURES: dict[str, tuple[list, object]] = {
    # stellargraph/core/utils.py
    "msg_normalize_adj": ([I, I, I, I, I], None),
    "msg_normalized_laplacian": ([I, I, I, I, I], None),
    "msg_rescale_laplacian": ([I, I, I, F], None),
    "msg_power_iteration": ([I, I, I, I, I, F], F),
    "msg_chebyshev_polynomial": ([I, I, I, I, I], None),
    "msg_PPNP_Aadj_feats_op": ([I, I, I, I, I, I, F], I),
    "msg_GCN_Aadj_feats_op": ([I, I, I, I, I, I, I, I, I], I),
    "msg_invert": ([I, I, I, I], I),
    # stellargraph/mapper/node_mappers.py
    "msg_self_loops": ([I, I, I], None),
    # stellargraph/layer/{gcn,preprocessing_layer,graph_attention}.py
    "msg_GraphConvolution_call": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, F, I],
        None,
    ),
    "msg_GraphPreProcessingLayer_call": ([I, I, I, I, I], None),
    "msg_GraphAttention_call": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, F, I, I, F, F],
        None,
    ),
    "msg_GraphAttentionSparse_call": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, F, I],
        None,
    ),
    # stellargraph/layer/graphsage.py
    "msg_MeanAggregator_group_aggregate": ([I, I, I, I, I, I, I, I, I, I], None),
    "msg_MaxPoolingAggregator_group_aggregate": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, I],
        None,
    ),
    "msg_MeanPoolingAggregator_group_aggregate": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, I],
        None,
    ),
    "msg_AttentionalAggregator_group_aggregate": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, I],
        None,
    ),
    "msg_AttentionalAggregator_call": ([I, I, I, I, I, I, I, I, F], None),
    "msg_GraphSAGEAggregator_call": (
        [I, I, I, I, I, I, I, I, F],
        None,
    ),
    "msg_GraphSAGE_normalization": ([I, I, I, I, I], None),
    # stellargraph/layer/hinsage.py
    "msg_MeanHinAggregator_call": (
        [I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, F],
        None,
    ),
    # stellargraph/layer/{ppnp,appnp}.py
    "msg_PPNPPropagationLayer_call": ([I, I, I, I, I, I, I, I, I], None),
    "msg_APPNPPropagationLayer_call": ([I, I, I, I, I, I, I, I, I, F, I], None),
    "msg_APPNP_propagate": ([I, I, I, I, I, I, F, I], None),
    # stellargraph/layer/link_inference.py
    "msg_LeakyClippedLinear_call": ([I, I, I, F, F, F], None),
    "msg_link_inference_edge_function": (
        [I, I, I, I, I, I, I, I, I, I, I, F, I, I, F, F],
        I,
    ),
    # stellargraph/data/explorer.py
    "msg_naive_weighted_choices": ([I, I, I, I, I, U], I),
    "msg_uniform_random_walk": ([I, I, I, I, I, I, I, I, I, I], None),
    "msg_biased_random_walk": ([I, I, I, I, I, I, I, I, I, I, I, F, F, I], None),
    # sparse helpers behind the two upstream TensorFlow sparse ops
    "msg_coo_to_csr": ([I, I, I, I, I, I, I, I, I, I], I),
    "msg_sparse_dense_matmul": ([I, I, I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOSTELLARGRAPH_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOSTELLARGRAPH_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-stellargraph.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    return np.ascontiguousarray(a, dtype=np.float64)


def i32(a) -> np.ndarray:
    """A C-contiguous int32 view of `a`; node indices are narrow upstream too."""
    return np.ascontiguousarray(a, dtype=np.int32)


def node_indices(a, n: int) -> np.ndarray:
    """C-contiguous int32 node indices, every one of them a node of an `n`-node
    graph. The kernels index straight into the result with no bound of their
    own, so an out-of-range value is an out-of-bounds read, not a wrong number.
    """
    out = i32(np.asarray(a).reshape(-1))
    if out.size and (int(out.min()) < 0 or int(out.max()) >= n):
        raise IndexError(
            "node index out of range for a graph of {} nodes".format(n)
        )
    return out


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojostellargraph._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
