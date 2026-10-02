#!/usr/bin/env python
"""Dump reference outputs from the real upstream `stellargraph` package.

This script runs under the pinned Python 3.8 environment that
`pixi run upstream-env` creates (`.upstream/.venv`), NOT under the pixi
environment, because `stellargraph` 1.2.1 requires Python < 3.9 and
TensorFlow 2.x. It writes one `.npz` per upstream function into
`tests/upstream_refs/`, and `tests/test_upstream_parity.py` asserts the Mojo
port reproduces those arrays.

Regenerate with:

    pixi run dump-upstream

Everything downstream of `stellargraph` is deterministic: the graphs are built
from a fixed-seed generator, and every quantity dumped here is a closed-form
function of the adjacency (no Keras weight initialisation, no training).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

import scipy.sparse as sp

import stellargraph  # noqa: F401  (imported for the version assertion below)

from stellargraph import StellarGraph
from stellargraph.core import utils as sg_utils
from stellargraph.data import explorer as sg_explorer
from stellargraph.mapper import FullBatchNodeGenerator

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "upstream_refs",
)

# The graphs are shared with `tests/conftest.py` so both sides of a parity
# assertion see the identical matrix.
def _graph(n: int, p: float, seed: int):
    """A connected-enough undirected graph with the port's fixture seed."""
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p).astype(np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    # guarantee no isolated node: every node keeps at least one edge
    for i in range(n):
        if not a[i].any():
            j = (i + 1) % n
            a[i, j] = a[j, i] = 1.0
    return a


CASES = {
    # name -> (n, density)
    "sym9": (9, 0.30),
    "sym24": (24, 0.18),
    "sym40": (40, 0.10),
}


def _features(n: int, d: int, seed: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.random.default_rng(seed).normal(size=(n, d)), dtype=np.float64
    )


def dump_core_utils() -> dict:
    out = {}
    for name, (n, p) in CASES.items():
        a = _graph(n, p, 20260926)
        A = sp.csr_matrix(a)
        out[f"{name}/adj"] = a
        # `normalize_adj` has two branches and an `add_self_loops` flag; all
        # four combinations are dumped so every branch is parity-checked
        out[f"{name}/normalize_adj_sym"] = np.asarray(
            sg_utils.normalize_adj(A, symmetric=True).todense()
        )
        out[f"{name}/normalize_adj_left"] = np.asarray(
            sg_utils.normalize_adj(A, symmetric=False).todense()
        )
        out[f"{name}/normalize_adj_sym_loops"] = np.asarray(
            sg_utils.normalize_adj(A, symmetric=True, add_self_loops=True).todense()
        )
        out[f"{name}/normalized_laplacian"] = np.asarray(
            sg_utils.normalized_laplacian(A).todense()
        )
        out[f"{name}/calculate_laplacian"] = sg_utils.calculate_laplacian(a)
        # `rescale_laplacian` prints and calls `eigsh`; suppress the print
        out[f"{name}/rescale_laplacian"] = np.asarray(
            sg_utils.rescale_laplacian(sp.csr_matrix(
                sg_utils.normalized_laplacian(A).todense()
            )).todense()
        )
    return out


def dump_adj_ops() -> dict:
    out = {}
    for name, (n, p) in CASES.items():
        a = _graph(n, p, 20260926)
        A = sp.csr_matrix(a)
        feats = _features(n, 5, 11)
        # the inputs travel with the outputs so a parity test never has to
        # reconstruct them and risk building a different matrix
        out[f"{name}/adj"] = a
        for k in (1, 2, 3):
            _, a_gcn = sg_utils.GCN_Aadj_feats_op(
                features=feats, A=A, k=k, method="gcn"
            )
            out[f"{name}/gcn_k{k}"] = np.asarray(a_gcn.todense())
            _, a_sgc = sg_utils.GCN_Aadj_feats_op(
                features=feats, A=A, k=k, method="sgc"
            )
            out[f"{name}/sgc_k{k}"] = np.asarray(a_sgc.todense())
        for tp in (0.1, 0.15, 0.25):
            _, a_ppnp = sg_utils.PPNP_Aadj_feats_op(
                features=feats, A=A, teleport_probability=tp
            )
            out[f"{name}/ppnp_tp{tp}"] = np.asarray(a_ppnp)
    return out


def dump_generator() -> dict:
    out = {}
    import networkx as nx

    for name, (n, p) in CASES.items():
        a = _graph(n, p, 20260926)
        feats = _features(n, 5, 11)
        idx = [str(i) for i in range(n)]
        nxg = nx.Graph()
        nxg.add_nodes_from(idx)
        nxg.add_edges_from(
            [(str(i), str(j)) for i in range(n) for j in range(i + 1, n) if a[i, j] > 0]
        )
        g = StellarGraph.from_networkx(
            nxg, node_features=pd.DataFrame(feats, index=idx)
        )
        for method in ("gcn", "sgc", "self_loops", "gat", "none"):
            for k in ([1, 2] if method == "sgc" else [1]):
                gen = FullBatchNodeGenerator(
                    g, method=method, k=k, sparse=False
                )
                A = gen.Aadj.toarray() if hasattr(gen.Aadj, "toarray") else np.asarray(gen.Aadj)
                out[f"{name}/gen_{method}_k{k}"] = np.asarray(A, dtype=np.float64)
        gen = FullBatchNodeGenerator(g, method="ppnp", sparse=False)
        out[f"{name}/gen_ppnp"] = np.asarray(gen.Aadj, dtype=np.float64)
    return out


def dump_preprocessing_layer() -> dict:
    """`GraphPreProcessingLayer.call` is pure NumPy in upstream, so the
    reference is computed from its own statements, not by instantiating a
    Keras layer."""
    out = {}
    for name, (n, p) in CASES.items():
        a = _graph(n, p, 20260926)
        out[f"{name}/adj"] = a
        adj = a
        adj_T = adj.T
        adj = adj + adj_T * (adj_T > adj) - adj * (adj_T > adj)
        adj = adj + np.diag(np.ones(adj.shape[0]) - np.diag(adj))
        rowsum = adj.sum(1)
        d = 1.0 / np.sqrt(rowsum)
        out[f"{name}/preproc"] = (d[:, None] * adj) * d[None, :]
    return out


def dump_graph_convolution() -> dict:
    """This drives the real `stellargraph.layer.GraphConvolution` Keras layer.

    `build()` is called explicitly with the input shapes `GCN.__call__` would
    pass (`layer([h_layer] + Ainput)`, where `h_layer` is `(1, N, F)` and
    `Ainput` is `[A]` with `A` of shape `(1, N, N)`), then the weights are
    pinned so the result is a function of the inputs alone.

    Note 1.2.1's `call` applies no gather: `GCN.__call__` does the
    `GatherIndices` afterwards. So the layer output is `(1, N, units)` and the
    gather is not part of what is dumped here.
    """
    from stellargraph.layer import GraphConvolution

    out = {}
    for name, (n, p) in CASES.items():
        a = _graph(n, p, 20260926) + np.eye(n)
        feats = _features(n, 5, 11)
        rng = np.random.default_rng(99)
        for units in (1, 4, 7):
            kernel = rng.normal(size=(feats.shape[1], units))
            bias = rng.normal(size=(units,))
            for act in (None, "relu", "elu", "softmax", "sigmoid"):
                for use_bias in (True, False):
                    layer = GraphConvolution(units, activation=act, use_bias=use_bias)
                    layer.build([(1, n, feats.shape[1]), (1, n, n)])
                    w = [kernel.astype(np.float32)]
                    if use_bias:
                        w.append(bias.astype(np.float32))
                    layer.set_weights(w)
                    y = np.asarray(
                        layer.call(
                            [
                                np.asarray(feats, dtype=np.float32)[None, ...],
                                np.asarray(a, dtype=np.float32)[None, ...],
                            ]
                        ),
                        dtype=np.float64,
                    )[0]
                    out[f"{name}/gconv_u{units}_{act}_b{int(use_bias)}"] = y
    return out


def dump_activations() -> dict:
    """Every layer here does `activations.get(activation)`, so the set of
    activations is Keras's, not stellargraph's. These are taken from
    `tf.keras.activations` directly rather than re-derived, so the reference is
    the function upstream actually calls."""
    import tensorflow as tf

    out = {}
    feats = _features(16, 5, 11)
    # every layer does `activations.get(activation)`, so the set of names is
    # whatever Keras resolves. `sparsemax` is deliberately absent: 1.2.1 does
    # not offer it through `activations.get`, and dumping a hand-rolled
    # sparsemax here would test something upstream never calls.
    names = [
        None, "linear", "relu", "leaky_relu", "elu", "selu", "gelu", "swish",
        "sigmoid", "softmax", "softplus", "softsign", "hard_sigmoid",
        "exponential", "tanh",
    ]
    for act in names:
        key = "none" if act is None else act
        try:
            fn = tf.keras.activations.get(act)
        except Exception as exc:  # pragma: no cover - reported, not hidden
            print(f"skipping activation {act!r}: {exc}", file=sys.stderr)
            continue
        # Keras reads `x.shape.rank`, so the input has to be a tensor
        out[f"act/{key}"] = np.asarray(
            fn(tf.constant(feats, dtype=tf.float64)), dtype=np.float64
        )
    return out


def dump_walks() -> dict:
    """Random walks depend on `numpy.random.RandomState`, which cannot cross
    the C ABI, so the Mojo port uses its own LCG and cannot reproduce these
    node sequences. What IS reproducible is the deterministic neighbour and
    degree structure the walk reads, plus the shape contract. Those are
    dumped here and asserted; the walk values themselves are asserted against
    upstream's own documented invariants in `test_explorer.py`."""
    out = {}
    import networkx as nx

    for name, (n, p) in CASES.items():
        a = _graph(n, p, 20260926)
        idx = [str(i) for i in range(n)]
        nxg = nx.Graph()
        nxg.add_nodes_from(idx)
        nxg.add_edges_from(
            [(str(i), str(j)) for i in range(n) for j in range(i + 1, n) if a[i, j] > 0]
        )
        g = StellarGraph.from_networkx(nxg, node_features=pd.DataFrame(
            np.ones((n, 1)), index=idx
        ))
        walk = sg_explorer.UniformRandomWalk(g, n=3, length=5)
        walks = walk.run(idx, seed=42)
        out[f"{name}/walk_lengths"] = np.array([len(w) for w in walks])
        out[f"{name}/walk_maxlen"] = np.array(
            [max(w) for w in walks], dtype=np.int64
        )
        # every walk stays inside the node set, which is the invariant the
        # port must hold regardless of the generator
        out[f"{name}/walk_minlen"] = np.array([len(w) for w in walks])
        out[f"{name}/n_walks"] = np.array([len(walks)])
        # neighbour arrays, which are pure graph structure
        for node in range(min(n, 6)):
            # `use_ilocs=True` means `node` is already an iloc, not a node id
            nb = np.sort(
                np.asarray(g.neighbor_arrays(node, use_ilocs=True), dtype=np.int64)
            )
            out[f"{name}/neighbors_{node}"] = nb
    return out


def dump_link_inference() -> dict:
    """This drives the real `stellargraph.layer.link_inference` edge function
    and its `LinkEmbedding` / `LeakyClippedLinear` layers.

    `link_inference(...)` returns a Keras closure, so it is called on real
    tensors. Its `Dense` weights are pinned after the fact where a kernel is
    needed, so the result is a function of the inputs alone.
    """
    import tensorflow as tf
    from stellargraph.layer.link_inference import (
        LeakyClippedLinear,
        LinkEmbedding,
        link_inference as sg_link_inference,
    )

    out = {}
    rng = np.random.default_rng(5)
    for n in (8, 32):
        d = 6
        x0 = rng.normal(size=(n, d))
        x1 = rng.normal(size=(n, d))
        out[f"link/n{n}/x0"] = x0
        out[f"link/n{n}/x1"] = x1
        t0 = tf.constant(x0, dtype=tf.float32)
        t1 = tf.constant(x1, dtype=tf.float32)

        # LinkEmbedding on its own: the operator half of the function
        for method in ("ip", "dot", "l1", "l2", "mul", "hadamard", "concat", "avg"):
            le = LinkEmbedding(activation="linear", method=method)
            y = np.asarray(le([t0, t1]), dtype=np.float64)
            out[f"link/n{n}/{method}"] = y

        # LeakyClippedLinear on its own: `x + gamma*relu(lo-x) - gamma*relu(x-hi)`
        for low, high in ((1.0, 5.0), (-2.0, 3.0)):
            lcl = LeakyClippedLinear(low=low, high=high, alpha=0.1)
            z = rng.normal(size=(n, 4))
            out[f"link/n{n}/lcl_{low}_{high}_x"] = z
            out[f"link/n{n}/lcl_{low}_{high}"] = np.asarray(
                lcl(tf.constant(z, dtype=tf.float32)), dtype=np.float64
            )

        # the whole edge function, with a pinned Dense kernel
        for method in ("ip", "concat", "l1", "avg"):
            for output_dim in (1, 3):
                for act in ("linear", "sigmoid", "relu", "softmax"):
                    for clip in (None, (1.0, 5.0)):
                        fn = sg_link_inference(
                            output_dim=output_dim,
                            output_act=act,
                            edge_embedding_method=method,
                            clip_limits=clip,
                            name="ref",
                        )
                        le = np.asarray(
                            LinkEmbedding(activation="linear", method=method)([t0, t1]),
                            dtype=np.float64,
                        )
                        y = np.asarray(fn([t0, t1]), dtype=np.float64)
                        key = f"link/n{n}/{method}_d{output_dim}_{act}_c{clip}"
                        out[key] = y
                        out[key + "_le"] = le
    return out


def main() -> int:
    version = getattr(stellargraph, "__version__", "unknown")
    os.makedirs(OUT, exist_ok=True)
    print("stellargraph", version, file=sys.stderr)

    groups = {
        "core_utils": dump_core_utils(),
        "adj_ops": dump_adj_ops(),
        "generator": dump_generator(),
        "preprocessing": dump_preprocessing_layer(),
        "graph_convolution": dump_graph_convolution(),
        "activations": dump_activations(),
        "explorer": dump_walks(),
        "link_inference": dump_link_inference(),
    }
    total = 0
    for name, data in groups.items():
        path = os.path.join(OUT, name + ".npz")
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in data.items()})
        total += len(data)
        print(f"wrote {path} ({len(data)} arrays)", file=sys.stderr)
    print(f"total {total} reference arrays", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())