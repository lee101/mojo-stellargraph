"""Shared fixtures.

The graph builders here mirror `tools/dump_upstream.py` exactly: same seed,
same density, same "no isolated node" fixup. That is what lets the parity tests
compare the port against the arrays `tools/dump_upstream.py` dumped from the
real upstream package, which is built from the same graphs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_refs")

# (name, n, density) -- must match `CASES` in tools/dump_upstream.py
CASES = {"sym9": (9, 0.30), "sym24": (24, 0.18), "sym40": (40, 0.10)}

SEED = 20260926


def make_adj(n: int, p: float, seed: int = SEED) -> np.ndarray:
    """The undirected adjacency both sides of every parity assertion use."""
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p).astype(np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    for i in range(n):
        if not a[i].any():
            j = (i + 1) % n
            a[i, j] = a[j, i] = 1.0
    return a


def make_features(n: int, d: int = 5, seed: int = 11) -> np.ndarray:
    return np.ascontiguousarray(
        np.random.default_rng(seed).normal(size=(n, d)), dtype=np.float64
    )


class Graph:
    """The minimum `FullBatchNodeGenerator` reads: `node_list`, `features`,
    `edges`. Upstream takes a `StellarGraph` and calls
    `to_adjacency_matrix()` / `node_features()` on it."""

    def __init__(self, adj: np.ndarray, features: np.ndarray, weights=None):
        self._adj = adj
        self.features = features
        self.node_list = np.arange(adj.shape[0])
        ii, jj = np.nonzero(adj)
        self.edges = np.stack([ii, jj], axis=1).astype(np.int64)
        self.edge_weights = weights

    @property
    def node_count(self) -> int:
        return len(self.node_list)

    def __len__(self) -> int:
        return len(self.node_list)


def load_ref(group: str):
    path = os.path.join(REFS, group + ".npz")
    if not os.path.exists(path):
        pytest.skip(
            "no upstream reference for {!r}; run `pixi run dump-upstream` "
            "(needs the Python 3.8 + stellargraph environment)".format(group)
        )
    return np.load(path)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def sym_graph(rng):
    """A small undirected adjacency, the shape `FullBatchNodeGenerator`
    produces from a `StellarGraph`."""
    return make_adj(*CASES["sym24"])


@pytest.fixture(scope="session")
def adj_with_loops(sym_graph):
    """The same graph with self loops, which GCN and GAT both require."""
    return sym_graph + np.eye(sym_graph.shape[0])


@pytest.fixture(scope="session")
def features(rng):
    """Node features whose width (7) deliberately differs from the fixture
    width used in `make_features` (5), so a kernel built for one cannot
    silently be used with the other."""
    n = CASES["sym24"][0]
    return np.ascontiguousarray(rng.normal(size=(n, 7)), dtype=np.float64)