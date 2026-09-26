import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260926)


@pytest.fixture(scope="session")
def sym_graph(rng):
    """A small undirected weighted-free adjacency with a self loop, the shape
    `FullBatchNodeGenerator` produces from a `networkx` graph."""
    n = 24
    a = (rng.random((n, n)) < 0.18).astype(np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


@pytest.fixture(scope="session")
def adj_with_loops(sym_graph):
    """The same graph with self loops, which GCN and GAT both require."""
    a = sym_graph + np.eye(sym_graph.shape[0])
    return a


@pytest.fixture(scope="session")
def features(rng):
    return np.ascontiguousarray(rng.normal(size=(24, 7)))
