"""Port of `stellargraph/data/explorer.py` (v0.8.1): `UniformRandomWalk`,
`BiasedRandomWalk` and `naive_weighted_choices`.

Upstream walks a `networkx` graph. Here the graph is a CSR adjacency
(`indptr`, `colind`), which is the same neighbour relation in the layout the
C ABI can carry. The walk logic, the neighbour shuffling, the dead-end `break`
and the `1/p` / `1.0` / `1/q` transition weights are upstream's; the generator
is a fixed LCG, since `numpy.random.RandomState` cannot cross the boundary.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i32, lib


def csr_from_edges(edges, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR `(indptr, colind)` for an `[e, 2]` list of `(u, v)` edges, sorted by
    row. This is the same neighbour order `self.neighbors(node)` gives
    upstream, where `node.successors()`/`predecessors()` are sorted."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    counts = np.bincount(e[:, 0], minlength=n) if e.size else np.zeros(n, np.int64)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)
    return indptr, i32(e[:, 1])


def _as_csr(graph, n):
    if isinstance(graph, tuple) and len(graph) == 2:
        return graph
    return csr_from_edges(graph, n)


class GraphWalk:
    """Base for the two walkers upstream shares, with the same `run` signature.

    Args:
        graph: an `[e, 2]` array of edges, or a `(indptr, colind)` CSR pair.
        n (int): the number of nodes.
    """

    def __init__(self, graph, n: int):
        self.n = int(n)
        self.indptr, self.colind = _as_csr(graph, self.n)
        self.degrees = np.diff(self.indptr.astype(np.int64))

    def _roots(self, nodes):
        if nodes is None:
            return np.arange(self.n, dtype=np.int32)
        return i32(np.asarray(nodes).reshape(-1))

    def run(self, nodes=None, n: int | None = None, length: int | None = None,
            seed=None):
        raise NotImplementedError

    @staticmethod
    def _unpad(walks, lens):
        """The `[roots * n, length]` padding comes off, so this is the list of
        lists of node ids upstream returns."""
        return [[int(x) for x in w[:l]] for w, l in zip(walks, lens)]


class UniformRandomWalk(GraphWalk):
    """
    Performs uniform random walks on the given graph.

    ```
    run(nodes=None, n=1, length=5, seed=None)
    ```
    """

    def run(self, nodes=None, n: int = 1, length: int = 5, seed: int = 0):
        """
        Returns:
            A list of lists of node ids, one list per walk.
        """
        roots = self._roots(nodes)
        n = int(n)
        length = int(length)
        walks_out = np.zeros((roots.size * n, length), dtype=np.int32)
        lens_out = np.zeros(roots.size * n, dtype=np.int32)
        work = np.zeros(max(int(self.degrees.max()) if self.degrees.size else 1, 1),
                        dtype=np.int32)
        lib().msg_uniform_random_walk(
            addr(self.indptr), addr(self.colind), addr(roots), addr(walks_out),
            addr(lens_out), addr(work), roots.size, n, length, int(seed or 0),
        )
        return self._unpad(walks_out, lens_out)


class BiasedRandomWalk(GraphWalk):
    """
    Performs biased second order random walks (like those used in the Node2Vec
    algorithm https://snap.stanford.edu/node2vec/) controlled by p and q.

    ```
    run(nodes=None, n=1, p=1.0, q=1.0, length=5, seed=None, weighted=False)
    ```
    """

    def run(self, nodes=None, n: int = 1, p: float = 1.0, q: float = 1.0,
            length: int = 5, seed: int = 0, weighted: bool = False,
            edge_weight_label: str = "weight"):
        """
        `weighted=True` is not implemented: upstream reads the per-edge label
        off a `networkx` graph, which this port has no graph object for. See the
        README.
        """
        if weighted:
            raise NotImplementedError(
                "weighted BiasedRandomWalk is not ported; it reads edge labels "
                "off a networkx graph"
            )
        if p <= 0.0:
            raise ValueError("Parameter p should be greater than 0.")
        if q <= 0.0:
            raise ValueError("Parameter q should be greater than 0.")
        roots = self._roots(nodes)
        n = int(n)
        length = int(length)
        walks_out = np.zeros((roots.size * n, length), dtype=np.int32)
        lens_out = np.zeros(roots.size * n, dtype=np.int32)
        deg_max = max(int(self.degrees.max()) if self.degrees.size else 1, 1)
        work = np.zeros(deg_max, dtype=np.float64)
        work2 = np.zeros(deg_max, dtype=np.float64)
        indices = np.zeros(deg_max, dtype=np.int32)
        lib().msg_biased_random_walk(
            addr(self.indptr), addr(self.colind), addr(roots), addr(walks_out),
            addr(lens_out), addr(work), addr(work2), addr(indices),
            roots.size, n, length, float(p), float(q), int(seed or 0),
        )
        return self._unpad(walks_out, lens_out)


def naive_weighted_choices(graph, weights, node: int, seed: int = 0) -> int:
    """
    Select a neighbour of `node` at random, weighted by `weights`, which are
    the transition probabilities for `node`'s neighbours in CSR order.

    Upstream's version:
    ```
    subinterval_ends = []
    running_total = 0
    for w in weights:
        if w < 0: raise ValueError(...)
        running_total += w
        subinterval_ends.append(running_total)
    x = rs.random() * running_total
    for idx, end in enumerate(subinterval_ends):
        if x < end: break
    return idx
    ```

    The generator is the same fixed LCG the walkers use, not
    `numpy.random.RandomState`; that is the one documented divergence.
    """
    indptr, colind = _as_csr_graph(graph)
    deg = int(indptr[node + 1]) - int(indptr[node])
    work = np.zeros(max(deg, 1), dtype=np.float64)
    weights = f64(np.asarray(weights, dtype=np.float64).reshape(-1))
    choice = lib().msg_naive_weighted_choices(
        addr(indptr), addr(colind), addr(weights), addr(work), int(node),
        (int(seed) + _INCR) & _MASK,
    )
    if choice < 0:
        raise ValueError("Detected negative weight in the transition weights")
    return int(choice)


def _as_csr_graph(graph):
    if isinstance(graph, GraphWalk):
        return graph.indptr, graph.colind
    if isinstance(graph, tuple) and len(graph) == 2:
        indptr, colind = graph
        n = int(np.asarray(indptr).size) - 1
        return i32(indptr), i32(colind)
    return csr_from_edges(graph, int(np.asarray(graph).max()) + 1)


_MASK = (1 << 64) - 1
_MULT = 6364136223846793005
_INCR = 1442695040888963407
