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

from ._lib import addr, f64, i32, lib, node_indices


def csr_from_edges(edges, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR `(indptr, colind)` for an `[e, 2]` list of `(u, v)` edges, sorted by
    row. This is the same neighbour order `self.neighbors(node)` gives
    upstream, where `node.successors()`/`predecessors()` are sorted."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    if e.size and (int(e.min()) < 0 or int(e.max()) >= n):
        raise IndexError(
            "edge endpoint out of range for a graph of {} nodes".format(n)
        )
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
        indptr, colind = _as_csr(graph, self.n)
        # the kernels read both as int32, and `np.cumsum` hands back int64
        self.indptr, self.colind = i32(indptr), i32(colind)
        if int(self.indptr[-1]) > int(self.colind.size):
            raise ValueError("indptr counts more entries than colind holds")
        self.degrees = np.diff(self.indptr.astype(np.int64))

    @staticmethod
    def _raise_error(message):
        raise ValueError(message)

    def _check_nodes(self, nodes):
        if nodes is None:
            self._raise_error("A list of root node IDs was not provided.")
        if not isinstance(nodes, (list, tuple, np.ndarray)):
            self._raise_error("Nodes parameter should be an iterable of node IDs.")
        if len(nodes) == 0:
            print(
                "({}) WARNING: No root node IDs given. An empty list will be "
                "returned as a result.".format(type(self).__name__)
            )

    def _check_repetitions(self, n):
        if type(n) != int:
            self._raise_error(
                "The number of walks per root node, n, should be integer type."
            )
        if n <= 0:
            self._raise_error(
                "The number of walks per root node, n, should be a positive integer."
            )

    def _check_length(self, length):
        if type(length) != int:
            self._raise_error("The walk length, length, should be integer type.")
        if length <= 0:
            # Technically, length 0 should be okay, but by consensus is invalid.
            self._raise_error("The walk length, length, should be a positive integer.")

    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be "
                    "integer type or None."
                )
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be "
                    "non-negative integer or None."
                )

    def _check_common_parameters(self, nodes, n, length, seed):
        self._check_nodes(nodes)
        self._check_repetitions(n)
        self._check_length(length)
        self._check_seed(seed)

    def _roots(self, nodes):
        return node_indices(nodes, self.n)

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
        self._check_common_parameters(nodes, n, length, seed)
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
        self._check_common_parameters(nodes, n, length, seed)
        roots = self._roots(nodes)
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
    indptr, colind = i32(indptr), i32(colind)
    deg = int(indptr[node + 1]) - int(indptr[node])
    work = np.zeros(max(deg, 1), dtype=np.float64)
    weights = f64(np.asarray(weights, dtype=np.float64).reshape(-1))
    # the kernel reads one weight per neighbour, so a shorter array is an
    # out-of-bounds read rather than a wrong draw
    if deg > 0 and weights.size < deg:
        raise ValueError(
            "node {} has {} neighbours but only {} weights".format(
                node, deg, weights.size
            )
        )
    choice = lib().msg_naive_weighted_choices(
        addr(indptr), addr(colind), addr(weights), addr(work), int(node),
        (int(seed) + _INCR) & _MASK,
    )
    if choice == -2:
        raise ValueError(
            "node {} has no neighbours to choose from".format(node)
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
