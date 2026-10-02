"""Port of `stellargraph/data/explorer.py` (upstream 1.2.1): `UniformRandomWalk`,
`BiasedRandomWalk` and `naive_weighted_choices`.

Upstream walks a `StellarGraph` and draws from `numpy.random.RandomState`.
Here the graph is a CSR adjacency (`indptr`, `colind`), which is the same
neighbour relation in the layout the C ABI can carry, and the generator is a
fixed LCG, since a `RandomState` cannot cross the boundary. That RNG is the one
documented numerical divergence: the walks are the same algorithm over the same
neighbours, but they do not reproduce upstream's node sequences for a given
`seed`. Everything deterministic -- the neighbour relation, the dead-end
`break`, the `1/p` / `1.0` / `1/q` transition weights, the padding contract --
is upstream's and is parity-checked.

Upstream 1.2.1 moved `n` and `length` (and `p`, `q`, `weighted`) into the
constructor with keyword-only overrides on `run`, so that shape is kept here.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i32, lib, node_indices

_MASK = (1 << 64) - 1
_MULT = 6364136223846793005
_INCR = 1442695040888963407


def csr_from_edges(edges, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CSR `(indptr, colind)` for an `[e, 2]` list of `(u, v)` edges, sorted by
    row. This is the same neighbour order upstream's `neighbor_arrays` gives,
    with the neighbours of a node in ascending order.

    `edges` may be an edge array or anything carrying `edges`/`node_list`, in
    which case `n` is taken from `node_list` unless given.
    """
    src = getattr(edges, "edges", edges)
    e = np.asarray(src, dtype=np.int64).reshape(-1, 2)
    if n is None:
        node_list = getattr(edges, "node_list", None)
        n = (
            int(np.asarray(node_list).size)
            if node_list is not None
            else (int(e.max()) + 1 if e.size else 0)
        )
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    if e.size and (int(e.min()) < 0 or int(e.max()) >= n):
        raise IndexError(
            "edge endpoint out of range for a graph of {} nodes".format(n)
        )
    counts = np.bincount(e[:, 0], minlength=n) if e.size else np.zeros(n, np.int64)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)
    return indptr, i32(e[:, 1])


def _as_csr(graph, n=None):
    """The CSR `(indptr, colind)` for `graph`, or `None` when the node count
    cannot be known yet (a bare edge list passed without one).

    Upstream takes a `StellarGraph`, whose node count is fixed. Here the count
    is `graph.node_list`'s length when there is one, and otherwise the largest
    node id in the edge list plus one, which is what `run` then has to be told
    explicitly via `n`.
    """
    if isinstance(graph, GraphWalk):
        return graph.indptr, graph.colind
    if isinstance(graph, tuple) and len(graph) == 2:
        return graph
    if n is None and not hasattr(graph, "edges") and not hasattr(graph, "node_list"):
        return None
    edges = getattr(graph, "edges", graph)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    node_list = getattr(graph, "node_list", None)
    if node_list is not None:
        n = int(np.asarray(node_list).size)
    elif n is None:
        n = int(arr.max()) + 1 if arr.size else 0
    return csr_from_edges(arr, n)


def _default_if_none(value, default, name, ensure_not_none: bool = True):
    """Upstream `_default_if_none`, verbatim."""
    value = value if value is not None else default
    if ensure_not_none and value is None:
        raise ValueError(
            f"{name}: expected a value to be specified in either `__init__` or "
            f"`run`, found None in both"
        )
    return value


def _require_integer_in_range(value, name, min_val: int = 0, max_val: int | None = None):
    """Upstream's argument check, which raises `ValueError`."""
    if value is None or not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name}: expected an integer value, found {value}")
    if value < min_val:
        raise ValueError(f"{name}: expected a value >= {min_val}, found {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name}: expected a value <= {max_val}, found {value}")


class GraphWalk:
    """
    Base class for exploring graphs.

    ```
    GraphWalk(graph, n=None, length=None, seed=None)
    ```

    Args:
        graph: an `[e, 2]` edge array, a `(indptr, colind)` CSR pair, or
            another `GraphWalk`; upstream takes a `StellarGraph`.
        n (int): total number of random walks per root node.
        length (int): maximum length of each random walk.
        seed (int): random number generator seed.
    """

    def __init__(self, graph, n=None, length=None, seed=None):
        self.graph = graph
        self.n = n
        self.length = length
        self.seed = seed
        self._check_seed(seed)

        if isinstance(graph, GraphWalk):
            indptr, colind = graph.indptr, graph.colind
        else:
            csr = _as_csr(graph)
            if csr is None:
                # A bare edge array names its own largest node id, so the graph
                # has `max + 1` nodes. Upstream's `StellarGraph` always knows.
                edges = np.asarray(graph, dtype=np.int64).reshape(-1, 2)
                csr = csr_from_edges(edges, int(edges.max()) + 1 if edges.size else 1)
            indptr, colind = csr
        # the kernels read both as int32, and `np.cumsum` hands back int64
        self.indptr, self.colind = i32(indptr), i32(colind)
        if int(self.indptr[-1]) > int(self.colind.size):
            raise ValueError("indptr counts more entries than colind holds")
        self.node_count = int(self.indptr.size) - 1
        self.degrees = np.diff(self.indptr.astype(np.int64))

    # ---------------------------------------------------- validation helpers
    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

    def _check_seed(self, seed):
        """Upstream `GraphWalk._check_seed`."""
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

    def _resolved_seed(self, seed) -> int:
        """The seed actually handed to the C ABI.

        Upstream's default is `seed=None`, meaning "draw one from NumPy's
        global state"; the stream it produces is the documented divergence from
        the fixed LCG, so this keeps the None as a seed rather than raising.
        A `None` seed still has to cross the ABI as an `Int`, so it becomes 0
        here: reproducible, and distinct from the `RandomState` stream upstream
        would have used.
        """
        if seed is None:
            return 0
        return int(seed)

    @staticmethod
    def _validate_walk_params(nodes, n, length):
        """Upstream `RandomWalk._validate_walk_params`."""
        if not isinstance(nodes, (list, tuple, np.ndarray)):
            raise ValueError(f"nodes: expected an iterable, found: {nodes}")
        if len(nodes) == 0:
            import warnings

            warnings.warn(
                "No root node IDs given. An empty list will be returned as a "
                "result.",
                RuntimeWarning,
                stacklevel=3,
            )
        _require_integer_in_range(n, "n", min_val=1)
        _require_integer_in_range(length, "length", min_val=1)

    def neighbors(self, node):
        """Upstream `GraphWalk.neighbors`: the neighbour ilocs of `node`."""
        return np.asarray(
            self.colind[self.indptr[node] : self.indptr[node + 1]], dtype=np.int64
        )

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def _roots(self, nodes):
        return node_indices(nodes, self.node_count)

    @staticmethod
    def _unpad(walks, lens):
        """The `[roots * n, length]` padding comes off, so this is the list of
        lists of node ids upstream returns."""
        return [[int(x) for x in w[:l]] for w, l in zip(walks, lens)]


class UniformRandomWalk(GraphWalk):
    """
    Performs uniform random walks on the given graph.

    ```
    UniformRandomWalk(graph, n=None, length=None, seed=None)
    run(nodes, *, n=None, length=None, seed=None)
    ```
    """

    def run(self, nodes, *, n=None, length=None, seed=None):
        """
        Perform a random walk starting from the root nodes. Optional parameters
        default to the values passed in during construction.

        Returns:
            A list of lists of node ids, one list per walk.
        """
        n = _default_if_none(n, self.n, "n")
        length = _default_if_none(length, self.length, "length")
        self._validate_walk_params(nodes, n, length)
        self._check_seed(seed)

        roots = self._roots(nodes)
        n = int(n)
        length = int(length)
        walks_out = np.zeros((roots.size * n, length), dtype=np.int32)
        lens_out = np.zeros(roots.size * n, dtype=np.int32)
        work = np.zeros(
            max(int(self.degrees.max()) if self.degrees.size else 1, 1), dtype=np.int32
        )
        lib().msg_uniform_random_walk(
            addr(self.indptr), addr(self.colind), addr(roots), addr(walks_out),
            addr(lens_out), addr(work), roots.size, n, length,
            self._resolved_seed(self.seed if seed is None else seed),
        )
        return self._unpad(walks_out, lens_out)


class BiasedRandomWalk(GraphWalk):
    """
    Performs biased second order random walks (like those used in Node2Vec,
    https://snap.stanford.edu/node2vec/) controlled by `p` and `q.

    ```
    BiasedRandomWalk(graph, n=None, length=None, p=1.0, q=1.0, weighted=False,
                     seed=None)
    run(nodes, *, n=None, length=None, p=None, q=None, seed=None, weighted=None)
    ```
    """

    def __init__(self, graph, n=None, length=None, p=1.0, q=1.0, weighted=False,
                 seed=None):
        super().__init__(graph, n=n, length=length, seed=seed)
        self.p = p
        self.q = q
        self.weighted = weighted
        if self.weighted:
            self._check_weights_valid()

    def _check_weights_valid(self):
        """Upstream checks that every edge weight is non-negative and finite."""
        weights = getattr(self.graph, "edge_weights", None)
        if weights is None:
            raise ValueError(
                "weighted=True needs `edge_weights` on the graph"
            )
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        invalid = np.nonzero((w < 0) | ~np.isfinite(w))[0]
        if invalid.size:
            raise ValueError(
                "graph: expected all edge weights to be non-negative and "
                "finite, found some negative or infinite: {}".format(
                    ", ".join(str(int(i)) for i in invalid)
                )
            )

    def _check_weights(self, p, q, weighted):
        """Upstream `BiasedRandomWalk._check_weights`, in the same order."""
        if p <= 0.0:
            raise ValueError(f"p: expected positive numeric value, found {p}")
        if q <= 0.0:
            raise ValueError(f"q: expected positive numeric value, found {q}")
        if type(weighted) != bool:
            raise ValueError(f"weighted: expected boolean value, found {weighted}")

    def run(self, nodes, *, n=None, length=None, p=None, q=None, seed=None,
            weighted=None):
        """
        Perform a biased random walk starting from the root nodes.

        `weighted=True` reads per-edge weights off the graph's `edge_weights`,
        which upstream reads off a `StellarGraph`.
        """
        n = _default_if_none(n, self.n, "n")
        length = _default_if_none(length, self.length, "length")
        p = _default_if_none(p, self.p, "p")
        q = _default_if_none(q, self.q, "q")
        weighted = _default_if_none(weighted, self.weighted, "weighted")
        self._validate_walk_params(nodes, n, length)
        self._check_weights(p, q, weighted)
        self._check_seed(seed)

        roots = self._roots(nodes)
        n = int(n)
        length = int(length)

        # upstream raises when `1/p` or `1/q` overflows the weight dtype; the
        # kernels take float64, so the same guard is applied here
        ip = 1.0 / float(p)
        iq = 1.0 / float(q)
        if np.isinf(ip):
            raise ValueError(f"p: value ({p}) is too small.")
        if np.isinf(iq):
            raise ValueError(f"q: value ({q}) is too small.")

        walks_out = np.zeros((roots.size * n, length), dtype=np.int32)
        lens_out = np.zeros(roots.size * n, dtype=np.int32)
        deg_max = max(int(self.degrees.max()) if self.degrees.size else 1, 1)
        work = np.zeros(deg_max, dtype=np.float64)
        work2 = np.zeros(deg_max, dtype=np.float64)
        indices = np.zeros(deg_max, dtype=np.int32)
        if weighted:
            self._check_weights_valid()
            weights = self._edge_weight_array()
        else:
            # `weights = np.ones(...)` upstream; the kernel substitutes the
            # constant, so no buffer crosses
            weights = np.zeros(1, dtype=np.float64)
        lib().msg_biased_random_walk(
            addr(self.indptr), addr(self.colind), addr(weights), addr(roots),
            addr(walks_out), addr(lens_out), addr(work), addr(work2),
            addr(indices), roots.size, n, length, float(p), float(q),
            self._resolved_seed(self.seed if seed is None else seed),
            int(bool(weighted)),
        )
        return self._unpad(walks_out, lens_out)

    def _edge_weight_array(self):
        """The per-edge weights in the CSR edge order, or `None`.

        Upstream reads them straight off the `StellarGraph`; the CSR pair the
        kernels take has no weight channel, so this keeps a parallel array in
        the same sorted order `csr_from_edges` produced.
        """
        weights = getattr(self.graph, "edge_weights", None)
        if weights is None:
            raise ValueError("weighted=True needs `edge_weights` on the graph")
        w = f64(weights).reshape(-1)
        edges = np.asarray(getattr(self.graph, "edges"), dtype=np.int64).reshape(-1, 2)
        if w.size != edges.shape[0]:
            raise ValueError(
                "edge_weights has {} entries for {} edges".format(
                    w.size, edges.shape[0]
                )
            )
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        return np.ascontiguousarray(w[order])


def naive_weighted_choices(weights, seed: int = 0, size: int | None = None):
    """Upstream `naive_weighted_choices(rs, weights, size=None)`.

    ```
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        return None
    thresholds = rs.random() if size is None else rs.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")
    ```

    Upstream's first argument is a `RandomState`. Here it is a `seed` for the
    fixed LCG the walkers use, which is the documented divergence.

    `np.searchsorted(..., side="left")` returns the index of the *first* entry
    `>= x`, which is why a leading zero weight can be returned even though its
    sub-interval is empty; the walk kernels reproduce this by scanning for the
    first index whose running total reaches the threshold. That scan is what
    runs here, rather than a hand-rolled binary search, so the tie behaviour
    at a zero threshold matches exactly.
    """
    w = f64(weights).reshape(-1)
    if w.size == 0:
        return None
    probs = np.zeros(w.size, dtype=np.float64)
    running = 0.0
    for i in range(w.size):
        if w[i] < 0:
            raise ValueError("Detected negative weight in the transition weights")
        running += w[i]
        probs[i] = running
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so upstream chooses nothing
        return None

    draws = int(size) if size is not None else 1
    state = (int(seed) + _INCR) & _MASK
    thresholds = []
    for _ in range(draws):
        state = (state * _MULT + _INCR) & _MASK
        thresholds.append(((state >> 11) & ((1 << 53) - 1)) / float(1 << 53))

    # `np.searchsorted(probs, x, side="left")`: the first index with
    # `probs[i] >= x`, scanning forward, exactly as the Mojo kernel does.
    idx = []
    for t in thresholds:
        x = t * total
        chosen = w.size - 1
        for i in range(w.size):
            if probs[i] >= x:
                chosen = i
                break
        idx.append(int(chosen))
    return idx[0] if size is None else np.asarray(idx, dtype=np.int64)