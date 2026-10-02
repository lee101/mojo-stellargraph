"""Parity tests for `sg.UniformRandomWalk`, `sg.BiasedRandomWalk`,
`sg.csr_from_edges` and `sg.naive_weighted_choices`.

Upstream `stellargraph` 0.8.1 cannot be installed here (it needs Python < 3.9
and TensorFlow 2.1), so `tests/upstream_reference.py` -- a line-by-line NumPy
transliteration of `stellargraph/data/explorer.py` with the same fixed LCG the
port uses in place of `numpy.random.RandomState` -- is the parity oracle. Both
sides drive that one generator, so the walks are compared for exact equality
rather than to a tolerance. The statistical tests at the end check properties
that hold whatever the generator does.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref


def _undirected_edges(n, p, seed):
    """A random undirected edge list, both directions, no self loops."""
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p).astype(np.int64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    return np.array(np.nonzero(a), dtype=np.int64).T


def _star_edges(leaves):
    """A star with node 0 as the hub, both directions. Node 0 therefore has
    exactly `leaves` neighbours, 1..leaves, in that CSR order."""
    out = []
    for leaf in range(1, leaves + 1):
        out.append((0, leaf))
        out.append((leaf, 0))
    return np.array(out, dtype=np.int64)


PATH_EDGES = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64)
"""The path 0 - 1 - 2, plus nothing: node 2 is a leaf and node 3 is isolated.

Node 3 appears in no edge, so the graph object names it explicitly; a bare
edge list would only imply nodes 0..2.
"""


class _Graph:
    """The node set upstream's `StellarGraph` carries, which a bare edge list
    cannot express when a node is isolated."""

    def __init__(self, edges, node_count):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.node_list = np.arange(node_count)


def _neighbours(indptr, colind, node):
    return [
        int(colind[k]) for k in range(int(indptr[node]), int(indptr[node + 1]))
    ]


def _as_ints(walks):
    """The port returns a list of lists of `np.int32`; the oracle plain ints."""
    return [[int(v) for v in w] for w in walks]


# ------------------------------------------------------------------- CSR
def test_csr_from_edges_groups_and_sorts_by_row():
    # deliberately unsorted edges, and both directions present
    edges = np.array(
        [[2, 3], [0, 2], [2, 0], [0, 1], [1, 0], [2, 1]], dtype=np.int64
    )
    indptr, colind = sg.csr_from_edges(edges, 4)
    assert indptr.dtype == np.int32
    assert colind.dtype == np.int32
    assert indptr.shape == (5,)
    assert list(indptr) == [0, 2, 3, 6, 6]
    for node in range(4):
        got = _neighbours(indptr, colind, node)
        assert got == sorted(got), "each row is sorted, as networkx neighbours are"
        assert got == sorted(int(v) for u, v in edges if int(u) == node)
    assert _neighbours(indptr, colind, 3) == []


def test_csr_from_edges_keeps_trailing_isolated_nodes():
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    indptr, colind = sg.csr_from_edges(edges, 5)
    assert list(indptr) == [0, 1, 2, 2, 2, 2]
    assert _neighbours(indptr, colind, 4) == []


def test_csr_from_edges_of_an_empty_graph():
    indptr, colind = sg.csr_from_edges(np.zeros((0, 2), dtype=np.int64), 3)
    assert list(indptr) == [0, 0, 0, 0]
    assert colind.shape == (0,)


# ---------------------------------------------------------- uniform walks
def _uniform_case(edges, n, nodes, n_walks, length, seed):
    # upstream's `_check_nodes` rejects `nodes=None`, so "every node" is
    # spelled out here rather than left to a default
    roots = list(range(n)) if nodes is None else [int(v) for v in nodes]
    got = _as_ints(
        sg.UniformRandomWalk(edges, n=n_walks, length=length).run(roots, seed=seed)
    )
    indptr, colind = sg.csr_from_edges(edges, n)
    want = ref.uniform_random_walk(indptr, colind, roots, n_walks, length, seed)
    return got, want


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_uniform_random_walk_matches_oracle(seed):
    edges = _undirected_edges(12, 0.3, seed=3)
    got, want = _uniform_case(edges, 12, None, 3, 5, seed)
    assert got == want
    assert len(got) == 36


@pytest.mark.parametrize("length", [1, 2, 4, 9])
def test_uniform_random_walk_lengths(length):
    edges = _undirected_edges(10, 0.35, seed=4)
    got, want = _uniform_case(edges, 10, None, 2, length, 11)
    assert got == want
    assert all(len(w) == length for w in got)


def test_uniform_random_walk_over_a_node_subset():
    edges = _undirected_edges(14, 0.3, seed=5)
    nodes = [13, 0, 7, 7, 2]
    got, want = _uniform_case(edges, 14, nodes, 2, 4, 5)
    assert got == want
    assert len(got) == 10
    # every walk starts at one of the requested roots
    assert {w[0] for w in got} <= {13, 0, 7, 2}


def test_uniform_random_walk_stops_at_a_dead_end():
    # node 3 is isolated and the path 0 - 1 - 2 ends at the leaf 2. Only an
    # isolated node stops the walk: upstream breaks on `len(neighbours) == 0`,
    # never on being a leaf, so the leaf is exercised by the requested length.
    graph = _Graph(PATH_EDGES, 4)
    got, want = _uniform_case(graph, 4, [3, 0], 1, 3, 5)
    assert got == want
    assert got[0] == [3], "an isolated root gives a length-1 walk"
    assert got[1] == [0, 1, 2], "the walk is cut at the requested length"
    longer, _ = _uniform_case(graph, 4, [3, 0], 1, 6, 5)
    assert longer[0] == [3] and len(longer[1]) == 6


def test_uniform_random_walk_from_every_node_of_a_path():
    got, want = _uniform_case(_Graph(PATH_EDGES, 4), 4, None, 2, 6, 13)
    assert got == want
    # two walks per root, roots 0, 1, 2, 3 in order
    assert got[0] == [0, 1, 2, 1, 2, 1]
    assert got[1] == [0, 1, 2, 1, 0, 1]
    assert got[2] == [1, 2, 1, 0, 1, 0]
    assert got[3] == [1, 2, 1, 0, 1, 2]
    assert got[4] == [2, 1, 2, 1, 2, 1]
    assert got[5] == [2, 1, 2, 1, 0, 1]
    assert got[6] == got[7] == [3]
    assert all(len(w) == 6 for w in got[:6])
    assert all(len(w) == 1 for w in got[6:])


def test_uniform_random_walk_on_a_single_edge():
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    got, want = _uniform_case(edges, 2, None, 5, 7, 0)
    assert got == want
    # the only walk on K2 alternates, and shuffling two neighbours keeps it
    assert all(w == [0, 1, 0, 1, 0, 1, 0] for w in got[:5])


def test_uniform_random_walk_is_a_path_in_the_graph():
    edges = _undirected_edges(12, 0.3, seed=3)
    adjacent = {(int(u), int(v)) for u, v in edges}
    for seed in (0, 1, 2):
        walks = _as_ints(
            sg.UniformRandomWalk(edges, n=4, length=6).run(list(range(12)), seed=seed)
        )
        for walk in walks:
            for a, b in zip(walk, walk[1:]):
                assert (a, b) in adjacent


# ------------------------------------------------------------ biased walks
def _biased_case(edges, n, nodes, n_walks, p, q, length, seed):
    roots = list(range(n)) if nodes is None else [int(v) for v in nodes]
    got = _as_ints(
        sg.BiasedRandomWalk(edges, n=n_walks, p=p, q=q, length=length).run(
            roots, seed=seed
        )
    )
    indptr, colind = sg.csr_from_edges(edges, n)
    want = ref.biased_random_walk(indptr, colind, roots, n_walks, p, q, length, seed)
    return got, want


@pytest.mark.parametrize("p, q", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (3.0, 3.0)])
def test_biased_random_walk_matches_oracle(p, q):
    edges = _undirected_edges(12, 0.3, seed=3)
    got, want = _biased_case(edges, 12, None, 3, p, q, 5, 4)
    assert got == want
    assert len(got) == 36


@pytest.mark.parametrize("length", [1, 2, 3, 8])
@pytest.mark.parametrize("seed", [0, 6])
def test_biased_random_walk_lengths_and_seeds(length, seed):
    edges = _undirected_edges(9, 0.4, seed=6)
    got, want = _biased_case(edges, 9, [4, 0, 8], 2, 0.25, 4.0, length, seed)
    assert got == want
    assert len(got) == 6
    # a length-1 walk is the root alone; otherwise the root is always first
    assert all(1 <= len(w) <= length for w in got)
    assert all(w[0] in (4, 0, 8) for w in got)


def test_biased_random_walk_stops_at_a_dead_end():
    got, want = _biased_case(_Graph(PATH_EDGES, 4), 4, None, 2, 1.0, 1.0, 5, 9)
    assert got == want
    # two walks per root, roots 0..3 in order
    assert got[0] == [0, 1, 0, 1, 2]
    assert got[1] == [0, 1, 0, 1, 0]
    assert got[4] == got[5] == [2, 1, 0, 1, 2]
    # the isolated node 3 has no neighbours, so upstream's `if not neighbours:
    # break` fires on the first step and the walk is the root alone
    assert got[6] == got[7] == [3]
    assert all(len(w) == 5 for w in got[:6])
    assert all(len(w) == 1 for w in got[6:])


@pytest.mark.parametrize("p, q", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)])
def test_biased_random_walk_rejects_non_positive_p_and_q(p, q):
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(ValueError):
        sg.BiasedRandomWalk(edges, n=1, p=p, q=q, length=3).run([0], seed=0)


def test_biased_random_walk_weighted_uses_the_edge_weights():
    """Upstream 1.2.1 added `weighted`, reading per-edge weights off the
    graph. The ported walk reads them off the graph's `edge_weights`."""
    from conftest import Graph

    adj = np.zeros((6, 6), dtype=np.float64)
    for u, v in _undirected_edges(6, 0.5, seed=7):
        adj[u, v] = 1.0
    weights = np.abs(np.random.default_rng(4).normal(size=int(adj.sum()))) + 0.25
    plain = Graph(adj, np.ones((6, 1)))
    plain.edge_weights = weights
    weighted_walks = sg.BiasedRandomWalk(
        plain, n=3, length=6, p=0.5, q=2.0, weighted=True
    ).run(list(range(6)), seed=3)
    unweighted_walks = sg.BiasedRandomWalk(
        plain, n=3, length=6, p=0.5, q=2.0
    ).run(list(range(6)), seed=3)
    assert len(weighted_walks) == len(unweighted_walks) == 3 * 6
    # the weights change the draw, so the two walks must differ somewhere
    assert weighted_walks != unweighted_walks


def test_biased_random_walk_weighted_rejects_a_negative_edge_weight():
    from conftest import Graph

    adj = np.zeros((6, 6), dtype=np.float64)
    for u, v in _undirected_edges(6, 0.5, seed=7):
        adj[u, v] = 1.0
    bad = np.ones(int(adj.sum()))
    bad[0] = -1.0
    g = Graph(adj, np.ones((6, 1)), weights=bad)
    with pytest.raises(ValueError, match="non-negative"):
        sg.BiasedRandomWalk(g, n=1, length=3, weighted=True)


# ------------------------------------------- _check_common_parameters
@pytest.mark.parametrize("walker", [sg.UniformRandomWalk, sg.BiasedRandomWalk])
def test_walkers_reject_missing_root_nodes(walker):
    """Upstream's `_validate_walk_params`: `nodes=None` is an error, not
    "every node"."""
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(ValueError, match="expected an iterable"):
        walker(edges, n=1, length=3).run(None, seed=0)


@pytest.mark.parametrize("walker", [sg.UniformRandomWalk, sg.BiasedRandomWalk])
@pytest.mark.parametrize("n", [0, -1, 1.5, "2"])
def test_walkers_reject_a_non_positive_or_non_integer_n(walker, n):
    """Upstream's `require_integer_in_range(n, "n", min_val=1)`."""
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(ValueError, match="n:"):
        walker(edges, n=n, length=3).run([0], seed=0)


@pytest.mark.parametrize("walker", [sg.UniformRandomWalk, sg.BiasedRandomWalk])
@pytest.mark.parametrize("length", [0, -3, 2.5])
def test_walkers_reject_a_non_positive_or_non_integer_length(walker, length):
    """Upstream's `require_integer_in_range(length, "length", min_val=1)`.
    Length 0 is invalid by consensus."""
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(ValueError, match="length:"):
        walker(edges, n=1, length=length).run([0], seed=0)


@pytest.mark.parametrize("walker", [sg.UniformRandomWalk, sg.BiasedRandomWalk])
@pytest.mark.parametrize("seed", [-1, 1.5, "3"])
def test_walkers_reject_a_bad_seed(walker, seed):
    """Upstream's `GraphWalk._check_seed`."""
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(ValueError, match="seed"):
        walker(edges, n=1, length=3).run([0], seed=seed)


@pytest.mark.parametrize("walker", [sg.UniformRandomWalk, sg.BiasedRandomWalk])
def test_walkers_report_a_parameter_missing_from_both_init_and_run(walker):
    """Upstream's `_default_if_none(..., ensure_not_none=True)`."""
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(ValueError, match="expected a value to be specified"):
        walker(edges).run([0], seed=0)


def test_biased_random_walk_is_a_path_in_the_graph():
    edges = _undirected_edges(12, 0.3, seed=3)
    adjacent = {(int(u), int(v)) for u, v in edges}
    for seed in (0, 1, 2):
        walks = _as_ints(
            sg.BiasedRandomWalk(edges, n=4, p=0.5, q=2.0, length=6).run(
                list(range(12)), seed=seed
            )
        )
        for walk in walks:
            for a, b in zip(walk, walk[1:]):
                assert (a, b) in adjacent


# ------------------------------------------------- naive_weighted_choices
def _choice_case(edges, n, weights, node, seed):
    """The port returns the index into `weights`, which is what upstream's
    `naive_weighted_choices(rs, weights)` returns; the CSR is only used by the
    oracle to name the neighbours."""
    got = sg.naive_weighted_choices(weights, seed=seed)
    want = ref.naive_weighted_choices(list(weights), ref.LCG(seed))
    return got, want


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 99])
def test_naive_weighted_choices_matches_oracle(seed):
    weights = [1.0, 2.0, 3.0, 4.0]
    got, want = _choice_case(None, 5, weights, 0, seed)
    assert got == want
    assert got in (0, 1, 2, 3)


@pytest.mark.parametrize("weights", [[1.0, 1.0, 1.0], [0.0, 1.0, 4.0], [3.0, 0.0, 0.0]])
def test_naive_weighted_choices_weight_vectors(weights):
    for seed in range(6):
        got, want = _choice_case(None, 4, weights, 0, seed)
        assert got == want


def test_naive_weighted_choices_never_picks_a_zero_weight():
    # the sub-interval for a zero weight is empty, so upstream's
    # `searchsorted` can never land there
    weights = [0.0, 1.0, 3.0]
    for seed in range(50):
        got = sg.naive_weighted_choices(weights, seed=seed)
        assert got != 0, "the zero-weight neighbour is never chosen"
        assert got == ref.naive_weighted_choices(weights, ref.LCG(seed))


def test_naive_weighted_choices_picks_the_only_neighbour():
    assert sg.naive_weighted_choices([2.0], seed=4) == 0


def test_naive_weighted_choices_rejects_a_negative_weight():
    with pytest.raises(ValueError, match="negative weight"):
        sg.naive_weighted_choices([1.0, -0.5, 2.0], seed=0)


def test_naive_weighted_choices_returns_none_when_every_weight_is_zero():
    """Upstream's `if total == 0: return None`."""
    assert sg.naive_weighted_choices([0.0, 0.0, 0.0], seed=0) is None


def test_naive_weighted_choices_size_draws_one_index_per_threshold():
    """Upstream's `size` argument draws `rs.random(size)` thresholds and
    returns an array; the C ABI returns a single index, so `size` is honoured
    by the same loop."""
    weights = [1.0, 1.0, 1.0, 1.0]
    got = sg.naive_weighted_choices(weights, seed=3, size=5)
    assert got.shape == (5,)
    assert np.all((got >= 0) & (got < 4))


def test_naive_weighted_choices_is_the_cumulative_weight_split():
    # the choice is `searchsorted(cumsum(weights), u * total, side="left")`,
    # so a one-hot weight vector always lands in its own interval
    for k in range(3):
        weights = [1.0 if i == k else 0.0 for i in range(3)]
        for seed in range(8):
            assert sg.naive_weighted_choices(weights, seed=seed) == k


# -------------------------------------------------------- statistical checks
def _visit_counts(walks, n):
    visits = np.zeros(n, dtype=np.int64)
    for walk in walks:
        for node in walk:
            visits[node] += 1
    return visits


def test_star_graph_walks_visit_the_hub_more_than_any_leaf():
    # every leaf has degree 1, so a walk arriving at a leaf must leave again
    # through the hub: the hub is a far more frequent destination than any
    # single leaf, whichever generator drives the walk
    leaves = 6
    n = leaves + 1
    edges = _star_edges(leaves)
    walks = _as_ints(
        sg.UniformRandomWalk(edges, n=200, length=6).run(list(range(n)), seed=3)
    )
    visits = _visit_counts(walks, n)
    assert visits[0] > visits[1:].max()
    assert visits[0] > 4 * visits[1:].max()


def test_star_graph_biased_walks_also_favour_the_hub():
    leaves = 6
    n = leaves + 1
    edges = _star_edges(leaves)
    walks = _as_ints(
        sg.BiasedRandomWalk(edges, n=200, p=0.5, q=2.0, length=6).run(
            list(range(n)), seed=3
        )
    )
    visits = _visit_counts(walks, n)
    assert visits[0] > visits[1:].max()
    assert visits[0] > 4 * visits[1:].max()


def test_star_graph_walk_nodes_and_lengths():
    leaves = 6
    n = leaves + 1
    edges = _star_edges(leaves)
    walks = _as_ints(
        sg.UniformRandomWalk(edges, n=50, length=8).run(list(range(n)), seed=1)
    )
    assert len(walks) == 50 * n
    assert all(len(w) == 8 for w in walks)
    assert all(0 <= v <= leaves for w in walks for v in w)


def test_no_seed_is_a_valid_seed():
    """`seed=None` is upstream's default, so it must run rather than raise.

    Upstream's `None` means "draw from NumPy's global random state", which the
    fixed LCG cannot reproduce; here a `None` seed crosses the C ABI as 0. That
    is the documented RNG divergence, not a failure to accept the default.
    """
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 0], [2, 1]], dtype=np.int64)
    for walker in (
        sg.UniformRandomWalk(edges, n=2, length=4),
        sg.BiasedRandomWalk(edges, n=2, p=0.5, q=2.0, length=4),
    ):
        assert walker.seed is None
        from_ctor = _as_ints(walker.run([0, 1, 2]))
        from_run = _as_ints(walker.run([0, 1, 2], seed=None))
        assert from_ctor == from_run, "a None seed must be stable across run()"
        assert len(from_ctor) == 6
        for walk in from_ctor:
            assert 1 <= len(walk) <= 4
            assert walk[0] in (0, 1, 2)
            assert all(0 <= node < 3 for node in walk)


def test_an_explicit_seed_still_overrides_a_none_constructor_seed():
    """A seed given to `run` wins, and a seed given to the constructor is used
    when `run` is given none. The claim is replay, not a particular walk: a
    four-node graph can produce the same walk for two seeds."""
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 0], [2, 1]], dtype=np.int64)
    walker = sg.UniformRandomWalk(edges, n=4, length=6)
    a = _as_ints(walker.run([0, 1, 2], seed=7))
    assert a == _as_ints(walker.run([0, 1, 2], seed=7))
    seeded = sg.UniformRandomWalk(edges, n=4, length=6, seed=7)
    assert _as_ints(seeded.run([0, 1, 2])) == a
    assert _as_ints(seeded.run([0, 1, 2], seed=8)) == _as_ints(
        sg.UniformRandomWalk(edges, n=4, length=6, seed=8).run([0, 1, 2])
    )
