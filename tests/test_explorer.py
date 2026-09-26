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
"""The path 0 - 1 - 2, plus nothing: node 2 is a leaf and node 3 is isolated."""


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
    got = _as_ints(
        sg.UniformRandomWalk(edges, n).run(nodes, n=n_walks, length=length, seed=seed)
    )
    indptr, colind = sg.csr_from_edges(edges, n)
    roots = list(range(n)) if nodes is None else [int(v) for v in nodes]
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
    got, want = _uniform_case(PATH_EDGES, 4, [3, 0], 1, 3, 5)
    assert got == want
    assert got[0] == [3], "an isolated root gives a length-1 walk"
    assert got[1] == [0, 1, 2], "the walk is cut at the requested length"
    longer, _ = _uniform_case(PATH_EDGES, 4, [3, 0], 1, 6, 5)
    assert longer[0] == [3] and len(longer[1]) == 6


def test_uniform_random_walk_from_every_node_of_a_path():
    got, want = _uniform_case(PATH_EDGES, 4, None, 2, 6, 13)
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
            sg.UniformRandomWalk(edges, 12).run(None, n=4, length=6, seed=seed)
        )
        for walk in walks:
            for a, b in zip(walk, walk[1:]):
                assert (a, b) in adjacent


# ------------------------------------------------------------ biased walks
def _biased_case(edges, n, nodes, n_walks, p, q, length, seed):
    got = _as_ints(
        sg.BiasedRandomWalk(edges, n).run(
            nodes, n=n_walks, p=p, q=q, length=length, seed=seed
        )
    )
    indptr, colind = sg.csr_from_edges(edges, n)
    roots = list(range(n)) if nodes is None else [int(v) for v in nodes]
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
    got, want = _biased_case(PATH_EDGES, 4, None, 2, 1.0, 1.0, 5, 9)
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
        sg.BiasedRandomWalk(edges, 6).run([0], n=1, p=p, q=q, length=3, seed=0)


def test_biased_random_walk_weighted_is_not_implemented():
    edges = _undirected_edges(6, 0.5, seed=7)
    with pytest.raises(NotImplementedError):
        sg.BiasedRandomWalk(edges, 6).run(
            [0], n=1, p=1.0, q=1.0, length=3, seed=0, weighted=True
        )


def test_biased_random_walk_is_a_path_in_the_graph():
    edges = _undirected_edges(12, 0.3, seed=3)
    adjacent = {(int(u), int(v)) for u, v in edges}
    for seed in (0, 1, 2):
        walks = _as_ints(
            sg.BiasedRandomWalk(edges, 12).run(
                None, n=4, p=0.5, q=2.0, length=6, seed=seed
            )
        )
        for walk in walks:
            for a, b in zip(walk, walk[1:]):
                assert (a, b) in adjacent


# ------------------------------------------------- naive_weighted_choices
def _choice_case(edges, n, weights, node, seed):
    """The port returns the chosen neighbour, the oracle the index into the
    node's neighbour list; the two have to be reconciled through the CSR."""
    got = sg.naive_weighted_choices(
        sg.csr_from_edges(edges, n), weights, node, seed=seed
    )
    indptr, colind = sg.csr_from_edges(edges, n)
    nbrs = _neighbours(indptr, colind, node)
    want = nbrs[ref.naive_weighted_choices(list(weights), ref.LCG(seed))]
    return got, want


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 99])
def test_naive_weighted_choices_matches_oracle(seed):
    edges = _star_edges(4)
    weights = [1.0, 2.0, 3.0, 4.0]
    got, want = _choice_case(edges, 5, weights, 0, seed)
    assert got == want
    assert got in (1, 2, 3, 4)


@pytest.mark.parametrize("weights", [[1.0, 1.0, 1.0], [0.0, 1.0, 4.0], [3.0, 0.0, 0.0]])
def test_naive_weighted_choices_weight_vectors(weights):
    # node 0 of a 3-leaf star has exactly these three neighbours, in CSR order
    edges = _star_edges(3)
    for seed in range(6):
        got, want = _choice_case(edges, 4, weights, 0, seed)
        assert got == want


def test_naive_weighted_choices_never_picks_a_zero_weight():
    # the sub-interval for a zero weight is empty, so upstream's `x < end` scan
    # can never land there; only the last index is the fallback
    edges = _star_edges(3)
    indptr, colind = sg.csr_from_edges(edges, 4)
    weights = [0.0, 1.0, 3.0]
    for seed in range(50):
        got = sg.naive_weighted_choices((indptr, colind), weights, 0, seed=seed)
        want = ref.naive_weighted_choices(weights, ref.LCG(seed))
        assert got == _neighbours(indptr, colind, 0)[want]
        assert got != 1, "the zero-weight neighbour is never chosen"


def test_naive_weighted_choices_picks_the_only_neighbour():
    edges = _star_edges(2)
    indptr, colind = sg.csr_from_edges(edges, 3)
    # node 1 has the single neighbour 0
    assert sg.naive_weighted_choices((indptr, colind), [2.0], 1, seed=4) == 0


def test_naive_weighted_choices_rejects_a_negative_weight():
    edges = _star_edges(3)
    indptr, colind = sg.csr_from_edges(edges, 4)
    with pytest.raises(ValueError):
        sg.naive_weighted_choices((indptr, colind), [1.0, -0.5, 2.0], 0)
    with pytest.raises(ValueError):
        ref.naive_weighted_choices([1.0, -0.5, 2.0], ref.LCG(0))


def test_naive_weighted_choices_is_the_cumulative_weight_split():
    # the choice is `argmin{idx : u * total < running_total(idx)}`, so a
    # one-hot weight vector always lands in its own interval, whatever u is
    edges = _star_edges(3)
    indptr, colind = sg.csr_from_edges(edges, 4)
    for k in range(3):
        weights = [1.0 if i == k else 0.0 for i in range(3)]
        for seed in range(8):
            got = sg.naive_weighted_choices((indptr, colind), weights, 0, seed=seed)
            assert got == k + 1


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
        sg.UniformRandomWalk(edges, n).run(None, n=200, length=6, seed=3)
    )
    visits = _visit_counts(walks, n)
    assert visits[0] > visits[1:].max()
    assert visits[0] > 4 * visits[1:].max()


def test_star_graph_biased_walks_also_favour_the_hub():
    leaves = 6
    n = leaves + 1
    edges = _star_edges(leaves)
    walks = _as_ints(
        sg.BiasedRandomWalk(edges, n).run(
            None, n=200, p=0.5, q=2.0, length=6, seed=3
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
        sg.UniformRandomWalk(edges, n).run(None, n=50, length=8, seed=1)
    )
    assert len(walks) == 50 * n
    assert all(len(w) == 8 for w in walks)
    assert all(0 <= v <= leaves for w in walks for v in w)
