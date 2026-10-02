"""Parity tests for `mojostellargraph.FullBatchNodeGenerator`.

Upstream `stellargraph` 0.8.1 needs Python < 3.9 and TensorFlow 2.1, so it
cannot be installed here and `tests/upstream_reference.py` -- a line-by-line
NumPy transliteration of the upstream functions -- is the oracle. The generator
itself is upstream Python, so what is under test is the dispatch over the
ported `core_utils` transforms plus the adjacency the edge list produces; each
transform is checked against the oracle and against a closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

ATOL = 1e-12
EIG_ATOL = 1e-9


class _Graph:
    """Everything upstream's `FullBatchNodeGenerator` reads off a `StellarGraph`."""

    def __init__(self, node_list, edges, features):
        self.node_list = list(node_list)
        self.edges = edges
        self.features = features


def _undirected_edges(adj: np.ndarray) -> np.ndarray:
    """Every `(i, j)` with `adj[i, j] != 0`, as an `[e, 2]` int array."""
    i, j = np.nonzero(adj)
    return np.stack([i, j], axis=1).astype(np.int64)


def _symmetric_graph(adj: np.ndarray, features: np.ndarray) -> _Graph:
    return _Graph(range(adj.shape[0]), _undirected_edges(adj), features)


def _from_pairs(n: int, edges) -> np.ndarray:
    """The `Aadj` the generator is documented to build: `A[src, dst] = 1`."""
    out = np.zeros((n, n), dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out[edges[:, 0], edges[:, 1]] = 1.0
    return out



# ------------------------------------------------------------------- gcn and sgc
@pytest.mark.parametrize("k", [1, 2])
def test_gcn_method_matches_reference(sym_graph, features, k):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="gcn", k=k)
    np.testing.assert_allclose(
        gen.Aadj, ref.gcn_aadj_feats_op(sym_graph, k, "gcn"), atol=ATOL, rtol=0.0
    )
    np.testing.assert_array_equal(gen.features, features)


@pytest.mark.parametrize("k", [1, 2])
def test_gcn_method_equals_the_preprocessing_layer(sym_graph, features, k):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="gcn", k=k)
    layer = sg.GraphPreProcessingLayer(sym_graph.shape[0])
    np.testing.assert_array_equal(gen.Aadj, layer(sym_graph))


def test_gcn_method_adds_a_self_loop_to_every_node(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="gcn")
    deg = (sym_graph + np.eye(sym_graph.shape[0])).sum(1)
    np.testing.assert_allclose(np.diag(gen.Aadj), 1.0 / deg, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("k", [1, 2])
def test_sgc_method_matches_reference(sym_graph, features, k):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="sgc", k=k)
    np.testing.assert_allclose(
        gen.Aadj, ref.gcn_aadj_feats_op(sym_graph, k, "sgc"), atol=ATOL, rtol=0.0
    )
    np.testing.assert_array_equal(gen.features, features)


@pytest.mark.parametrize("k", [1, 2])
def test_sgc_method_is_the_kth_power_of_the_gcn_adjacency(sym_graph, features, k):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="sgc", k=k)
    a_norm = sg.FullBatchNodeGenerator(g, method="gcn").Aadj
    np.testing.assert_allclose(
        gen.Aadj, np.linalg.matrix_power(a_norm, k), atol=ATOL, rtol=0.0
    )


@pytest.mark.parametrize("k", [3, 4])
def test_sgc_method_kth_power(sym_graph, features, k):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="sgc", k=k)
    np.testing.assert_allclose(
        gen.Aadj, ref.gcn_aadj_feats_op(sym_graph, k, "sgc"), atol=ATOL, rtol=0.0
    )


# --------------------------------------------------------- chebyshev (removed)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_chebyshev_method_is_rejected_as_upstream_removed_it(sym_graph, features, k):
    """Upstream 1.2.1 removed `method="chebyshev"` from
    `FullBatchGenerator.__init__`, so it falls through to the same
    "Undefined method" error as any other unknown method."""
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError, match="Undefined method"):
        sg.FullBatchNodeGenerator(g, method="chebyshev", k=k)


# ------------------------------------------------------------------------- none
@pytest.mark.parametrize("method", ["none", None])
def test_none_method_is_the_raw_edge_adjacency(sym_graph, features, method):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method=method)
    np.testing.assert_array_equal(gen.Aadj, sym_graph)
    assert gen.Aadj.max() == 1.0
    assert int((gen.Aadj != 0.0).sum()) == len(g.edges)
    np.testing.assert_array_equal(np.diag(gen.Aadj), np.zeros(sym_graph.shape[0]))
    np.testing.assert_array_equal(gen.features, features)


def test_none_method_does_not_symmetrize(features):
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    g = _Graph(range(4), edges, np.ones((4, 2)))
    gen = sg.FullBatchNodeGenerator(g, method="none")
    np.testing.assert_array_equal(gen.Aadj, _from_pairs(4, edges))
    assert not np.array_equal(gen.Aadj, gen.Aadj.T)
    assert not np.array_equal(gen.Aadj, ref._symmetrize(gen.Aadj))



def test_none_method_is_symmetric_when_the_edges_are(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="none")
    np.testing.assert_array_equal(gen.Aadj, gen.Aadj.T)
    np.testing.assert_array_equal(gen.Aadj, ref._symmetrize(gen.Aadj))


def test_none_method_equals_gcn_features_for_both_spellings(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    np.testing.assert_array_equal(
        sg.FullBatchNodeGenerator(g, method="none").Aadj,
        sg.FullBatchNodeGenerator(g, method=None).Aadj,
    )


# ----------------------------------------------------------------- gat and loops
@pytest.mark.parametrize("method", ["gat", "self_loops"])
def test_gat_and_self_loops_add_exactly_one_self_loop(sym_graph, features, method):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method=method)
    np.testing.assert_array_equal(gen.Aadj, sym_graph + np.eye(sym_graph.shape[0]))
    np.testing.assert_array_equal(np.diag(gen.Aadj), np.ones(sym_graph.shape[0]))
    np.testing.assert_array_equal(gen.features, features)


@pytest.mark.parametrize("method", ["gat", "self_loops"])
def test_gat_and_self_loops_do_not_normalize(sym_graph, features, method):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method=method)
    degrees = sym_graph.sum(1) + 1.0
    # Unnormalized, so a row sums to its degree rather than to one.
    np.testing.assert_array_equal(gen.Aadj.sum(1), degrees)
    assert gen.Aadj.max() == 1.0
    normalized = sg.normalize_adj(sym_graph + np.eye(sym_graph.shape[0]), True)
    np.testing.assert_array_equal(gen.Aadj != 0.0, normalized != 0.0)
    assert np.abs(gen.Aadj - normalized).max() > 0.1
    assert not np.allclose(gen.Aadj.sum(1), 1.0)


def test_gat_and_self_loops_produce_the_same_adjacency(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gat = sg.FullBatchNodeGenerator(g, method="gat")
    loops = sg.FullBatchNodeGenerator(g, method="self_loops")
    np.testing.assert_array_equal(gat.Aadj, loops.Aadj)
    np.testing.assert_array_equal(gat.features, loops.features)
    assert gat.Aadj.max() > 1.0 - 1e-12 and loops.Aadj.max() == 1.0


def test_self_loops_method_is_not_normalized_where_gcn_is(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gcn = sg.FullBatchNodeGenerator(g, method="gcn")
    loops = sg.FullBatchNodeGenerator(g, method="self_loops")
    # Same support, different magnitudes: the looped one keeps its 0/1 weights.
    np.testing.assert_array_equal(gcn.Aadj != 0.0, loops.Aadj != 0.0)
    assert np.abs(gcn.Aadj.max() - loops.Aadj.max()) > 0.5


# ------------------------------------------------------------------------- ppnp
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5])
def test_ppnp_method_matches_reference(sym_graph, features, alpha):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(
        g, method="ppnp", sparse=False, teleport_probability=alpha
    )
    np.testing.assert_allclose(
        gen.Aadj, ref.ppnp_aadj_feats_op(sym_graph, alpha), atol=EIG_ATOL, rtol=0.0
    )
    np.testing.assert_array_equal(gen.features, features)


def test_ppnp_method_defaults_to_a_tenth(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    default = sg.FullBatchNodeGenerator(g, method="ppnp", sparse=False)
    explicit = sg.FullBatchNodeGenerator(
        g, method="ppnp", sparse=False, teleport_probability=0.1
    )
    np.testing.assert_array_equal(default.Aadj, explicit.Aadj)
    np.testing.assert_allclose(
        default.Aadj, ref.ppnp_aadj_feats_op(sym_graph, 0.1), atol=EIG_ATOL, rtol=0.0
    )


def test_ppnp_at_alpha_one_is_the_identity(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(
        g, method="ppnp", sparse=False, teleport_probability=1.0
    )
    np.testing.assert_array_equal(gen.Aadj, np.eye(sym_graph.shape[0]))


def test_ppnp_is_dense_where_the_edge_list_is_sparse(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="ppnp", sparse=False)
    # The personalized PageRank matrix is dense even though the graph is not.
    assert int((gen.Aadj != 0.0).sum()) > int((sym_graph != 0.0).sum())


def test_ppnp_rejects_sparse_true(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(g, method="ppnp", sparse=True)


def test_ppnp_rejects_the_sparse_default(sym_graph, features):
    # `sparse` defaults to True upstream, so the default must raise too.
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(g, method="ppnp")


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_ppnp_rejects_an_out_of_range_alpha(sym_graph, features, alpha):
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(
            g, method="ppnp", sparse=False, teleport_probability=alpha
        )


# --------------------------------------------------------------------- transform
def test_transform_receives_the_raw_adjacency_and_features(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    seen = {}

    def transform(features=None, A=None):
        seen["features"] = features
        seen["A"] = A
        return features, A

    sg.FullBatchNodeGenerator(g, method="gcn", transform=transform)
    # The transform is called on the edge-list adjacency, before any
    # symmetrization, self looping or normalization.
    np.testing.assert_array_equal(seen["A"], sym_graph)
    np.testing.assert_array_equal(seen["features"], features)


def test_transform_overrides_the_method(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(
        g, method="gcn", transform=lambda features, A: (features * 2.0, A * 3.0)
    )
    np.testing.assert_array_equal(gen.Aadj, sym_graph * 3.0)
    np.testing.assert_array_equal(gen.features, features * 2.0)
    # Which is not what method="gcn" would have produced.
    assert np.abs(gen.Aadj.max() - 3.0) < ATOL


@pytest.mark.parametrize("method", ["gat", "ppnp", "none", "bogus"])
def test_transform_short_circuits_every_other_method(sym_graph, features, method):
    # An unknown method never reaches the dispatch when a transform is given,
    # so the transform is what decides the adjacency.
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(
        g, method=method, sparse=False, transform=lambda features, A: (features, A)
    )
    np.testing.assert_array_equal(gen.Aadj, sym_graph)
    np.testing.assert_array_equal(gen.features, features)


@pytest.mark.parametrize("bad", ["nope", 3, object(), np.zeros(3)])
def test_transform_must_be_callable(sym_graph, features, bad):
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(g, method="gcn", transform=bad)


# ------------------------------------------------------------ dispatch and shape
@pytest.mark.parametrize("method", ["gcn", "sgc", "gat", "self_loops", "none", None])
def test_every_method_produces_a_square_adjacency(sym_graph, features, method):
    n = sym_graph.shape[0]
    gen = sg.FullBatchNodeGenerator(_symmetric_graph(sym_graph, features), method=method)
    assert gen.Aadj.shape == (n, n)
    assert np.isfinite(gen.Aadj).all()
    assert gen.Aadj.dtype == np.float64


@pytest.mark.parametrize("method", ["bogus", "GCN", "gat2", "sgcn", 1])
def test_unknown_method_raises(sym_graph, features, method):
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(g, method=method)


@pytest.mark.parametrize("method,k", [("chebyshev", 1), ("sgc", 0)])
def test_bad_k_raises_for_the_smoothing_methods(sym_graph, features, method, k):
    g = _symmetric_graph(sym_graph, features)
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(g, method=method, k=k)


def test_node_list_and_len(sym_graph, features):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(g, method="gcn")
    assert list(gen.node_list) == list(range(sym_graph.shape[0]))
    assert len(gen) == sym_graph.shape[0]


def test_adjacency_is_sized_by_the_node_list(features):
    g = _Graph(range(5), np.array([[0, 1], [1, 2]], dtype=np.int64), features[:5])
    gen = sg.FullBatchNodeGenerator(g, method="self_loops")
    assert gen.Aadj.shape == (5, 5)
    np.testing.assert_array_equal(gen.Aadj, _from_pairs(5, g.edges) + np.eye(5))


def test_duplicate_edges_are_counted_once(features):
    edges = np.array([[0, 1], [0, 1], [1, 2]], dtype=np.int64)
    gen = sg.FullBatchNodeGenerator(
        _Graph(range(3), edges, features[:3]), method="self_loops"
    )
    # Two distinct edges plus the three self loops `method="self_loops"` adds;
    # the repeat of (0, 1) must not add a second one.
    np.testing.assert_array_equal(gen.Aadj, _from_pairs(3, edges) + np.eye(3))
    assert int((gen.Aadj != 0.0).sum()) == 5
    np.testing.assert_array_equal(gen.Aadj[0, 1], np.array(1.0))


def test_a_flat_edge_array_is_reshaped(features):
    pairs = sg.FullBatchNodeGenerator(
        _Graph(range(3), np.array([0, 1, 1, 2], dtype=np.int64), features[:3]),
        method="self_loops",
    )
    nested = sg.FullBatchNodeGenerator(
        _Graph(range(3), [[0, 1], [1, 2]], features[:3]), method="self_loops"
    )
    np.testing.assert_array_equal(pairs.Aadj, nested.Aadj)
    np.testing.assert_array_equal(pairs.Aadj, _from_pairs(3, [[0, 1], [1, 2]]) + np.eye(3))


def test_a_graph_without_edges_is_rejected(features):
    class NoEdges:
        node_list = [0, 1]
        features = np.ones((2, 2))

    with pytest.raises(TypeError):
        sg.FullBatchNodeGenerator(NoEdges(), method="gcn")


def test_an_empty_edge_list_gives_a_zero_adjacency(features):
    gen = sg.FullBatchNodeGenerator(
        _Graph(range(4), np.zeros((0, 2), dtype=np.int64), features[:4]), method="none"
    )
    np.testing.assert_array_equal(gen.Aadj, np.zeros((4, 4)))


def test_the_input_graph_is_not_mutated(sym_graph, features):
    edges = _undirected_edges(sym_graph)
    g = _symmetric_graph(sym_graph, features)
    g.edges = edges
    before = (edges.copy(), np.array(features, copy=True))
    sg.FullBatchNodeGenerator(g, method="gcn")
    np.testing.assert_array_equal(g.edges, before[0])
    np.testing.assert_array_equal(g.features, before[1])
    np.testing.assert_array_equal(g.edges, _undirected_edges(sym_graph))


@pytest.mark.parametrize("sparse", [True, False])
def test_the_constructor_arguments_are_stored(sym_graph, features, sparse):
    g = _symmetric_graph(sym_graph, features)
    gen = sg.FullBatchNodeGenerator(
        g, name="gen", method="gcn", k=2, sparse=sparse, teleport_probability=0.3
    )
    assert gen.name == "gen"
    assert gen.method == "gcn"
    assert gen.k == 2
    assert gen.use_sparse is sparse
    assert gen.teleport_probability == 0.3
    assert gen.graph is g


# ------------------------------------------------------------------------- flow
@pytest.mark.parametrize("method", ["gcn", "sgc", "gat", "self_loops", "none", None])
def test_flow_returns_node_ids_features_and_adjacency(sym_graph, features, method):
    gen = sg.FullBatchNodeGenerator(_symmetric_graph(sym_graph, features), method=method)
    node_ids = np.array([3, 1, 4])
    ids, out_features, out_adj = gen.flow(node_ids)
    np.testing.assert_array_equal(ids, node_ids)
    assert out_features is gen.features
    assert out_adj is gen.Aadj
    np.testing.assert_array_equal(out_adj, gen.Aadj)
    assert len(gen.flow(node_ids)) == 3


def test_flow_carries_the_features_through(sym_graph, features):
    gen = sg.FullBatchNodeGenerator(
        _symmetric_graph(sym_graph, features), method="sgc", k=2
    )
    _, out_features, out_adj = gen.flow(np.arange(4))
    assert out_features is gen.features
    np.testing.assert_array_equal(out_features, features)
    assert out_adj is gen.Aadj


def test_the_readme_usage_example_runs_and_says_what_it_says():
    """The example in README.md, executed: the same shapes, the same softmax
    rows and the same first walk."""
    rng = np.random.default_rng(0)
    n, d = 200, 32
    edge_array = np.array(np.nonzero(rng.random((n, n)) < 0.03), dtype=np.int64).T
    feature_matrix = np.ascontiguousarray(rng.normal(size=(n, d)))
    edges, features = edge_array, feature_matrix

    class Graph:
        node_list = np.arange(n)
        edges = edge_array
        features = feature_matrix

    gen = sg.FullBatchNodeGenerator(Graph(), method="gcn")
    assert gen.Aadj.shape == (200, 200)
    np.testing.assert_allclose(gen.Aadj, gen.Aadj.T, atol=1e-12)
    # `GCN_Aadj_feats_op(method="gcn")` is `D^-1/2 (A + I) D^-1/2` on the
    # symmetrized graph, so the diagonal is the self loop's `1 / deg(i)`.
    # `add_self_loops` is `A + diag(1 - diag(A))`, so a node that already has
    # a self loop in the edge list gets none added.
    raw = np.zeros((n, n))
    raw[edge_array[:, 0], edge_array[:, 1]] = 1.0
    raw = np.maximum(raw, raw.T)
    deg = raw.sum(1) + 1.0 - np.diag(raw)
    np.testing.assert_allclose(np.diag(gen.Aadj), 1.0 / deg, atol=1e-12)

    model = sg.GCN([16, 4], gen, activations=["elu", "softmax"])
    model.build(d, seed=0)
    out = model(features, gen.Aadj)
    assert out.shape == (200, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)

    gen_gat = sg.FullBatchNodeGenerator(Graph(), method="gat")
    gat = sg.GraphAttention(8, attn_heads=4, activation="softmax", final_layer=True)
    gat.build(d)
    gat.kernels[:] = rng.normal(size=(4, d, 8))
    gat.attn_kernels[:] = rng.normal(size=(4, 16))
    gat.biases[:] = rng.normal(size=(4, 8))
    out = gat(features, gen_gat.Aadj, np.arange(50))
    assert out.shape == (50, 32)

    walks = sg.UniformRandomWalk(edges, n=2, length=5, seed=7).run([0, 1, 2])
    assert len(walks) == 6
    assert all(len(w) == 5 for w in walks)
    assert all(w[0] in (0, 1, 2) for w in walks)
    # a real path: every step is an edge of the graph
    index = {tuple(sorted(e)) for e in edges}
    for walk in walks:
        for a, b in zip(walk, walk[1:]):
            assert (a, b) in index or (b, a) in index
