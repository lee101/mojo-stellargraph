"""Parity tests for `sg.PPNP`, `sg.APPNP` and their propagation layers.

Upstream `stellargraph` 0.8.1 cannot be installed here (it needs Python < 3.9
and TensorFlow 2.1), so `tests/upstream_reference.py` -- a line-by-line NumPy
transliteration of `stellargraph/layer/ppnp.py` and `stellargraph/layer/appnp.py`
-- is the parity oracle. Every numerical test below compares the Mojo port
against it; the closed-form tests pin the algebra the oracle cannot check
itself, namely that both propagation matrices are the personalized-PageRank
matrix `alpha (I - (1 - alpha) S)^-1` of the symmetrically normalized
adjacency `S`, which fixes `sqrt(degree + 1)` and is row-stochastic on a
regular graph.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

TOL = 1e-12
"""float64 algebra that sums the same terms in the same order."""

TOL_INV = 1e-9
"""Anything routed through a matrix inverse or a 3000-step power iteration."""


class _Graph:
    """The minimum upstream's `StellarGraph` offers the generator."""

    def __init__(self, adj, features):
        self.node_list = np.arange(adj.shape[0])
        self.edges = np.array(np.nonzero(adj), dtype=np.int64).T
        self.features = features


def _random_graph(n=14, p=0.25, seed=5):
    """A connected-enough undirected graph, no self loops, no isolated node."""
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p).astype(np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    for i in range(n - 1):  # a spanning path, so no node is isolated
        a[i, i + 1] = a[i + 1, i] = 1.0
    features = np.ascontiguousarray(rng.normal(size=(n, 7)))
    return a, features


def _cycle_graph(n=12):
    """A 2-regular graph, on which the symmetric normalization is doubly
    stochastic: `D^-1/2 A D^-1/2` has every row summing to 1, which is what
    makes the row-sum closed form below meaningful."""
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1.0
    rng = np.random.default_rng(11)
    return a, np.ascontiguousarray(rng.normal(size=(n, 7)))


def _sym_norm(adj):
    """`D^-1/2 A D^-1/2`, written here from the definition rather than through
    either the port or the oracle, so the closed forms are independent of both."""
    d = adj.sum(axis=1)
    return adj / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]


def _sqrt_degree(adj):
    """`sqrt(degree + 1)`: the eigenvector of `D^-1/2 (A + I) D^-1/2` that both
    propagation matrices fix. Both of them normalize `A + I`, not `A`."""
    return np.sqrt(adj.sum(axis=1) + 1.0)


@pytest.fixture(scope="module")
def gcn_gen():
    adj, features = _random_graph()
    return sg.FullBatchNodeGenerator(_Graph(adj, features), method="gcn")


@pytest.fixture(scope="module")
def ppnp_gen():
    adj, features = _random_graph()
    return sg.FullBatchNodeGenerator(
        _Graph(adj, features), method="ppnp", sparse=False
    )


# ------------------------------------------------------------ PPNP propagation
def test_ppnp_propagation_layer_matches_oracle(gcn_gen):
    layer = sg.PPNPPropagationLayer(7)
    got = layer.call(gcn_gen.features, gcn_gen.Aadj)
    want = ref.ppnp_propagation_layer_call(gcn_gen.features, gcn_gen.Aadj)
    assert got.shape == (14, 7)
    assert np.abs(got - want).max() <= TOL


def test_ppnp_propagation_layer_is_a_matrix_product(gcn_gen):
    layer = sg.PPNPPropagationLayer(7)
    got = layer.call(gcn_gen.features, gcn_gen.Aadj)
    assert np.abs(got - gcn_gen.Aadj @ gcn_gen.features).max() <= TOL


def test_ppnp_propagation_layer_final_layer_gathers(gcn_gen):
    out_indices = np.array([5, 0, 13, 2, 2], dtype=np.int32)
    layer = sg.PPNPPropagationLayer(7, final_layer=True)
    got = layer.call(gcn_gen.features, gcn_gen.Aadj, out_indices)
    want = ref.ppnp_propagation_layer_call(
        gcn_gen.features, gcn_gen.Aadj, out_indices
    )
    assert got.shape == (5, 7)
    assert np.abs(got - want).max() <= TOL
    # the gather is by node id, so the repeated id 2 repeats the row, and the
    # unsorted ids do not disturb each other
    assert np.abs(got[3] - got[4]).max() == 0.0


def test_ppnp_propagation_layer_without_final_layer_ignores_out_indices(
    gcn_gen,
):
    # upstream only gathers `if self.final_layer`, so the full [n, units] stays
    out_indices = np.array([3, 1], dtype=np.int32)
    layer = sg.PPNPPropagationLayer(7, final_layer=False)
    got = layer.call(gcn_gen.features, gcn_gen.Aadj, out_indices)
    want = ref.ppnp_propagation_layer_call(gcn_gen.features, gcn_gen.Aadj)
    assert got.shape == (14, 7)
    assert np.abs(got - want).max() <= TOL


# ------------------------------------------------------------ APPNP propagation
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 1.0])
def test_appnp_propagation_layer_matches_oracle(gcn_gen, alpha):
    rng = np.random.default_rng(3)
    propagated = np.ascontiguousarray(rng.normal(size=(14, 7)))
    layer = sg.APPNPPropagationLayer(7, teleport_probability=alpha)
    got = layer.call(propagated, gcn_gen.features, gcn_gen.Aadj)
    want = ref.appnp_propagation_layer_call(
        propagated, gcn_gen.features, gcn_gen.Aadj, alpha, None
    )
    assert got.shape == (14, 7)
    assert np.abs(got - want).max() <= TOL


def test_appnp_propagation_layer_closed_form(gcn_gen):
    # (1 - a) A propagated + a features, straight from the layer's docstring
    alpha = 0.25
    rng = np.random.default_rng(4)
    propagated = np.ascontiguousarray(rng.normal(size=(14, 7)))
    layer = sg.APPNPPropagationLayer(7, teleport_probability=alpha)
    got = layer.call(propagated, gcn_gen.features, gcn_gen.Aadj)
    want = (1.0 - alpha) * (gcn_gen.Aadj @ propagated) + alpha * gcn_gen.features
    assert np.abs(got - want).max() <= TOL


def test_appnp_propagation_layer_final_layer_gathers(gcn_gen):
    alpha = 0.1
    out_indices = np.array([4, 9, 4], dtype=np.int32)
    rng = np.random.default_rng(6)
    propagated = np.ascontiguousarray(rng.normal(size=(14, 7)))
    layer = sg.APPNPPropagationLayer(7, alpha, final_layer=True)
    got = layer.call(propagated, gcn_gen.features, gcn_gen.Aadj, out_indices)
    want = ref.appnp_propagation_layer_call(
        propagated, gcn_gen.features, gcn_gen.Aadj, alpha, out_indices
    )
    assert got.shape == (3, 7)
    assert np.abs(got - want).max() <= TOL
    # the gather is by node id, so the repeated id repeats the row, and the
    # unsorted ids do not disturb each other
    assert np.abs(got[0] - got[2]).max() == 0.0


def test_appnp_propagate_zero_steps_is_the_identity(gcn_gen):
    model = sg.APPNP([5], ["relu"], gcn_gen)
    got = model.propagate(gcn_gen.features, gcn_gen.Aadj, 0)
    assert np.abs(got - gcn_gen.features).max() == 0.0


def test_appnp_propagate_without_teleport_is_a_matrix_power(gcn_gen):
    # alpha = 0 kills the teleport term, so k steps of APPNP are exactly A^k X
    model = sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=0.0)
    a = gcn_gen.Aadj
    power = np.eye(a.shape[0])
    for k in range(6):
        got = model.propagate(gcn_gen.features, a, k)
        assert np.abs(got - power @ gcn_gen.features).max() <= TOL
        power = power @ a


def test_appnp_propagate_counts_steps(gcn_gen):
    # k propagations are k applications of the one-step map, checked against
    # the same map applied in Python
    alpha = 0.3
    model = sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=alpha)
    a = gcn_gen.Aadj
    x = gcn_gen.features
    z = x.copy()
    for k in range(6):
        got = model.propagate(x, a, k)
        assert np.abs(got - z).max() <= TOL
        z = (1.0 - alpha) * (a @ z) + alpha * x


# ------------------------------------------------- closed form of the PPR matrix
def test_ppr_matrix_fixes_the_sqrt_degree_vector(ppnp_gen):
    # `PPNP_Aadj_feats_op` normalizes `A + I`, so with d the degree of `A + I`
    # (the degree of `A` plus the self loop) and S = D^-1/2 (A + I) D^-1/2,
    #   S sqrt(d) = D^-1/2 (A + I) D^-1/2 sqrt(d) = D^-1/2 (A + I) 1 = sqrt(d)
    # so (I - (1 - alpha) S) sqrt(d) = alpha sqrt(d) and therefore
    # P sqrt(d) = alpha (I - (1 - alpha) S)^-1 sqrt(d) = sqrt(d).
    adj, _ = _random_graph()
    root_degree = _sqrt_degree(adj)
    got = ppnp_gen.Aadj @ root_degree
    assert np.abs(got - root_degree).max() <= TOL_INV


def test_ppr_matrix_satisfies_its_defining_equation(ppnp_gen):
    # P (I - (1 - alpha) S) = alpha I, with S built from its definition here
    alpha = 0.1
    adj, _ = _random_graph()
    s = _sym_norm(ref._add_self_loops(adj))
    lhs = ppnp_gen.Aadj @ (np.eye(adj.shape[0]) - (1.0 - alpha) * s)
    assert np.abs(lhs - alpha * np.eye(adj.shape[0])).max() <= TOL_INV


def test_ppr_matrix_matches_the_oracle(ppnp_gen):
    adj, _ = _random_graph()
    want = ref.ppnp_aadj_feats_op(adj, 0.1)
    assert ppnp_gen.Aadj.shape == (14, 14)
    assert np.abs(ppnp_gen.Aadj - want).max() <= TOL_INV


def test_ppr_matrix_rows_sum_to_one_on_a_regular_graph():
    # A 2-regular graph makes S doubly stochastic, so S 1 = 1 and therefore
    # P 1 = alpha/(1 - (1 - alpha)) 1 = 1: the personalized-PageRank matrix is
    # row-stochastic and `Aadj @ ones == ones`.
    adj, features = _cycle_graph(12)
    gen = sg.FullBatchNodeGenerator(
        _Graph(adj, features), method="ppnp", sparse=False
    )
    ones = np.ones(12)
    assert np.abs(gen.Aadj @ ones - ones).max() <= TOL_INV
    assert np.abs(gen.Aadj.sum(axis=1) - ones).max() <= TOL_INV

def test_gcn_adjacency_row_sums_closed_form(gcn_gen):
    # S is doubly stochastic only on a regular graph. In general, with
    # B = A + I and d the row sums of B,
    #   (S 1)_i = sum_j B_ij / sqrt(d_i d_j)
    #           = (1 / sqrt(d_i)) * sum_{j : B_ij = 1} 1 / sqrt(d_j),
    # which is 1 exactly when every closed neighbourhood has the same degree
    # sum. The column sums agree because S is symmetric. The regular case is
    # pinned by test_ppr_matrix_rows_sum_to_one_on_a_regular_graph.
    adj, _ = _random_graph()
    b = ref._add_self_loops(adj)
    d = b.sum(axis=1)
    want = np.array(
        [b[i] @ (1.0 / np.sqrt(d)) / np.sqrt(d[i]) for i in range(14)]
    )
    ones = np.ones(14)
    assert np.abs(gcn_gen.Aadj @ ones - want).max() <= TOL
    assert np.abs(gcn_gen.Aadj.sum(axis=0) - want).max() <= TOL



def test_ppnp_generator_rejects_the_sparse_default():
    adj, features = _random_graph()
    with pytest.raises(ValueError):
        sg.FullBatchNodeGenerator(_Graph(adj, features), method="ppnp")


# ------------------------------------------------ PPNP / APPNP in the large-k limit
def test_appnp_converges_to_the_ppnp_matrix(gcn_gen, ppnp_gen):
    # (1 - alpha) S has spectral radius <= 1 - alpha < 1, so
    # ((1 - alpha) S)^k -> 0 and the k-step APPNP propagation
    #   Z_k = ((1 - alpha) S)^k X + alpha sum_{i<k} ((1 - alpha) S)^i X
    # tends to alpha (I - (1 - alpha) S)^-1 X = P X, which is exactly the
    # matrix PPNP's generator precomputes and applies in one propagation. The
    # two models therefore agree in the limit, for the same base features.
    alpha = 0.1
    model = sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=alpha)
    got = model.propagate(gcn_gen.features, gcn_gen.Aadj, 3000)
    assert np.abs(got - ppnp_gen.Aadj @ gcn_gen.features).max() <= TOL_INV


def test_appnp_limit_is_the_hand_built_ppr_matrix(gcn_gen):
    # the same limit, against alpha (I - (1 - alpha) S)^-1 written out here
    alpha = 0.1
    model = sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=alpha)
    got = model.propagate(gcn_gen.features, gcn_gen.Aadj, 3000)
    want = alpha * np.linalg.inv(np.eye(14) - (1.0 - alpha) * gcn_gen.Aadj)
    assert np.abs(got - want @ gcn_gen.features).max() <= TOL_INV


def test_appnp_error_shrinks_with_k(gcn_gen):
    alpha = 0.1
    model = sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=alpha)
    target = alpha * np.linalg.inv(np.eye(14) - (1.0 - alpha) * gcn_gen.Aadj)
    target = target @ gcn_gen.features
    err_20 = np.abs(model.propagate(gcn_gen.features, gcn_gen.Aadj, 20) - target).max()
    err_400 = np.abs(
        model.propagate(gcn_gen.features, gcn_gen.Aadj, 400) - target
    ).max()
    assert err_400 < err_20
    assert err_400 <= TOL_INV


def test_appnp_limit_fixes_the_sqrt_degree_vector(gcn_gen):
    # the limit matrix has the same eigenvector property as PPNP's, so this
    # checks the k -> infinity propagation against the generator's matrix
    # without forming the inverse
    model = sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=0.1)
    adj, _ = _random_graph()
    root_degree = _sqrt_degree(adj)
    got = model.propagate(root_degree[:, None], gcn_gen.Aadj, 3000)
    assert np.abs(got[:, 0] - root_degree).max() <= TOL_INV


# --------------------------------------------------------------------- the models
def _layers_of(model):
    """The port's Dense stack as `(kernel, bias, act)` triples, which is the
    shape the oracle's `_dense_stack` consumes."""
    return [(l["kernel"], l["bias"], l["act"]) for l in model._layers]


def test_appnp_model_is_dense_layers_then_propagation(gcn_gen):
    model = sg.APPNP([5, 3], ["relu", "tanh"], gcn_gen, approx_iter=2).build(7)
    got = model(gcn_gen.features, gcn_gen.Aadj)
    want = ref.appnp_model_call(
        gcn_gen.features, gcn_gen.Aadj, _layers_of(model), 0.1, 2
    )
    assert got.shape == (14, 3)
    assert np.abs(got - want).max() <= TOL


def test_appnp_model_runs_approx_iter_propagation_layers(gcn_gen):
    """Upstream's `__init__` appends `approx_iter` propagation layers and
    defaults that to ten, so the default model is a ten-step power iteration
    and not a single one."""
    model = sg.APPNP([5], ["relu"], gcn_gen).build(7)
    assert model.approx_iter == 10
    got = model(gcn_gen.features, gcn_gen.Aadj)
    want = ref.appnp_model_call(
        gcn_gen.features, gcn_gen.Aadj, _layers_of(model), 0.1, 10
    )
    assert np.abs(got - want).max() <= TOL


def test_appnp_model_gathers_out_indices_on_the_final_layer(gcn_gen):
    """Upstream builds the last of its `approx_iter` propagation layers with
    `final_layer=True`, so `__call__` returns one row per output index."""
    model = sg.APPNP([5], ["relu"], gcn_gen, approx_iter=3).build(7)
    out_indices = np.array([4, 0, 11], dtype=np.int32)
    got = model(gcn_gen.features, gcn_gen.Aadj, out_indices)
    want = ref.appnp_model_call(
        gcn_gen.features, gcn_gen.Aadj, _layers_of(model), 0.1, 3, out_indices
    )
    assert got.shape == (3, 5)
    assert np.abs(got - want).max() <= TOL


def test_appnp_model_rejects_its_upstream_preconditions(gcn_gen):
    with pytest.raises(ValueError):
        sg.APPNP([5, 3], ["relu"], gcn_gen)
    with pytest.raises(ValueError):
        sg.APPNP([5], ["relu"], gcn_gen, approx_iter=0)
    with pytest.raises(ValueError):
        sg.APPNP([5], ["relu"], gcn_gen, approx_iter=1.5)
    with pytest.raises(ValueError):
        sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=7.0)
    with pytest.raises(ValueError):
        sg.APPNP([5], ["relu"], gcn_gen, teleport_probability=-0.5)
    with pytest.raises(TypeError):
        sg.APPNP([5], ["relu"], object())


def test_ppnp_model_rejects_mismatched_layer_and_activation_counts(ppnp_gen):
    with pytest.raises(ValueError):
        sg.PPNP([5, 3], ["relu"], ppnp_gen)


def test_ppnp_model_shapes(ppnp_gen):
    model = sg.PPNP([5, 3], ["relu", "tanh"], ppnp_gen).build(7)
    out_indices = np.array([4, 11, 0], dtype=np.int32)
    assert model(ppnp_gen.features, ppnp_gen.Aadj).shape == (14, 3)
    assert model(ppnp_gen.features, ppnp_gen.Aadj, out_indices).shape == (3, 3)


def test_ppnp_model_is_dense_layers_then_propagation(ppnp_gen):
    # the reference dense steps, using the oracle's own activation, followed by
    # the oracle's propagation call
    model = sg.PPNP([5, 3], ["relu", "tanh"], ppnp_gen).build(7)
    h = ppnp_gen.features
    for layer in model._layers[:-1]:
        h = h @ layer["kernel"]
        if layer["bias"] is not None:
            h = h + layer["bias"]
        h = ref._activation(h, layer["act"], 0.01)
    want = ref.ppnp_propagation_layer_call(h, ppnp_gen.Aadj)
    got = model(ppnp_gen.features, ppnp_gen.Aadj)
    assert np.abs(got - want).max() <= TOL


def test_ppnp_model_gathers_out_indices(ppnp_gen):
    model = sg.PPNP([5, 3], ["relu", "tanh"], ppnp_gen).build(7)
    out_indices = np.array([7, 7, 1], dtype=np.int32)
    h = ppnp_gen.features
    for layer in model._layers[:-1]:
        h = h @ layer["kernel"] + layer["bias"]
        h = ref._activation(h, layer["act"], 0.01)
    want = ref.ppnp_propagation_layer_call(h, ppnp_gen.Aadj, out_indices)
    got = model(ppnp_gen.features, ppnp_gen.Aadj, out_indices)
    assert np.abs(got - want).max() <= TOL
    assert np.abs(got[0] - got[1]).max() == 0.0
