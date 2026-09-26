"""Parity tests for `sg.GraphConvolution` and `sg.GCN`, the port of
`stellargraph/layer/gcn.py` (v0.8.1).

Upstream `stellargraph` cannot be installed here (it pins Python < 3.9 and
TensorFlow 2.1), so `tests/upstream_reference.py` -- a line-by-line NumPy
transliteration of the upstream sources -- IS the oracle for these tests. Each
`GraphConvolution` is given the same kernel and bias the oracle is given, and
the two must agree numerically; the closed-form tests below pin the definition
itself (an affine function of `A @ features`, a normalised-adjacency weighted
mean, a per-column bias) without consulting the oracle at all.

Tolerances: `ATOL` is 1e-12, the kernel and NumPy summing the same products in
the same order. `RTOL_EXP`/`ATOL_EXP` cover the paths that go through `exp`
(the Mojo `exp` is accurate to ~1e-11 relative) and `SOFTPLUS_RTOL` the one
activation, `softplus`, where the kernel's `log(1 + exp(x))` and the oracle's
`log1p(exp(x))` disagree at the 1e-9 level.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

ATOL = 1e-12
ATOL_EXP = 1e-12
RTOL_EXP = 1e-9
# The kernel computes `log(1 + exp(x))` where the oracle computes the more
# accurate `log1p(exp(x))`; the two agree only to ~1e-9 relative.
SOFTPLUS_RTOL = 1e-8

# A deliberately unsorted, duplicate-free index list: `K.gather` is a pure row
# selection, so the output must be those rows in exactly that order.
OUT_INDICES = np.array([17, 2, 23, 0, 9], dtype=np.int32)


class _Graph:
    """The three attributes `FullBatchNodeGenerator` reads off a `StellarGraph`."""

    def __init__(self, adj, features):
        self.node_list = np.arange(adj.shape[0])
        self.edges = np.argwhere(adj > 0.0)
        self.features = features


def _generator(adj, features, method="gcn"):
    return sg.FullBatchNodeGenerator(_Graph(adj, features), method=method)


def _conv(units, activation=None, use_bias=True, final_layer=False, f=7, seed=0):
    """A built `GraphConvolution` with non-degenerate weights."""
    layer = sg.GraphConvolution(
        units, activation=activation, use_bias=use_bias, final_layer=final_layer
    )
    layer.build(f)
    rng = np.random.default_rng(seed)
    layer.kernel[:] = 0.7 * rng.normal(size=(f, units))
    if layer.bias is not None:
        layer.bias[:] = rng.normal(size=units)
    return layer


def _stack_reference(model, features, adj, out_indices):
    """The reference `GCN`: one `graph_convolution_call` per layer, the last
    one gathering `out_indices` because it carries `final_layer`."""
    h = features
    last = len(model._layers) - 1
    for i, layer in enumerate(model._layers):
        h = ref.graph_convolution_call(
            h,
            adj,
            layer.kernel,
            layer.bias if layer.use_bias else None,
            layer.activation,
            out_indices if i == last else None,
        )
    return h


# --------------------------------------------------------------- GraphConvolution
def test_convolution_matches_reference(adj_with_loops, features):
    layer = _conv(5, seed=1)
    want = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, None
    )
    assert layer(features, adj_with_loops).shape == (24, 5)
    assert np.abs(layer(features, adj_with_loops) - want).max() <= ATOL


def test_convolution_is_affine_in_a_times_features(adj_with_loops, features):
    """`K.dot(K.dot(A, features), kernel) + bias`, with no oracle involved."""
    layer = _conv(5, seed=2)
    h_graph = adj_with_loops @ features
    want = h_graph @ layer.kernel + layer.bias
    np.testing.assert_allclose(layer(features, adj_with_loops), want, rtol=0, atol=ATOL)


def test_convolution_identity_kernel_is_the_normalised_neighbour_mean(
    adj_with_loops, features
):
    """With `kernel = I` the layer is the row-stochastic weighting `A @ X`, i.e.
    the mean over each node's closed neighbourhood."""
    f = features.shape[1]
    A = sg.normalize_adj(adj_with_loops, symmetric=False)  # row-stochastic
    np.testing.assert_allclose(A.sum(axis=1), np.ones(24), rtol=0, atol=1e-12)
    layer = _conv(f, activation=None, use_bias=False, f=f, seed=3)
    layer.kernel[:] = np.eye(f)
    got = layer(features, A)
    np.testing.assert_allclose(got, A @ features, rtol=0, atol=ATOL)
    # A closed-form value, not a comparison: every output element is a convex
    # combination of the feature column, so it lies inside that column's range.
    assert (got >= features.min(axis=0) - ATOL).all()
    assert (got <= features.max(axis=0) + ATOL).all()


def test_convolution_bias_is_added_once_per_column(adj_with_loops, features):
    layer = _conv(6, activation=None, seed=4)
    pre = (adj_with_loops @ features) @ layer.kernel
    got = layer(features, adj_with_loops)
    offset = got - pre
    # A bias is a per-output-column constant: it does not vary down a column.
    assert np.abs(offset - offset[0]).max() <= ATOL
    np.testing.assert_allclose(offset[0], layer.bias, rtol=0, atol=ATOL)


def test_convolution_without_bias_leaves_bias_as_none(adj_with_loops, features):
    layer = _conv(5, use_bias=False, seed=5)
    assert layer.bias is None
    got = layer(features, adj_with_loops)
    want = ref.graph_convolution_call(features, adj_with_loops, layer.kernel, None, None)
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)
    np.testing.assert_allclose(
        got, (adj_with_loops @ features) @ layer.kernel, rtol=0, atol=ATOL
    )


def test_convolution_relu_is_non_negative_and_matches_reference(
    adj_with_loops, features
):
    layer = _conv(5, activation="relu", seed=6)
    got = layer(features, adj_with_loops)
    assert got.min() >= 0.0
    pre = (adj_with_loops @ features) @ layer.kernel + layer.bias
    assert (pre < 0.0).any(), "the relu clip must actually be exercised"
    np.testing.assert_allclose(got, np.maximum(pre, 0.0), rtol=0, atol=ATOL)
    want = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, "relu"
    )
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "activation", [None, "linear", "relu", "elu", "softmax", "sigmoid", "tanh",
                   "softplus", "hard_sigmoid", "exponential"]
)
def test_convolution_activation_matches_reference(adj_with_loops, features, activation):
    layer = _conv(5, activation=activation, seed=7)
    got = layer(features, adj_with_loops)
    want = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, activation
    )
    rtol = SOFTPLUS_RTOL if activation == "softplus" else RTOL_EXP
    np.testing.assert_allclose(got, want, rtol=rtol, atol=ATOL_EXP)


def test_convolution_softmax_rows_sum_to_one(adj_with_loops, features):
    layer = _conv(5, activation="softmax", seed=8)
    got = layer(features, adj_with_loops)
    np.testing.assert_allclose(got.sum(axis=1), np.ones(24), rtol=0, atol=ATOL)
    assert got.min() >= 0.0


def test_convolution_final_layer_gathers_out_indices(adj_with_loops, features):
    layer = _conv(5, activation="relu", final_layer=True, seed=9)
    got = layer(features, adj_with_loops, OUT_INDICES)
    full = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, "relu"
    )
    assert got.shape == (5, 5)
    # A gather is data movement, so the rows come back bit for bit.
    np.testing.assert_array_equal(got, full[OUT_INDICES])
    np.testing.assert_array_equal(got[0], full[17])


def test_convolution_final_layer_without_indices_keeps_every_node(
    adj_with_loops, features
):
    layer = _conv(5, final_layer=True, seed=10)
    got = layer(features, adj_with_loops)
    # `out_indices=None` means no gather, so a final layer keeps every node
    assert got.shape == (adj_with_loops.shape[0], 5)
    full = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, None
    )
    np.testing.assert_allclose(got, full, rtol=0, atol=0.0)


def test_convolution_gather_repeats_a_row_for_a_repeated_index(
    adj_with_loops, features
):
    """`K.gather` is a pure row selection, so a repeated index repeats the row
    and the index list's order is the output's order."""
    repeated = np.array([3, 3, 0, 23, 3], dtype=np.int32)
    layer = _conv(5, activation="relu", final_layer=True, seed=22)
    got = layer(features, adj_with_loops, repeated)
    full = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, "relu"
    )
    assert got.shape == (5, 5)
    np.testing.assert_array_equal(got, full[repeated])


@pytest.mark.parametrize("m", [1, 25, 60])
def test_convolution_gather_length_is_unconstrained_by_the_node_count(
    adj_with_loops, features, m
):
    """`K.gather` has no bound on the output length, and upstream's
    `out_indices_t` is `Input(batch_shape=(1, None))`, so `m` may exceed `n`:
    a generator may repeat or overshoot the node set. Every output index must
    still produce exactly one row, taken from the full output."""
    n = features.shape[0]
    indices = (np.arange(m, dtype=np.int64) * 7) % n
    layer = _conv(5, activation="relu", final_layer=True, seed=22)
    got = layer(features, adj_with_loops, indices.astype(np.int32))
    full = ref.graph_convolution_call(
        features, adj_with_loops, layer.kernel, layer.bias, "relu"
    )
    assert got.shape == (m, 5)
    np.testing.assert_array_equal(got, full[indices])


def test_convolution_ignores_indices_when_not_the_final_layer(
    adj_with_loops, features
):
    layer = _conv(5, final_layer=False, seed=11)
    got = layer(features, adj_with_loops, OUT_INDICES)
    assert got.shape == (24, 5)
    np.testing.assert_allclose(
        got,
        ref.graph_convolution_call(features, adj_with_loops, layer.kernel, layer.bias, None),
        rtol=0,
        atol=ATOL,
    )


def test_convolution_build_allocates_kernel_and_bias():
    layer = sg.GraphConvolution(4, use_bias=True)
    assert layer.kernel is None and layer.bias is None
    assert layer.build(6) is layer
    assert layer.kernel.shape == (6, 4)
    assert layer.bias.shape == (4,)
    assert not layer.kernel.any() and not layer.bias.any()
    nobias = sg.GraphConvolution(4, use_bias=False).build(6)
    assert nobias.kernel.shape == (6, 4)
    assert nobias.bias is None


def test_convolution_rejects_a_kernel_built_for_another_input_dim(
    adj_with_loops, features
):
    layer = _conv(5, seed=12)
    with pytest.raises(ValueError):
        layer(features[:, :5], adj_with_loops)


def test_convolution_rejects_an_unsupported_activation():
    with pytest.raises(ValueError):
        sg.GraphConvolution(3, activation="not_an_activation")


# --------------------------------------------------------------------------- GCN
def _gcn(layer_sizes, generator, f=7, seed=0, **kwargs):
    model = sg.GCN(layer_sizes, generator, **kwargs)
    model.build(f, seed=seed)
    return model


def test_gcn_two_layers_match_the_reference_stack(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([5, 3], gen)
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    want = _stack_reference(model, gen.features, gen.Aadj, OUT_INDICES)
    assert got.shape == (5, 3)
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


def test_gcn_default_out_indices_keeps_every_node(
    adj_with_loops, features
):
    """Upstream `GCN` always takes an `out_indices` input; the port's default is
    `None`, which means no gather, so the model output keeps every node."""
    gen = _generator(adj_with_loops, features)
    model = _gcn([4, 2], gen)
    got = model(gen.features, gen.Aadj)
    assert got.shape == (adj_with_loops.shape[0], 2)
    full = _stack_reference(model, gen.features, gen.Aadj, None)
    np.testing.assert_allclose(got, full, rtol=0, atol=ATOL)


def test_gcn_gathers_out_indices_on_the_last_layer(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([5, 3], gen)
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    full = _stack_reference(model, gen.features, gen.Aadj, None)
    np.testing.assert_allclose(got, full[OUT_INDICES], rtol=0, atol=ATOL)
    # Only the last layer gathers: an intermediate layer of the same stack,
    # given the same indices, keeps every row.
    assert model._layers[0].final_layer is False
    assert model._layers[-1].final_layer is True


def test_gcn_three_layers_match_the_reference_stack(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([6, 4, 2], gen, seed=3)
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    want = _stack_reference(model, gen.features, gen.Aadj, OUT_INDICES)
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


def test_gcn_build_chains_kernel_shapes_and_respects_glorot_uniform(
    adj_with_loops, features
):
    gen = _generator(adj_with_loops, features)
    model = _gcn([5, 3], gen, seed=1)
    assert [k.shape for k in model.kernels] == [(7, 5), (5, 3)]
    for kernel, (fan_in, fan_out) in zip(model.kernels, [(7, 5), (5, 3)]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        assert np.abs(kernel).max() <= bound
        assert np.abs(kernel).max() > 0.5 * bound, "glorot_uniform must fill the range"
        assert np.abs(kernel).mean() < 0.5 * bound
    assert all(layer.bias is not None and not layer.bias.any() for layer in model._layers)


def test_gcn_build_is_deterministic_in_the_seed(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    a = _gcn([4, 2], gen, seed=7)
    b = _gcn([4, 2], gen, seed=7)
    c = _gcn([4, 2], gen, seed=8)
    for ka, kb in zip(a.kernels, b.kernels):
        assert (ka == kb).all()
    assert any((ka != kc).any() for ka, kc in zip(a.kernels, c.kernels))
    np.testing.assert_allclose(
        a(gen.features, gen.Aadj, OUT_INDICES), b(gen.features, gen.Aadj, OUT_INDICES),
        rtol=0, atol=0.0,
    )


def test_gcn_without_bias_matches_the_reference(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([4], gen, bias=False, seed=2)
    assert all(layer.bias is None for layer in model._layers)
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    want = ref.graph_convolution_call(
        gen.features, gen.Aadj, model._layers[0].kernel, None, "relu", OUT_INDICES
    )
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


def test_gcn_dropout_is_the_identity_at_inference(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    plain = _gcn([5, 3], gen, seed=4)
    dropped = _gcn([5, 3], gen, seed=4, dropout=0.75)
    assert dropped.dropout == 0.75
    np.testing.assert_allclose(
        dropped(gen.features, gen.Aadj, OUT_INDICES),
        plain(gen.features, gen.Aadj, OUT_INDICES),
        rtol=0,
        atol=0.0,
    )


def test_gcn_method_none_preprocesses_the_adjacency(sym_graph, features):
    """`method='none'` makes the generator leave `Aadj` raw, so the model
    inserts the `GraphPreProcessingLayer` the oracle reproduces."""
    gen = _generator(sym_graph, features, method="none")
    np.testing.assert_allclose(gen.Aadj, sym_graph, rtol=0, atol=0.0)
    model = _gcn([5, 2], gen, seed=5)
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    A = ref.graph_pre_processing_layer(sym_graph)
    np.testing.assert_allclose(A, sg.GraphPreProcessingLayer(24)(sym_graph), rtol=0, atol=ATOL)
    # A symmetric normalisation: equal in both directions, with a unit diagonal
    # contribution of exactly 1/deg(i) from the self loop.
    np.testing.assert_allclose(A, A.T, rtol=0, atol=0.0)
    deg = (sym_graph + np.eye(24)).sum(axis=1)
    np.testing.assert_allclose(np.diag(A), 1.0 / np.sqrt(deg * deg), rtol=0, atol=ATOL)
    want = _stack_reference(model, gen.features, A, OUT_INDICES)
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


def test_gcn_softmax_final_layer_rows_sum_to_one(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([7], gen, activations=["softmax"], seed=6)
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    np.testing.assert_allclose(got.sum(axis=1), np.ones(5), rtol=0, atol=ATOL)
    assert got.min() >= 0.0


def test_gcn_activations_are_one_per_layer(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([4, 2], gen, activations=["tanh", "elu"])
    assert model.activations == ["tanh", "elu"]
    got = model(gen.features, gen.Aadj, OUT_INDICES)
    want = _stack_reference(model, gen.features, gen.Aadj, OUT_INDICES)
    np.testing.assert_allclose(got, want, rtol=RTOL_EXP, atol=ATOL_EXP)
    with pytest.raises(ValueError):
        sg.GCN([4, 2], gen, activations=["relu"])
    with pytest.raises(ValueError):
        sg.GCN([4, 2], gen, activations=["relu", "not_an_activation"])


def test_gcn_reports_the_generator_input_shape(adj_with_loops, features):
    gen = _generator(adj_with_loops, features)
    model = _gcn([4, 2], gen)
    assert model.node_model() == (((24, 7),), None)
    assert model.generator is gen
    assert model.method == "gcn"


def test_softsign_activation_matches_its_closed_form():
    """`softsign(x) = x / (1 + |x|)`. It is the one activation the parity
    sweep over the documented names does not reach, because upstream's
    `keras.activations.get` list is the source of truth and it is listed
    there."""
    r = np.random.default_rng(31)
    x = r.normal(size=(6, 5)) * 4.0
    expected = x / (1.0 + np.abs(x))
    conv = sg.GraphConvolution(5, activation="softsign", use_bias=False)
    conv.build(5)
    conv.kernel[:] = np.eye(5)
    np.testing.assert_allclose(conv(x, np.eye(6)), expected, atol=ATOL, rtol=0.0)
    # the asymmetric half of the closed form, which the table above cannot see
    wide = np.array([[-1e6, 1e6]], dtype=np.float64)
    conv2 = sg.GraphConvolution(2, activation="softsign", use_bias=False)
    conv2.build(2)
    conv2.kernel[:] = np.eye(2)
    np.testing.assert_allclose(
        conv2(wide, np.eye(1)), wide / (1.0 + np.abs(wide)), atol=1e-12, rtol=0.0
    )
