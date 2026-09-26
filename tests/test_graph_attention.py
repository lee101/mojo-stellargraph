"""Parity tests for `sg.GraphAttention` and `sg.GraphAttentionSparse`, the port
of `stellargraph/layer/graph_attention.py` (v0.8.1).

Upstream `stellargraph` cannot be installed here (it pins Python < 3.9 and
TensorFlow 2.1), so `tests/upstream_reference.py` -- a line-by-line NumPy
transliteration of the upstream sources -- IS the oracle for these tests.

The closed-form tests below do not consult the oracle at all: they rebuild the
published GAT equations (a LeakyReLU(0.2) score per edge, a masked row softmax,
a weighted mean of the projected features) from the layer's own weights, and
they pin the two structural facts that must hold whatever the weights are --
each attention row is a probability distribution over that node's edges, and
the `-10e9 * (1 - A)` mask makes it vanish off the adjacency.

Tolerance: `ATOL` is 1e-9, because every dense path here runs through `exp`
(the attention softmax, and the output activation when there is one) and the
Mojo `exp` is accurate to about 1e-11 relative. The `softplus` case adds
`rtol=1e-8`, the kernel computing `log(1 + exp(x))` where the oracle computes
the more accurate `log1p(exp(x))`.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref


ATOL = 1e-9

# A deliberately unsorted, duplicate-free index list: `K.gather` is a pure row
# selection, so the output must be those rows in exactly that order.
OUT_INDICES = np.array([17, 2, 23, 0, 9], dtype=np.int32)


# ------------------------------------------------------- the GAT equations, restated
def _leaky_relu(x, alpha=0.2):
    return np.where(x >= 0.0, x, alpha * x)


def _row_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _scores(X, kernel, attn_kernel, head):
    """`features`, and the per-node self / neighbour attention scores."""
    features = X @ kernel[head]
    return (
        features,
        features @ attn_kernel[head][0],
        features @ attn_kernel[head][1],
    )


def _attention(X, A, kernel, attn_kernel, head):
    """The dense `[n, n]` attention matrix of Eq. 3 for one head."""
    features, attn_self, attn_neighs = _scores(X, kernel, attn_kernel, head)
    dense = _leaky_relu(attn_self[:, None] + attn_neighs[None, :], 0.2)
    return _row_softmax(dense - 1e10 * (1.0 - A)), features


def _layer(cls, f, units, heads, reduction="concat", activation="relu",
           use_bias=True, final_layer=False, saliency_map_support=False,
           seed=0, scale=0.7):
    layer = cls(
        units,
        attn_heads=heads,
        attn_heads_reduction=reduction,
        activation=activation,
        use_bias=use_bias,
        final_layer=final_layer,
        saliency_map_support=saliency_map_support,
    )
    layer.build(f)
    rng = np.random.default_rng(seed)
    layer.kernels[:] = scale * rng.normal(size=(heads, f, units))
    # The port stores a head's two `(units, 1)` kernels flattened; the oracle
    # takes them as `[heads, 2, units]`.
    layer.attn_kernels[:] = (scale * rng.normal(size=(heads, 2, units))).reshape(
        heads, 2 * units
    )
    if use_bias:
        layer.biases[:] = rng.normal(size=(heads, units))
    return layer


# ----------------------------------------------------------------- GraphAttention
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("reduction", ["concat", "average"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_dense_matches_reference(adj_with_loops, features, heads, reduction, use_bias):
    layer = _layer(sg.GraphAttention, 7, 4, heads, reduction, "relu", use_bias, seed=heads)
    ak = layer.attn_kernels.reshape(heads, 2, 4)
    bias = layer.biases if use_bias else None
    got = layer(features, adj_with_loops)
    want = ref.graph_attention_call(
        features, adj_with_loops, layer.kernels, ak, bias, heads, reduction, "relu",
        False, None, False, 1.0, 0.0,
    )
    assert got.shape == (24, 4 * heads if reduction == "concat" else 4)
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    "activation", [None, "linear", "relu", "elu", "tanh", "sigmoid", "softplus",
                   "hard_sigmoid", "exponential", "softmax"]
)
def test_dense_activation_matches_reference(adj_with_loops, features, activation):
    layer = _layer(sg.GraphAttention, 7, 4, 2, "concat", activation, seed=3)
    ak = layer.attn_kernels.reshape(2, 2, 4)
    got = layer(features, adj_with_loops)
    want = ref.graph_attention_call(
        features, adj_with_loops, layer.kernels, ak, layer.biases, 2, "concat",
        activation, False, None, False, 1.0, 0.0,
    )
    # The kernel's `log(1 + exp(x))` against the oracle's `log1p(exp(x))`.
    rtol = 1e-8 if activation == "softplus" else 0.0
    np.testing.assert_allclose(got, want, rtol=rtol, atol=ATOL)


def test_dense_is_the_attention_weighted_mean(adj_with_loops, features):
    """`output[head] = attention[head] @ (X @ kernel[head]) + bias[head]`."""
    layer = _layer(sg.GraphAttention, 7, 4, 1, "concat", None, True, seed=4)
    attn, projected = _attention(features, adj_with_loops, layer.kernels, layer.attn_kernels.reshape(1, 2, 4), 0)
    want = attn @ projected + layer.biases[0]
    np.testing.assert_allclose(layer(features, adj_with_loops), want, rtol=0, atol=ATOL)


def test_dense_attention_rows_are_a_distribution(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 2, "concat", None, False, seed=5)
    for head in range(2):
        attn, _ = _attention(
            features, adj_with_loops, layer.kernels, layer.attn_kernels.reshape(2, 2, 4), head
        )
        np.testing.assert_allclose(attn.sum(axis=1), np.ones(24), rtol=0, atol=1e-12)
        assert attn.min() >= 0.0
        assert attn.max() <= 1.0


def test_dense_attention_vanishes_off_the_adjacency(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 1, "concat", None, False, seed=6)
    attn, _ = _attention(features, adj_with_loops, layer.kernels, layer.attn_kernels.reshape(1, 2, 4), 0)
    # The `-10e9 * (1 - A)` mask: exactly zero weight on a non-edge, positive
    # weight on every edge of the closed neighbourhood.
    assert (attn[adj_with_loops == 0.0] == 0.0).all()
    assert (attn[adj_with_loops > 0.0] > 0.0).all()
    assert (np.diag(attn) > 0.0).all(), "the self loop keeps a node attending to itself"


def test_dense_concat_lays_heads_out_side_by_side(adj_with_loops, features):
    units, heads = 4, 3
    layer = _layer(sg.GraphAttention, 7, units, heads, "concat", None, True, seed=7)
    got = layer(features, adj_with_loops)
    for head in range(heads):
        attn, projected = _attention(
            features, adj_with_loops, layer.kernels, layer.attn_kernels.reshape(heads, 2, units), head
        )
        want = attn @ projected + layer.biases[head]
        np.testing.assert_allclose(
            got[:, head * units:(head + 1) * units], want, rtol=0, atol=ATOL
        )


def test_dense_average_reduction_is_the_mean_of_the_heads(adj_with_loops, features):
    units, heads = 4, 3
    layer = _layer(sg.GraphAttention, 7, units, heads, "average", None, True, seed=8)
    got = layer(features, adj_with_loops)
    per_head = []
    for head in range(heads):
        attn, projected = _attention(
            features, adj_with_loops, layer.kernels, layer.attn_kernels.reshape(heads, 2, units), head
        )
        per_head.append(attn @ projected + layer.biases[head])
    np.testing.assert_allclose(got, np.mean(per_head, axis=0), rtol=0, atol=ATOL)
    # A mean over heads is a convex combination, so every column stays inside
    # the range the per-head outputs span.
    assert got.min() >= np.min(per_head) - ATOL
    assert got.max() <= np.max(per_head) + ATOL


def test_dense_without_bias_leaves_biases_as_none(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 2, "concat", "relu", False, seed=9)
    assert layer.biases is None
    ak = layer.attn_kernels.reshape(2, 2, 4)
    np.testing.assert_allclose(
        layer(features, adj_with_loops),
        ref.graph_attention_call(features, adj_with_loops, layer.kernels, ak, None, 2,
                                 "concat", "relu", False, None, False, 1.0, 0.0),
        rtol=0,
        atol=ATOL,
    )


def test_dense_relu_output_is_non_negative(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 2, "concat", "relu", True, seed=10)
    got = layer(features, adj_with_loops)
    assert got.min() >= 0.0
    pre = _layer(sg.GraphAttention, 7, 4, 2, "concat", None, True, seed=10)(features, adj_with_loops)
    assert (pre < 0.0).any(), "the relu clip must actually be exercised"
    np.testing.assert_allclose(got, np.maximum(pre, 0.0), rtol=0, atol=ATOL)


def test_dense_final_layer_gathers_out_indices(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 2, "concat", "relu", True, final_layer=True, seed=11)
    ak = layer.attn_kernels.reshape(2, 2, 4)
    got = layer(features, adj_with_loops, OUT_INDICES)
    full = ref.graph_attention_call(
        features, adj_with_loops, layer.kernels, ak, layer.biases, 2, "concat", "relu",
        False, None, False, 1.0, 0.0,
    )
    assert got.shape == (5, 8)
    np.testing.assert_allclose(got, full[OUT_INDICES], rtol=0, atol=ATOL)


def test_dense_final_layer_without_indices_keeps_every_node(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 1, "concat", "relu", True, final_layer=True, seed=12)
    got = layer(features, adj_with_loops)
    # `out_indices=None` means no gather, so a final layer keeps every node
    assert got.shape == (adj_with_loops.shape[0], 4)
    np.testing.assert_allclose(
        got[0], layer(features, adj_with_loops, np.array([0]))[0], rtol=0, atol=0.0
    )


def test_dense_gather_repeats_a_row_for_a_repeated_index(adj_with_loops, features):
    """`K.gather` is a pure row selection, so a repeated index repeats the row
    and the index list's order is the output's order."""
    repeated = np.array([3, 3, 0, 23, 3], dtype=np.int32)
    layer = _layer(sg.GraphAttention, 7, 4, 1, "concat", "relu", True,
                   final_layer=True, seed=22)
    got = layer(features, adj_with_loops, repeated)
    ak = layer.attn_kernels.reshape(1, 2, 4)
    full = ref.graph_attention_call(
        features, adj_with_loops, layer.kernels, ak, layer.biases, 1, "concat", "relu",
        False, None, False, 1.0, 0.0,
    )
    assert got.shape == (5, 4)
    np.testing.assert_allclose(got, full[repeated], rtol=0, atol=ATOL)


def test_dense_ignores_indices_when_not_the_final_layer(adj_with_loops, features):
    layer = _layer(sg.GraphAttention, 7, 4, 1, "concat", "relu", True, final_layer=False, seed=13)
    got = layer(features, adj_with_loops, OUT_INDICES)
    assert got.shape == (24, 4)
    np.testing.assert_allclose(
        got, layer(features, adj_with_loops), rtol=0, atol=0.0
    )


def test_dense_build_allocates_one_kernel_triple_per_head():
    layer = sg.GraphAttention(5, attn_heads=3)
    assert layer.kernels is None and layer.biases is None and layer.attn_kernels is None
    assert layer.output_dim == 15
    assert layer.build(7) is layer
    assert layer.kernels.shape == (3, 7, 5)
    assert layer.attn_kernels.shape == (3, 10)
    assert layer.biases.shape == (3, 5)
    assert sg.GraphAttention(5, attn_heads=3, attn_heads_reduction="average").output_dim == 5
    assert sg.GraphAttention(5, use_bias=False).build(7).biases is None


def test_dense_rejects_an_unknown_heads_reduction():
    with pytest.raises(ValueError):
        sg.GraphAttention(4, attn_heads_reduction="sum")


def test_dense_rejects_an_unsupported_activation():
    with pytest.raises(ValueError):
        sg.GraphAttention(4, activation="not_an_activation")


# ---------------------------------------------------- the saliency_map_support branch
@pytest.mark.parametrize("delta", [1.0, 3.0])
@pytest.mark.parametrize("non_exist_edge", [0.0, 0.1])
def test_saliency_branch_matches_reference(adj_with_loops, features, delta, non_exist_edge):
    layer = _layer(sg.GraphAttention, 7, 4, 2, "concat", "relu", True,
                   saliency_map_support=True, seed=14)
    layer.delta = delta
    layer.non_exist_edge = non_exist_edge
    ak = layer.attn_kernels.reshape(2, 2, 4)
    got = layer(features, adj_with_loops)
    want = ref.graph_attention_call(
        features, adj_with_loops, layer.kernels, ak, layer.biases, 2, "concat", "relu",
        False, None, True, delta, non_exist_edge,
    )
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


def _saliency_attention(X, A, kernel, attn_kernel, head, delta, non_exist_edge):
    """The `W / sum(W)` weighting of the saliency branch, restated."""
    features, attn_self, attn_neighs = _scores(X, kernel, attn_kernel, head)
    dense = _leaky_relu(attn_self[:, None] + attn_neighs[None, :], 0.2)
    shifted = np.exp(dense - dense.max(axis=1, keepdims=True))
    n = X.shape[0]
    w = (
        delta * A * shifted * (1.0 - non_exist_edge)
        + non_exist_edge
        * (A + delta * (np.ones((n, n)) - A) + np.eye(n))
        * shifted
    )
    return w / w.sum(axis=1, keepdims=True), features


def test_saliency_attention_is_the_normalised_weighted_mean(adj_with_loops, features):
    units, heads, delta, non_exist_edge = 4, 1, 2.0, 0.0
    layer = _layer(sg.GraphAttention, 7, units, heads, "concat", None, False,
                   saliency_map_support=True, seed=15)
    layer.delta, layer.non_exist_edge = delta, non_exist_edge
    ak = layer.attn_kernels.reshape(heads, 2, units)
    attn, projected = _saliency_attention(
        features, adj_with_loops, layer.kernels, ak, 0, delta, non_exist_edge
    )
    np.testing.assert_allclose(attn.sum(axis=1), np.ones(24), rtol=0, atol=1e-12)
    assert (attn >= 0.0).all()
    np.testing.assert_allclose(layer(features, adj_with_loops), attn @ projected,
                               rtol=0, atol=ATOL)


def test_saliency_non_exist_edge_puts_weight_on_missing_edges(adj_with_loops, features):
    """The `(1 - non_exist_edge)` term exists so a saliency map still has a
    value on an edge the graph does not have."""
    units, delta = 4, 1.0
    layer = _layer(sg.GraphAttention, 7, units, 1, "concat", None, False,
                   saliency_map_support=True, seed=16)
    ak = layer.attn_kernels.reshape(1, 2, units)
    off = adj_with_loops == 0.0
    layer.delta, layer.non_exist_edge = delta, 0.0
    zeroed, _ = _saliency_attention(features, adj_with_loops, layer.kernels, ak, 0, delta, 0.0)
    layer.non_exist_edge = 0.25
    spread, _ = _saliency_attention(features, adj_with_loops, layer.kernels, ak, 0, delta, 0.25)
    assert (zeroed[off] == 0.0).all()
    assert (spread[off] > 0.0).all()
    np.testing.assert_allclose(spread.sum(axis=1), np.ones(24), rtol=0, atol=1e-12)


# ---------------------------------------------------------- GraphAttentionSparse
def _indices(adj):
    rows, cols = np.nonzero(adj)
    return np.stack([rows, cols], axis=1)


@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("reduction", ["concat", "average"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_sparse_matches_reference(adj_with_loops, features, heads, reduction, use_bias):
    layer = _layer(sg.GraphAttentionSparse, 7, 4, heads, reduction, "relu", use_bias, seed=17)
    ak = layer.attn_kernels.reshape(heads, 2, 4)
    bias = layer.biases if use_bias else None
    got = layer(features, sg.SparseTensor.from_dense(adj_with_loops))
    want = ref.graph_attention_sparse_call(
        features, _indices(adj_with_loops), layer.kernels, ak, bias, heads, reduction,
        "relu", False, None,
    )
    assert got.shape == (24, 4 * heads if reduction == "concat" else 4)
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("reduction", ["concat", "average"])
def test_sparse_equals_dense(adj_with_loops, features, heads, reduction):
    """The two variants are the same function of the same binary adjacency: the
    dense mask keeps exactly the edges the sparse index list holds."""
    dense = _layer(sg.GraphAttention, 7, 4, heads, reduction, "relu", True, seed=18)
    sparse = _layer(sg.GraphAttentionSparse, 7, 4, heads, reduction, "relu", True, seed=18)
    np.testing.assert_allclose(
        dense.kernels, sparse.kernels, rtol=0, atol=0.0
    )
    np.testing.assert_allclose(
        dense(features, adj_with_loops),
        sparse(features, sg.SparseTensor.from_dense(adj_with_loops)),
        rtol=0,
        atol=ATOL,
    )


def test_sparse_is_the_per_row_softmax_weighted_mean(adj_with_loops, features):
    """The closed form of `tf.sparse.softmax` followed by `tf.sparse.matmul`:
    a softmax over each row of the index list, then a weighted mean of the
    neighbour features."""
    units = 4
    layer = _layer(sg.GraphAttentionSparse, 7, units, 1, "concat", None, False, seed=19)
    ak = layer.attn_kernels.reshape(1, 2, units)
    features_ = features @ layer.kernels[0]
    attn_self = features_ @ ak[0][0]
    attn_neighs = features_ @ ak[0][1]
    indices = _indices(adj_with_loops)
    values = _leaky_relu(attn_self[indices[:, 0]] + attn_neighs[indices[:, 1]], 0.2)

    weights = np.zeros_like(values)
    for row in range(24):
        member = indices[:, 0] == row
        assert member.any()
        e = np.exp(values[member] - values[member].max())
        weights[member] = e / e.sum()
    np.testing.assert_allclose(weights.sum(), 24.0, rtol=0, atol=1e-12)

    want = np.zeros_like(features_)
    for edge, (row, col) in enumerate(indices):
        want[row] += weights[edge] * features_[col]
    np.testing.assert_allclose(
        layer(features, sg.SparseTensor.from_dense(adj_with_loops)), want, rtol=0, atol=ATOL
    )


def test_sparse_final_layer_gathers_out_indices(adj_with_loops, features):
    layer = _layer(sg.GraphAttentionSparse, 7, 4, 2, "concat", "relu", True,
                   final_layer=True, seed=20)
    ak = layer.attn_kernels.reshape(2, 2, 4)
    adjacency = sg.SparseTensor.from_dense(adj_with_loops)
    got = layer(features, adjacency, OUT_INDICES)
    full = ref.graph_attention_sparse_call(
        features, _indices(adj_with_loops), layer.kernels, ak, layer.biases, 2, "concat",
        "relu", False, None,
    )
    assert got.shape == (5, 8)
    np.testing.assert_allclose(got, full[OUT_INDICES], rtol=0, atol=ATOL)


def test_sparse_accepts_a_dense_adjacency(adj_with_loops, features):
    layer = _layer(sg.GraphAttentionSparse, 7, 4, 1, "concat", "relu", True, seed=21)
    np.testing.assert_allclose(
        layer(features, adj_with_loops),
        layer(features, sg.SparseTensor.from_dense(adj_with_loops)),
        rtol=0,
        atol=0.0,
    )


def test_sparse_adjacency_is_the_nonzero_entries_of_the_dense_matrix(adj_with_loops):
    adjacency = sg.SparseTensor.from_dense(adj_with_loops)
    indices = _indices(adj_with_loops)
    assert len(adjacency) == indices.shape[0] == int(adj_with_loops.sum())
    np.testing.assert_array_equal(adjacency.indices, indices)
    # Canonical row-major order, which `tf.sparse.softmax` requires.
    assert np.all(np.diff(adjacency.indices[:, 0]) >= 0)
    np.testing.assert_array_equal(adjacency.canonical_values, np.ones(len(adjacency)))
    np.testing.assert_array_equal(adjacency.toarray(), adj_with_loops)
    np.testing.assert_array_equal(adjacency.indptr, np.append(0, np.cumsum(adj_with_loops.sum(1))))
    assert adjacency.toarray().sum() == adj_with_loops.sum()
