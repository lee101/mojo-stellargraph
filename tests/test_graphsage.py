"""Parity tests for `mojostellargraph.graphsage` against the upstream oracle.

Upstream `stellargraph` v0.8.1 cannot be installed here: it pins Python < 3.9
and TensorFlow 2.1. `tests/upstream_reference.py` is therefore the oracle, a
line-by-line NumPy transliteration of the upstream methods, and this suite
checks the Mojo port against it and, where a closed form exists, against that
form directly.

Tolerances:
  * `ATOL` (1e-12) for float64 algebra that sums the same terms in the same
    order on both sides;
  * `SOFTMAX_ATOL` (1e-9) for anything that goes through a softmax.
"""

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

ATOL = 1e-12
SOFTMAX_ATOL = 1e-9

B, H, D, S, OUT = 3, 2, 5, 4, 7


@pytest.fixture
def r():
    return np.random.default_rng(4242)


def head(r):
    return r.normal(size=(B, H, D))


def neigh(r):
    return r.normal(size=(B, H, S, D))


# ------------------------------------------------------------ MeanAggregator


def test_mean_aggregator_head_group_matches_oracle(r):
    agg = sg.MeanAggregator(OUT, bias=True, act="relu")
    w = agg._build_group_weights(D, 0, OUT)
    x = head(r)
    got = agg.group_aggregate(x, 0)
    exp = ref.mean_aggregator_group_aggregate(x, w, 0)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_mean_aggregator_neighbour_group_matches_oracle(r):
    agg = sg.MeanAggregator(OUT, bias=True, act="relu")
    w = agg._build_group_weights(D, 1, OUT)
    x = neigh(r)
    got = agg.group_aggregate(x, 1)
    exp = ref.mean_aggregator_group_aggregate(x, w, 1)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_mean_aggregator_neighbour_group_is_mean_then_dot(r):
    """`K.dot(K.mean(x_group, axis=2), w)`: the closed form, no oracle."""
    agg = sg.MeanAggregator(OUT, bias=True, act="relu")
    w = agg._build_group_weights(D, 1, OUT)
    x = neigh(r)
    exp = x.mean(axis=2) @ w
    assert np.abs(agg.group_aggregate(x, 1) - exp).max() < ATOL


def test_mean_aggregator_head_group_is_a_plain_dot(r):
    """Group 0 is not averaged: `K.dot(x_group, w)` contracts the features."""
    agg = sg.MeanAggregator(OUT, bias=True, act="relu")
    w = agg._build_group_weights(D, 0, OUT)
    x = head(r)
    exp = x @ w
    assert np.abs(agg.group_aggregate(x, 0) - exp).max() < ATOL


def test_mean_aggregator_group_weights_are_shaped_and_deterministic():
    a1 = sg.MeanAggregator(OUT)
    a2 = sg.MeanAggregator(OUT)
    w = a1._build_group_weights(D, 1, OUT)
    assert w.shape == (D, OUT)
    # the weight is a pure function of (in_dim, group_idx, out_size)
    assert np.array_equal(w, a2._build_group_weights(D, 1, OUT))
    assert a1.w_group[1].shape == (D, OUT)
    # a different group index draws a different matrix
    assert not np.array_equal(w, a1._build_group_weights(D, 2, OUT))
    assert a1._build_group_weights(D + 1, 3, OUT).shape == (D + 1, OUT)


def test_mean_aggregator_call_applies_bias_then_activation(r):
    agg = sg.MeanAggregator(2 * OUT, bias=True, act="relu")
    agg.bias = r.normal(size=2 * OUT)
    sources = r.normal(size=(B, H, 2 * OUT))
    got = agg.call(sources)
    exp = ref.graphsage_aggregator_call(sources, agg.bias, "relu")
    assert got.shape == (B, H, 2 * OUT)
    assert np.abs(got - exp).max() < ATOL
    # relu, so nothing is left negative
    assert np.minimum(got, 0.0).max() <= 0.0


def test_mean_aggregator_without_bias_matches_oracle(r):
    agg = sg.MeanAggregator(OUT, bias=False, act="linear")
    w = agg._build_group_weights(D, 1, OUT)
    assert agg.bias is None
    x = neigh(r)
    got = agg.group_aggregate(x, 1)
    assert np.abs(got - ref.mean_aggregator_group_aggregate(x, w, 1)).max() < ATOL
    sources = r.normal(size=(B, H, OUT))
    assert np.abs(agg.call(sources) - sources).max() < ATOL


def test_mean_aggregator_rejects_unsupported_activation():
    with pytest.raises(ValueError):
        sg.MeanAggregator(OUT, act="not-an-activation")


def test_mean_aggregator_rejects_wrong_rank(r):
    agg = sg.MeanAggregator(OUT)
    agg._build_group_weights(D, 1, OUT)
    with pytest.raises(ValueError):
        agg.group_aggregate(r.normal(size=(B, D)), 1)


# --------------------------------------------------------- pooling aggregators
POOLS = [
    (sg.MaxPoolingAggregator, ref.max_pooling_aggregator_group_aggregate),
    (sg.MeanPoolingAggregator, ref.mean_pooling_aggregator_group_aggregate),
]


def built_pool(cls):
    agg = cls(OUT, bias=True, act="relu")
    agg._build_group_weights(D, 0, OUT)
    agg._build_group_weights(D, 1, OUT)
    return agg


@pytest.mark.parametrize("cls,oracle", POOLS)
def test_pooling_head_group_matches_oracle(r, cls, oracle):
    agg = built_pool(cls)
    x = head(r)
    got = agg.group_aggregate(x, 0)
    exp = oracle(x, agg.w_group[0], agg.w_pool[0], agg.b_pool[0], 0)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


@pytest.mark.parametrize("cls,oracle", POOLS)
def test_pooling_neighbour_group_matches_oracle(r, cls, oracle):
    agg = built_pool(cls)
    x = neigh(r)
    got = agg.group_aggregate(x, 1)
    exp = oracle(x, agg.w_group[1], agg.w_pool[1], agg.b_pool[1], 1)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_max_pooling_neighbour_group_is_relu_then_max_then_dot(r):
    """Eq. (3) of Hamilton et al. (2017), written out without the oracle."""
    agg = built_pool(sg.MaxPoolingAggregator)
    x = neigh(r)
    xw = np.maximum(x @ agg.w_pool[1] + agg.b_pool[1], 0.0)
    exp = xw.max(axis=2) @ agg.w_group[1]
    assert np.abs(agg.group_aggregate(x, 1) - exp).max() < ATOL
    # the max is at least the mean before the projection
    assert (xw.max(axis=2) >= xw.mean(axis=2) - ATOL).all()


def test_mean_pooling_neighbour_group_is_relu_then_mean_then_dot(r):
    agg = built_pool(sg.MeanPoolingAggregator)
    x = neigh(r)
    xw = np.maximum(x @ agg.w_pool[1] + agg.b_pool[1], 0.0)
    exp = xw.mean(axis=2) @ agg.w_group[1]
    assert np.abs(agg.group_aggregate(x, 1) - exp).max() < ATOL


def test_max_and_mean_pooling_reduce_differently(r):
    mx = built_pool(sg.MaxPoolingAggregator)
    mn = built_pool(sg.MeanPoolingAggregator)
    x = neigh(r)
    a = mx.group_aggregate(x, 1)
    b = mn.group_aggregate(x, 1)
    assert np.abs(a - b).max() > 1e-3


def test_pooling_group_weight_shapes():
    """A neighbour group pools into `hidden_dim` first, then into `output_dim`;
    the head group only ever gets the plain projection."""
    agg = built_pool(sg.MaxPoolingAggregator)
    assert agg.hidden_dim == OUT
    assert agg.hidden_act == "relu"
    assert agg.w_group[0].shape == (D, OUT)
    assert agg.w_group[1].shape == (agg.hidden_dim, OUT)
    assert agg.w_pool[1].shape == (D, agg.hidden_dim)
    assert agg.b_pool[1].shape == (agg.hidden_dim,)
    assert np.abs(agg.b_pool[1]).max() == 0.0


@pytest.mark.parametrize("cls,oracle", POOLS)
def test_pooling_head_group_ignores_the_pool_weights(r, cls, oracle):
    """Group 0 is `K.dot(x_group, w_group[0])`; w_pool[0]/b_pool[0] are never
    read, so overwriting them cannot move the result."""
    agg = built_pool(cls)
    x = head(r)
    before = agg.group_aggregate(x, 0)
    agg.w_pool[0] = np.full((D, OUT), 7.0)
    agg.b_pool[0] = np.full(OUT, -3.0)
    assert np.abs(agg.group_aggregate(x, 0) - before).max() == 0.0


@pytest.mark.parametrize("cls,oracle", POOLS)
def test_pooling_group_aggregate_rejects_wrong_rank(r, cls, oracle):
    agg = built_pool(cls)
    with pytest.raises(ValueError):
        agg.group_aggregate(r.normal(size=(B, H, S, D, 1)), 1)


# ------------------------------------------------------ AttentionalAggregator


def test_attentional_group_aggregate_matches_oracle(r):
    agg = sg.AttentionalAggregator(OUT, bias=True, act="relu")
    w = agg._build_group_weights(D, 1, OUT)
    x_self, x_g = head(r), neigh(r)
    got = agg.group_aggregate(x_self, x_g, 1)
    exp = ref.attentional_aggregator_group_aggregate(
        x_self, x_g, w, agg.w_attn_s, agg.w_attn_g)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < SOFTMAX_ATOL


def test_attentional_output_is_a_convex_combination_of_the_projections(r):
    """`K.sum(softmax(attn) * xw_all, axis=2)`: every output entry lies
    between the smallest and the largest of the projected rows."""
    agg = sg.AttentionalAggregator(OUT)
    w = agg._build_group_weights(D, 1, OUT)
    x_self, x_g = head(r), neigh(r)
    xw_all = np.concatenate([(x_self @ w)[:, :, None, :], x_g @ w], axis=2)
    got = agg.group_aggregate(x_self, x_g, 1)
    assert (got >= xw_all.min(axis=2) - SOFTMAX_ATOL).all()
    assert (got <= xw_all.max(axis=2) + SOFTMAX_ATOL).all()


def test_attentional_uniform_logits_average_the_projections(r):
    """With both attention kernels zero every logit is 0, the softmax is
    uniform over the `n_neighbour + 1` rows, and the output is their mean."""
    agg = sg.AttentionalAggregator(OUT)
    w = agg._build_group_weights(D, 1, OUT)
    agg.w_attn_s = np.zeros((OUT, 1))
    agg.w_attn_g = np.zeros((OUT, 1))
    x_self, x_g = head(r), neigh(r)
    xw_all = np.concatenate([(x_self @ w)[:, :, None, :], x_g @ w], axis=2)
    exp = xw_all.mean(axis=2)
    assert np.abs(agg.group_aggregate(x_self, x_g, 1) - exp).max() < SOFTMAX_ATOL


def test_attentional_call_matches_oracle(r):
    """The group loop, then `K.concatenate`, the bias and the activation.

    `calculate_group_sizes` splits `output_dim` across the groups, so the
    concatenation is `output_dim` wide and the one bias lines up with it. The
    split is exercised on an `output_dim` that divides evenly, because the one
    `w_attn_s` / `w_attn_g` pair the layer holds is built at the last group's
    width and every group reads it; upstream overwrites it per group the same
    way, so unequal group widths are outside what either layer can express.
    """
    width = 6
    agg = sg.AttentionalAggregator(width, bias=True, act="relu")
    w1 = agg._build_group_weights(D, 1, width // 2)
    w2 = agg._build_group_weights(D, 2, width // 2)
    # the two groups share the one attention kernel pair the layer holds
    w_attn_s, w_attn_g = agg.w_attn_s, agg.w_attn_g
    x_self, x_g = head(r), neigh(r)
    agg.bias = r.normal(size=width)
    got = agg.call(x_self, np.stack([x_g, x_g], axis=0))
    exp = ref.graphsage_aggregator_call(
        np.concatenate([
            ref.attentional_aggregator_group_aggregate(
                x_self, x_g, w, w_attn_s, w_attn_g)
            for w in (w1, w2)], axis=2),
        agg.bias, "relu")
    assert got.shape == (B, H, width)
    assert np.abs(got - exp).max() < SOFTMAX_ATOL


def test_attentional_call_rejects_a_bias_of_the_wrong_width(r):
    """The kernel adds `self.bias` over the whole concatenated row, so a bias
    of any other length would be an out-of-bounds read, not a wrong number."""
    agg = sg.AttentionalAggregator(OUT, bias=True, act="relu")
    x_self, x_g = head(r), neigh(r)
    for idx in (1, 2):
        agg._build_group_weights(D, idx, OUT // 2)
    agg.bias = r.normal(size=OUT + 1)
    with pytest.raises(ValueError):
        agg.call(x_self, np.stack([x_g, x_g], axis=0))


def test_attentional_rejects_wrong_rank(r):
    agg = sg.AttentionalAggregator(OUT)
    agg._build_group_weights(D, 1, OUT)
    with pytest.raises(ValueError):
        agg.group_aggregate(head(r), r.normal(size=(B, H, D)), 1)


# ------------------------------------------------------------------ GraphSAGE


def oracle_mean_model(layer_sizes, aggs, xin, normalize="l2"):
    """The upstream layer loop for `MeanAggregator`, from the oracle only."""
    h = [np.asarray(x, dtype=np.float64) for x in xin]
    for layer in range(len(layer_sizes)):
        out = []
        for i in range(len(layer_sizes) - layer):
            row = aggs[layer][i]
            x_self, x_next = h[i], h[i + 1]
            neigh_in = x_next.reshape(
                x_self.shape[0], x_self.shape[1], -1, x_next.shape[-1])
            parts = [ref.mean_aggregator_group_aggregate(
                x_self, row[0].w_group[0], 0)]
            parts += [ref.mean_aggregator_group_aggregate(
                neigh_in, a.w_group[g], g) for g, a in enumerate(row[1:], 1)]
            out.append(ref.graphsage_aggregator_call(
                np.concatenate(parts, axis=2), row[0].bias, row[0].act))
        h = out
    outs = []
    for x in h:
        # `Reshape(K.int_shape(x)[2:])(x) if K.int_shape(x)[1] == 1 else x`, with
        # the batch axis the Keras tensor keeps and the squeezed
        # representation does not
        if x.shape[1] == 1:
            x = x.reshape((x.shape[0],) + x.shape[2:])
        outs.append(ref.graphsage_normalization(x, normalize))
    return outs[0] if len(outs) == 1 else outs


def test_graphsage_single_layer_matches_oracle(r):
    xin = [head(r), neigh(r)]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                     normalize="l2")
    got = m(xin)
    exp = oracle_mean_model([OUT], m._aggs, xin, "l2")
    # the self and neighbour groups split the layer's output width
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_graphsage_two_layers_match_oracle(r):
    # n_neighbour == 1 on the first hop, so the self group of the second head
    # is the plain projection upstream takes
    xin = [head(r), r.normal(size=(B, H, 1, D)), r.normal(size=(B, H, 3, D))]
    m = sg.GraphSAGE([OUT, 5], aggregator=sg.MeanAggregator, bias=False,
                     normalize="l2")
    got = m(xin)
    exp = oracle_mean_model([OUT, 5], m._aggs, xin, "l2")
    assert got.shape == (B, H, 5)
    assert np.abs(got - exp).max() < ATOL


def test_graphsage_single_head_drops_the_neighbourhood_axis(r):
    """Upstream's `Reshape(K.int_shape(x)[2:])(x) if K.int_shape(x)[1] == 1`:
    `neighbourhood_sizes[0]` is 1 upstream, so one head is the canonical
    layout and the head axis goes away with it."""
    xin = [r.normal(size=(B, 1, D)), r.normal(size=(B, 1, 3, D))]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                     normalize="l2")
    got = m(xin)
    assert got.shape == (B, OUT)
    np.testing.assert_allclose(
        got, oracle_mean_model([OUT], m._aggs, xin, "l2"), rtol=0, atol=ATOL
    )


def test_graphsage_head_group_contracts_the_whole_neighbour_block(r):
    """`K.dot(x[i], w)` takes a rank-2 left operand, so the head node group
    contracts everything below `n_head` at once rather than one row of it."""
    n_neigh, heads = 3, 2
    xin = [r.normal(size=(B, heads, n_neigh, D)), r.normal(size=(B, heads, 3, D))]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                     normalize=None)
    m.build([[(n_neigh * D, D)]])
    agg = m._aggs[0][0][0]
    got = agg.group_aggregate(xin[0], 0)
    want = (xin[0].reshape(B * heads, n_neigh * D) @ agg.w_group[0]).reshape(
        B, heads, agg.output_dim
    )
    assert np.abs(got - want).max() == 0.0


def test_attentional_aggregator_accepts_more_neighbours_than_output_width(r):
    """Upstream keeps `attn` as its own `[b, h, n_neighbour + 1, 1]` tensor; the
    port stages it in the output buffer, which therefore has to be at least
    that wide even when it is wider than the output."""
    n_neigh, units = 12, 4
    agg = sg.AttentionalAggregator(units, bias=True, act="relu")
    agg._build_group_weights(D, 0, units)
    agg._build_group_weights(D, 1, units)
    x_self = r.normal(size=(B, H, 1, D))
    x_neigh = r.normal(size=(B, H, n_neigh, D))
    got = agg.call(x_self, x_neigh[None, ...])
    assert got.shape == (B, H, units)
    assert np.isfinite(got).all()
    # a single group, so the whole `output_dim` is that group's, and the
    # numbers are the oracle's rather than merely finite
    w, w_attn_s, w_attn_g = agg.w_group[0], agg.w_attn_s, agg.w_attn_g
    want = ref.graphsage_aggregator_call(
        ref.attentional_aggregator_group_aggregate(
            x_self[:, :, 0, :], x_neigh, w, w_attn_s, w_attn_g
        ),
        agg.bias, "relu",
    )
    np.testing.assert_allclose(got, want, rtol=0, atol=SOFTMAX_ATOL)


def test_graphsage_l2_normalization_gives_unit_rows(r):
    xin = [head(r), neigh(r)]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                     normalize="l2")
    got = np.asarray(m(xin))
    norms = np.linalg.norm(got.reshape(-1, got.shape[-1]), axis=1)
    assert np.abs(norms - 1.0).max() < SOFTMAX_ATOL


def test_graphsage_without_normalization_keeps_row_norms(r):
    xin = [head(r), neigh(r)]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                     normalize=None)
    got = np.asarray(m(xin))
    # unnormalized output is exactly the pre-normalization activation
    exp = oracle_mean_model([OUT], m._aggs, xin, "none")
    assert np.abs(got - exp).max() < ATOL
    # and it is what the l2-normalized model would have rescaled
    l2 = ref.graphsage_normalization(got, "l2")
    assert np.abs(
        np.linalg.norm(l2, axis=-1) - 1.0).max() < SOFTMAX_ATOL
    assert np.abs(got - l2).max() > 0.1


def test_graphsage_normalization_leaves_a_zero_row_at_zero(r):
    """The L2 normalization clamps the norm away from zero, so an all-zero
    input stays all-zero instead of becoming NaN."""
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                     normalize="l2")
    got = np.asarray(m([np.zeros((B, H, D)), np.zeros((B, H, S, D))]))
    assert got.shape == (B, H, OUT)
    assert np.isfinite(got).all()
    assert np.abs(got).max() == 0.0


def test_graphsage_pooling_aggregator_end_to_end(r):
    xin = [head(r), neigh(r)]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanPoolingAggregator, bias=False,
                     normalize=None)
    got = np.asarray(m(xin))
    row = m._aggs[0][0]
    p0 = ref.mean_pooling_aggregator_group_aggregate(
        xin[0], row[0].w_group[0], row[0].w_pool[0], row[0].b_pool[0], 0)
    p1 = ref.mean_pooling_aggregator_group_aggregate(
        xin[1], row[1].w_group[1], row[1].w_pool[1], row[1].b_pool[1], 1)
    exp = ref.graphsage_aggregator_call(
        np.concatenate([p0, p1], axis=2), None, row[0].act)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_graphsage_explicit_build_matches_the_lazy_build(r):
    xin = [head(r), neigh(r)]
    lazy = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                        normalize=None)
    a = lazy(xin)
    built = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False,
                         normalize=None)
    assert built.build([[(D, D)]]) is built
    b = built(xin)
    assert np.abs(a - b).max() == 0.0
    assert len(built._aggs) == 1 and len(built._aggs[0]) == 1


def test_graphsage_bias_spans_the_output_width(r):
    """`h_out = K.concatenate(sources, axis=2) + self.bias`, so the bias is
    added once over the whole layer output."""
    xin = [head(r), neigh(r)]
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=True,
                     normalize=None)
    m.build([[(D, D)]])
    aggs = m._aggs[0][0]
    bias = r.normal(size=OUT)
    aggs[0].bias = bias
    got = np.asarray(m(xin))
    p0 = ref.mean_aggregator_group_aggregate(xin[0], aggs[0].w_group[0], 0)
    p1 = ref.mean_aggregator_group_aggregate(xin[1], aggs[1].w_group[1], 1)
    exp = ref.graphsage_aggregator_call(
        np.concatenate([p0, p1], axis=2), bias, aggs[0].act)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_graphsage_attentional_aggregator_end_to_end(r):
    """The attentional layer has no self group of its own: it attends over the
    head node and its neighbours and projects once."""
    xin = [head(r), neigh(r)]
    m = sg.GraphSAGE([OUT], aggregator=sg.AttentionalAggregator, bias=False,
                     normalize=None)
    m.build([[(D, D)]])
    agg = m._aggs[0][0][0]
    got = np.asarray(m(xin))
    exp = ref.attentional_aggregator_group_aggregate(
        xin[0], xin[1], agg.w_group[1], agg.w_attn_s, agg.w_attn_g)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < SOFTMAX_ATOL


def test_graphsage_splits_the_output_width_like_upstream():
    """`calculate_group_sizes`: `output_dim // num_groups` per group, with the
    remainder on the first, and the head node group takes no weight of its own
    for the attentional aggregator."""
    m = sg.GraphSAGE([7], aggregator=sg.MeanAggregator, bias=False)
    m.build([[(D, D)]])
    aggs = m._aggs[0][0]
    assert [a.output_dim for a in aggs] == [4, 3]
    even = sg.GraphSAGE([8], aggregator=sg.MeanAggregator, bias=False)
    even.build([[(D, D)]])
    assert [a.output_dim for a in even._aggs[0][0]] == [4, 4]
    att = sg.GraphSAGE([8], aggregator=sg.AttentionalAggregator, bias=False)
    att.build([[(D, D)]])
    assert len(att._aggs[0][0]) == 1
    assert att._aggs[0][0][0].output_dim == 8


def test_graphsage_rejects_a_non_list_input(r):
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False)
    with pytest.raises(TypeError):
        m(head(r))
    with pytest.raises(TypeError):
        m(np.zeros((B, H, D)))


def test_graphsage_rejects_the_wrong_number_of_inputs(r):
    m = sg.GraphSAGE([OUT], aggregator=sg.MeanAggregator, bias=False)
    with pytest.raises(ValueError):
        m([head(r)])
    with pytest.raises(ValueError):
        m([head(r), neigh(r), neigh(r)])
    two = sg.GraphSAGE([OUT, 5], aggregator=sg.MeanAggregator, bias=False)
    with pytest.raises(ValueError):
        two([head(r), neigh(r)])


def test_graphsage_rejects_an_unknown_normalization():
    with pytest.raises(ValueError):
        sg.GraphSAGE([OUT], normalize="l1")
