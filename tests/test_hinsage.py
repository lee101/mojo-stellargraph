"""Parity tests for `mojostellargraph.hinsage` against the upstream oracle.

Upstream `stellargraph` v0.8.1 cannot be installed here: it pins Python < 3.9
and TensorFlow 2.1. `tests/upstream_reference.py` is therefore the oracle, a
line-by-line NumPy transliteration of the upstream methods, and this suite
checks the Mojo port against it and, where a closed form exists, against that
form directly.

Tolerances:
  * `ATOL` (1e-12) for float64 algebra that sums the same terms in the same
    order on both sides;
  * `ACT_ATOL` (1e-9) for anything that goes through the activation.
"""

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

ATOL = 1e-12
ACT_ATOL = 1e-9

B, H, D_SELF, D_NEIGH, S, NR, OUT = 3, 2, 5, 4, 3, 2, 8
HALF = OUT // 2


@pytest.fixture
def r():
    return np.random.default_rng(9091)


def head(r):
    return r.normal(size=(B, H, D_SELF))


def neigh(r, s=S):
    return r.normal(size=(NR, B, H, s, D_NEIGH))


def built(r, bias=True, activation="relu", nr=NR, out=OUT):
    agg = sg.MeanHinAggregator(out, nr, activation=activation, bias=bias)
    agg.build(D_SELF, D_NEIGH, seed=5)
    if bias:
        agg.bias = r.normal(size=out)
    return agg


# -------------------------------------------------------- MeanHinAggregator


def test_mean_hin_aggregator_matches_oracle(r):
    agg = built(r)
    x_self, x_neigh = head(r), neigh(r)
    got = agg.call(x_self, x_neigh)
    exp = ref.mean_hin_aggregator_call(
        x_self, x_neigh, agg.w_self, agg.w_neigh, agg.bias, "relu")
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ACT_ATOL


def test_mean_hin_aggregator_neighbourhood_half_closed_form(r):
    """`from_neigh = sum_r K.dot(K.mean(z_r, axis=2), w_neigh[r]) / nr`:
    the mean over relations of the mean over neighbours of the projection."""
    agg = built(r, bias=False, activation="linear")
    x_self, x_neigh = head(r), neigh(r)
    got = agg.call(x_self, x_neigh)
    per_rel = np.stack([
        (x_neigh[rel] @ agg.w_neigh[rel]).mean(axis=2) for rel in range(NR)])
    exp = np.concatenate([x_self @ agg.w_self, per_rel.mean(axis=0)], axis=2)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL


def test_mean_hin_aggregator_self_half_closed_form(r):
    """The self half is `K.dot(x[0], w_self)` and nothing else."""
    agg = built(r, bias=False, activation="linear")
    x_self, x_neigh = head(r), neigh(r)
    got = agg.call(x_self, x_neigh)
    assert np.abs(got[..., :HALF] - x_self @ agg.w_self).max() < ATOL


def test_mean_hin_aggregator_halves_are_independent(r):
    """The two halves come from disjoint inputs: changing one relation's
    neighbours cannot move the self half, and the self features cannot move
    the neighbourhood half."""
    agg = built(r, bias=False, activation="linear")
    x_self, x_neigh = head(r), neigh(r)
    before = agg.call(x_self, x_neigh)
    moved = x_neigh.copy()
    moved[0] += 3.0
    after = agg.call(x_self, moved)
    assert np.abs(after[..., :HALF] - before[..., :HALF]).max() == 0.0
    assert np.abs(after[..., HALF:] - before[..., HALF:]).max() > 1e-3
    after2 = agg.call(x_self + 3.0, x_neigh)
    assert np.abs(after2[..., HALF:] - before[..., HALF:]).max() == 0.0
    assert np.abs(after2[..., :HALF] - before[..., :HALF]).max() > 1e-3


def test_mean_hin_aggregator_output_width_is_twice_the_half(r):
    """`half_output_dim = output_dim // 2`; the two halves are concatenated."""
    for out in (8, 9, 12):
        agg = built(r, out=out)
        assert agg.half_output_dim == out // 2
        got = agg.call(head(r), neigh(r))
        assert got.shape == (B, H, out)


def test_mean_hin_aggregator_build_shapes(r):
    agg = built(r)
    assert agg.w_self.shape == (D_SELF, HALF)
    assert agg.w_neigh.shape == (NR, D_NEIGH, HALF)
    assert agg.bias.shape == (OUT,)
    assert np.isfinite(agg.w_self).all() and np.isfinite(agg.w_neigh).all()


def test_mean_hin_aggregator_build_is_deterministic_per_seed():
    a = sg.MeanHinAggregator(OUT, NR).build(D_SELF, D_NEIGH, seed=1)
    b = sg.MeanHinAggregator(OUT, NR).build(D_SELF, D_NEIGH, seed=1)
    c = sg.MeanHinAggregator(OUT, NR).build(D_SELF, D_NEIGH, seed=2)
    assert np.array_equal(a.w_self, b.w_self)
    assert np.array_equal(a.w_neigh, b.w_neigh)
    assert not np.array_equal(a.w_self, c.w_self)
    # glorot-uniform: every entry is inside +/- sqrt(6 / (fan_in + fan_out))
    lim = np.sqrt(6.0 / (D_SELF + HALF))
    assert np.abs(a.w_self).max() <= lim


def test_mean_hin_aggregator_relu_clamps(r):
    """`self.act((total + self.bias))` runs after the concatenation, so the
    negative entries of both halves are clamped to exactly zero."""
    agg = built(r)
    x_self, x_neigh = head(r), neigh(r)
    got = agg.call(x_self, x_neigh)
    assert np.minimum(got, 0.0).max() <= 0.0
    linear = sg.MeanHinAggregator(OUT, NR, activation="linear", bias=False)
    linear.build(D_SELF, D_NEIGH, seed=5)
    pre = linear.call(x_self, x_neigh) + agg.bias
    assert np.abs(np.maximum(pre, 0.0) - got).max() < ACT_ATOL


def test_mean_hin_aggregator_without_bias_matches_a_zero_bias(r):
    no_bias = built(r, bias=False, activation="linear")
    x_self, x_neigh = head(r), neigh(r)
    zero = built(r, bias=False, activation="linear")
    zero.bias = np.zeros(OUT)
    assert np.abs(no_bias.call(x_self, x_neigh) - zero.call(x_self, x_neigh)).max() == 0.0
    assert no_bias.bias is None


def test_mean_hin_aggregator_empty_neighbourhood_contributes_nothing(r):
    """A relation with `n_neighbour == 0` aggregates to a zero vector, so the
    neighbourhood half is just the self projection plus the bias."""
    agg = built(r, activation="linear")
    x_self = head(r)
    got = agg.call(x_self, np.zeros((NR, B, H, 0, D_NEIGH)))
    exp = np.concatenate(
        [x_self @ agg.w_self, np.zeros((B, H, HALF))], axis=2) + agg.bias
    assert got.shape == (B, H, OUT)
    assert np.abs(got - exp).max() < ATOL
    # the oracle treats the same empty relation the same way
    assert np.abs(got - ref.mean_hin_aggregator_call(
        x_self, np.zeros((NR, B, H, 0, D_NEIGH)), agg.w_self, agg.w_neigh,
        agg.bias, "linear")).max() < ATOL


def test_mean_hin_aggregator_swapping_relations_and_weights_is_a_no_op(r):
    """`sum(neigh_agg_by_relation) / nr` pairs relation `r` with `w_neigh[r]`,
    so permuting the stack and the weights together cannot move the output."""
    agg = built(r, bias=False, activation="linear")
    x_self, x_neigh = head(r), neigh(r)
    before = agg.call(x_self, x_neigh)
    agg.w_neigh = agg.w_neigh[::-1].copy()
    after = agg.call(x_self, x_neigh[::-1])
    assert np.abs(before - after).max() == 0.0


def test_mean_hin_aggregator_rejects_a_wrong_relation_count(r):
    agg = built(r)
    with pytest.raises(ValueError):
        agg.call(head(r), r.normal(size=(NR + 1, B, H, S, D_NEIGH)))


def test_mean_hin_aggregator_rejects_unsupported_activation():
    with pytest.raises(ValueError):
        sg.MeanHinAggregator(OUT, NR, activation="not-an-activation")


# -------------------------------------------------------------------- HinSAGE


def test_hinsage_model_shape_and_oracle_parity(r):
    m = sg.HinSAGE([OUT, OUT], None, NR, "relu", True)
    m.build([(D_SELF, D_NEIGH), (OUT, OUT)], seed=0)
    head_in = head(r)
    nbrs = [neigh(r), r.normal(size=(NR, B, H, 2, OUT))]
    got = m(head_in, nbrs)
    h = head_in
    for agg, rel in zip(m._aggs, nbrs):
        h = ref.mean_hin_aggregator_call(
            h, rel, agg.w_self, agg.w_neigh, agg.bias, agg.act)
    assert got.shape == (B, H, OUT)
    assert np.abs(got - h).max() < ACT_ATOL


def test_hinsage_layer_widths_follow_the_layer_sizes(r):
    m = sg.HinSAGE([OUT, OUT], None, NR, "linear", False)
    m.build([(D_SELF, D_NEIGH), (OUT, OUT)], seed=0)
    assert [a.output_dim for a in m._aggs] == [OUT, OUT]
    assert [a.half_output_dim for a in m._aggs] == [HALF, HALF]
    # the second layer consumes the first layer's output width
    assert m._aggs[1].w_self.shape == (OUT, HALF)
    assert m._aggs[1].w_neigh.shape == (NR, OUT, HALF)
    # one aggregator per layer, each with its own weights
    assert m._aggs[0].w_self is not m._aggs[1].w_self


def test_hinsage_is_the_aggregation_chain(r):
    """`HinSAGE` is one `MeanHinAggregator` per layer, applied in order."""
    m = sg.HinSAGE([OUT, OUT], None, NR, "relu", True)
    m.build([(D_SELF, D_NEIGH), (OUT, OUT)], seed=0)
    x_self = head(r)
    nbrs = [neigh(r), r.normal(size=(NR, B, H, 2, OUT))]
    got = m(x_self, nbrs)
    step = m._aggs[0].call(x_self, nbrs[0])
    final = m._aggs[1].call(step, nbrs[1])
    assert np.abs(got - final).max() == 0.0
    assert step.shape == (B, H, OUT)


def test_hinsage_neighbourhood_half_closed_form(r):
    """`nr` relations of `s` neighbours each: the neighbourhood half of the
    output is the mean over relations of the mean over neighbours of the
    projection, exactly as upstream's `neigh_agg_by_relation` loop builds it."""
    m = sg.HinSAGE([8], None, NR, "linear", False)
    m.build([(D_SELF, D_NEIGH)], seed=2)
    agg = m._aggs[0]
    x_self, x_neigh = head(r), neigh(r, s=5)
    got = agg.call(x_self, x_neigh)
    per_rel = np.stack([
        (x_neigh[rel] @ agg.w_neigh[rel]).mean(axis=2) for rel in range(NR)])
    assert np.abs(got[..., HALF:] - per_rel.mean(axis=0)).max() < ATOL
    # a single relation makes the mean over relations a no-op
    one = sg.MeanHinAggregator(OUT, 1, activation="linear", bias=False)
    one.build(D_SELF, D_NEIGH, seed=2)
    solo = one.call(x_self, x_neigh[:1])
    assert np.abs(solo[..., HALF:] - per_rel[0]).max() < ATOL


def test_hinsage_more_relations_change_the_neighbourhood_half(r):
    """A third relation enters only through the mean over relations, and
    moving it does not disturb the self half."""
    two = sg.MeanHinAggregator(OUT, 2, activation="linear", bias=False)
    two.build(D_SELF, D_NEIGH, seed=0)
    x_self, x_neigh = head(r), neigh(r)
    base = two.call(x_self, x_neigh)
    moved = x_neigh.copy()
    moved[1] *= 3.0
    after = two.call(x_self, moved)
    # only the neighbourhood half moves, and it moves by the halved change
    assert np.abs(after[..., :HALF] - base[..., :HALF]).max() == 0.0
    assert np.abs(after[..., HALF:] - base[..., HALF:]).max() > 1e-3
    expected = 0.5 * ((x_neigh[0] @ two.w_neigh[0]).mean(axis=2)
                      + (3.0 * x_neigh[1] @ two.w_neigh[1]).mean(axis=2))
    assert np.abs(after[..., HALF:] - expected).max() < ATOL


def test_hinsage_build_is_deterministic(r):
    dims = [(D_SELF, D_NEIGH), (OUT, OUT)]
    a = sg.HinSAGE([OUT, OUT], None, NR, "relu", True).build(dims, seed=0)
    b = sg.HinSAGE([OUT, OUT], None, NR, "relu", True).build(dims, seed=0)
    for x, y in zip(a._aggs, b._aggs):
        assert np.array_equal(x.w_self, y.w_self)
        assert np.array_equal(x.w_neigh, y.w_neigh)
    x_self = head(r)
    nbrs = [neigh(r), r.normal(size=(NR, B, H, 2, OUT))]
    assert np.abs(a(x_self, nbrs) - b(x_self, nbrs)).max() == 0.0


def test_hinsage_rejects_the_wrong_number_of_neighbourhoods(r):
    m = sg.HinSAGE([OUT, OUT], None, NR, "relu", True)
    m.build([(D_SELF, D_NEIGH), (OUT, OUT)], seed=0)
    nbrs = [neigh(r)]
    with pytest.raises(ValueError):
        m(head(r), nbrs)
    with pytest.raises(ValueError):
        m(head(r), nbrs * 3)


def test_hinsage_requires_build_first(r):
    m = sg.HinSAGE([8], None, NR, "relu", True)
    with pytest.raises(RuntimeError):
        m(head(r), [neigh(r)])


def test_hinsage_single_layer_returns_one_array(r):
    m = sg.HinSAGE([8], None, NR, "linear", True)
    m.build([(D_SELF, D_NEIGH)], seed=0)
    got = m(head(r), [neigh(r)])
    assert isinstance(got, np.ndarray)
    assert got.shape == (B, H, OUT)
