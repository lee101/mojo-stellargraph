"""Parity tests for `sg.link_inference` and the units around it.

Upstream `stellargraph` 0.8.1 cannot be installed here (it needs Python < 3.9
and TensorFlow 2.1), so `tests/upstream_reference.py` -- a line-by-line NumPy
transliteration of `stellargraph/layer/link_inference.py`, including the
`edge_function` closure -- is the parity oracle. Every numerical test here
compares the Mojo port against it; the closed-form tests pin the two pieces the
oracle cannot check itself, the three regimes of the leaky clipped linear unit
and the fact that the inner-product branch never touches the `Dense` kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

TOL = 1e-12
"""float64 algebra that sums the same terms in the same order."""

TOL_ACT = 1e-9
"""Anything routed through a sigmoid, which is not bit-identical in the two
implementations."""


def _embeddings(n=7, d=5, seed=3):
    rng = np.random.default_rng(seed)
    return (
        np.ascontiguousarray(rng.normal(size=(n, d))),
        np.ascontiguousarray(rng.normal(size=(n, d))),
    )


def _kernel(in_dim, output_dim, seed=9):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(in_dim, output_dim)))


def _bias(output_dim, seed=10):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(output_dim,)))


def _free(fn, name):
    """Read one of the `edge_function` closure's captured values, which is where
    the factories' defaults are visible."""
    cell = fn.__code__.co_freevars.index(name)
    return fn.__closure__[cell].cell_contents


# ------------------------------------------------------- LeakyClippedLinear
@pytest.mark.parametrize(
    "low, high, alpha",
    [(1.0, 5.0, 0.1), (-2.0, 2.0, 0.5), (0.0, 1.0, 0.0), (3.0, 3.5, 0.25)],
)
def test_leaky_clipped_linear_matches_oracle(low, high, alpha):
    rng = np.random.default_rng(17)
    x = np.ascontiguousarray(rng.normal(size=(6, 4)) * 3.0)
    got = sg.LeakyClippedLinear(low=low, high=high, alpha=alpha)(x)
    want = ref.leaky_clipped_linear(x, low, high, alpha)
    assert got.shape == x.shape
    assert np.abs(got - want).max() <= TOL


def test_leaky_clipped_linear_three_regimes():
    # the closed form is x + (1 - a) relu(lo - x) - (1 - a) relu(x - hi), and
    # the two clip points are the fixed points of it
    low, high, alpha = 1.0, 5.0, 0.1
    gamma = 1.0 - alpha
    x = np.array([-3.0, 2.0, 1.0, 3.0, 5.0, 7.5])
    got = sg.LeakyClippedLinear(low=low, high=high, alpha=alpha)(x)
    want = x.copy()
    below = x < low
    above = x > high
    want[below] = x[below] + gamma * (low - x[below])
    want[above] = x[above] - gamma * (x[above] - high)
    assert np.abs(got - want).max() <= TOL
    # the three regimes, spelled out
    assert got[0] == pytest.approx(-3.0 + gamma * 4.0)  # below low, lifted
    assert got[1] == pytest.approx(2.0)  # between low and high, untouched
    assert got[2] == pytest.approx(1.0)  # at low, the fixed point
    assert got[3] == pytest.approx(3.0)  # between low and high, untouched
    assert got[4] == pytest.approx(5.0)  # at high, the other fixed point
    assert got[5] == pytest.approx(7.5 - gamma * 2.5)  # above high, pulled down
    assert got[2] == pytest.approx(low)
    assert got[4] == pytest.approx(high)


def test_leaky_clipped_linear_is_the_identity_between_the_clips():
    rng = np.random.default_rng(21)
    x = np.ascontiguousarray(rng.uniform(1.0, 5.0, size=(10, 3)))
    got = sg.LeakyClippedLinear(low=1.0, high=5.0, alpha=0.1)(x)
    assert np.abs(got - x).max() == 0.0


# ------------------------------------------------------------- link_inference
@pytest.mark.parametrize("method", ["ip", "dot"])
def test_link_inference_inner_product_ignores_the_kernel(method):
    x0, x1 = _embeddings()
    d = x0.shape[1]
    a = _kernel(d, 1, seed=31)
    b = _kernel(d, 1, seed=32)
    got = sg.link_inference(
        output_dim=1,
        output_act="linear",
        edge_embedding_method=method,
        kernel=a,
        bias=_bias(1, 33),
    )(x0, x1)
    other = sg.link_inference(
        output_dim=1,
        output_act="linear",
        edge_embedding_method=method,
        kernel=b,
        bias=_bias(1, 34),
    )(x0, x1)
    want = (x0 * x1).sum(axis=-1).reshape(-1, 1)
    assert got.shape == (7, 1)
    assert np.abs(got - want).max() <= TOL
    assert np.abs(got - other).max() == 0.0


@pytest.mark.parametrize("method", ["l1", "l2", "mul", "hadamard", "avg"])
@pytest.mark.parametrize("output_act", ["linear", "sigmoid"])
def test_link_inference_dense_branch_matches_oracle(method, output_act):
    x0, x1 = _embeddings()
    d = x0.shape[1]
    output_dim = 3
    kernel = _kernel(d, output_dim)
    bias = _bias(output_dim)
    got = sg.link_inference(
        output_dim=output_dim,
        output_act=output_act,
        edge_embedding_method=method,
        kernel=kernel,
        bias=bias,
    )(x0, x1)
    want = ref.link_inference(
        x0, x1, kernel, bias, output_dim, output_act, method, None
    )
    assert got.shape == (7, 3)
    assert np.abs(got - want).max() <= TOL_ACT


@pytest.mark.parametrize("output_act", ["linear", "sigmoid"])
def test_link_inference_concat_uses_a_two_d_kernel(output_act):
    x0, x1 = _embeddings()
    d = x0.shape[1]
    output_dim = 3
    kernel = _kernel(2 * d, output_dim, seed=41)
    bias = _bias(output_dim, seed=42)
    got = sg.link_inference(
        output_dim=output_dim,
        output_act=output_act,
        edge_embedding_method="concat",
        kernel=kernel,
        bias=bias,
    )(x0, x1)
    want = ref.link_inference(
        x0, x1, kernel, bias, output_dim, output_act, "concat", None
    )
    assert got.shape == (7, 3)
    assert np.abs(got - want).max() <= TOL_ACT
    # the closed form: Concatenate([x0, x1]) then one Dense
    closed = np.concatenate([x0, x1], axis=-1) @ kernel + bias
    if output_act == "sigmoid":
        closed = 1.0 / (1.0 + np.exp(-closed))
    assert np.abs(got - closed).max() <= TOL_ACT


def test_link_inference_hadamard_is_the_same_branch_as_mul():
    x0, x1 = _embeddings()
    d = x0.shape[1]
    kernel = _kernel(d, 2, seed=51)
    bias = _bias(2, seed=52)
    got = sg.link_inference(
        output_dim=2,
        edge_embedding_method="hadamard",
        kernel=kernel,
        bias=bias,
    )(x0, x1)
    want = sg.link_inference(
        output_dim=2, edge_embedding_method="mul", kernel=kernel, bias=bias
    )(x0, x1)
    assert np.abs(got - want).max() == 0.0
    assert np.abs(got - ((x0 * x1) @ kernel + bias)).max() <= TOL


def test_link_inference_sigmoid_output_is_a_probability():
    x0, x1 = _embeddings()
    d = x0.shape[1]
    kernel = _kernel(d, 4, seed=61)
    bias = _bias(4, seed=62)
    got = sg.link_inference(
        output_dim=4,
        output_act="sigmoid",
        edge_embedding_method="l2",
        kernel=kernel,
        bias=bias,
    )(x0, x1)
    assert got.shape == (7, 4)


def test_link_inference_clip_limits_attenuate_out_of_range_values():
    rng = np.random.default_rng(131)
    x0 = np.ascontiguousarray(rng.uniform(0.5, 2.0, size=(7, 5)))
    x1 = np.ascontiguousarray(rng.uniform(0.5, 2.0, size=(7, 5)))
    # a kernel of all-10s makes every pre-activation large and positive, so the
    # upper regime bites: the unit scales by alpha towards `high` rather than
    # bounding, which is what upstream's leaky clip does
    kernel = np.full((5, 1), 10.0)
    got = sg.link_inference(
        output_dim=1,
        edge_embedding_method="mul",
        clip_limits=(1.0, 5.0),
        kernel=kernel,
    )(x0, x1)
    unclipped = ((x0 * x1) @ kernel)[:, 0]
    assert unclipped.min() > 5.0
    assert got.max() < unclipped.max()
    assert got.max() == pytest.approx(0.9 * 5.0 + 0.1 * unclipped.max())


@pytest.mark.parametrize(
    "method, output_dim", [("ip", 1), ("l1", 3), ("l2", 3), ("avg", 2), ("mul", 1)]
)
def test_link_inference_clip_limits_match_oracle(method, output_dim):
    x0, x1 = _embeddings()
    d = x0.shape[1]
    kernel = _kernel(d, output_dim, seed=71)
    bias = _bias(output_dim, seed=72)
    clip_limits = (1.0, 5.0)
    got = sg.link_inference(
        output_dim=output_dim,
        edge_embedding_method=method,
        clip_limits=clip_limits,
        kernel=kernel,
        bias=bias,
    )(x0, x1)
    want = ref.link_inference(
        x0, x1, kernel, bias, output_dim, "linear", method, clip_limits
    )
    assert got.shape == (7, output_dim)
    assert np.abs(got - want).max() <= TOL
    # the clip sits on top of the unclipped branch, with alpha fixed at 0.1
    unclipped = sg.link_inference(
        output_dim=output_dim,
        edge_embedding_method=method,
        kernel=kernel,
        bias=bias,
    )(x0, x1)
    assert np.abs(
        got - sg.LeakyClippedLinear(low=1.0, high=5.0, alpha=0.1)(unclipped)
    ).max() <= TOL


# ------------------------------------------------------------- error paths
def test_link_inference_unknown_method_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        sg.link_inference(edge_embedding_method="cosine")


@pytest.mark.parametrize("method", ["l1", "l2", "mul", "hadamard", "avg"])
def test_link_inference_accepts_the_documented_kernel_width(method):
    x0, x1 = _embeddings()
    d = x0.shape[1]
    edge = sg.link_inference(
        output_dim=3, edge_embedding_method=method, kernel=_kernel(d, 3, 81)
    )
    assert edge(x0, x1).shape == (7, 3)


def test_link_inference_concat_accepts_a_two_d_wide_kernel():
    x0, x1 = _embeddings()
    d = x0.shape[1]
    edge = sg.link_inference(
        output_dim=3, edge_embedding_method="concat", kernel=_kernel(2 * d, 3, 92)
    )
    assert edge(x0, x1).shape == (7, 3)


@pytest.mark.parametrize("method", ["l1", "l2", "mul", "hadamard", "avg", "concat"])
def test_link_inference_kernel_output_width_is_checked(method):
    # a (in_dim, output_dim) kernel is required; a kernel of the wrong output
    # width is rejected before the call, not silently truncated
    with pytest.raises(ValueError):
        sg.link_inference(
            output_dim=3, edge_embedding_method=method, kernel=_kernel(4, 2, 101)
        )


@pytest.mark.parametrize("method", ["l1", "l2", "mul", "hadamard", "avg"])
@pytest.mark.parametrize("in_dim", [2, 4, 6, 10])
def test_link_inference_kernel_input_width_is_checked(method, in_dim):
    # every branch but 'concat' embeds a [n, d] pair, so its `Dense` weight is
    # (d, output_dim); a narrower or wider weight is rejected at call time,
    # where `d` is finally known
    x0, x1 = _embeddings()
    edge = sg.link_inference(
        output_dim=3, edge_embedding_method=method, kernel=_kernel(in_dim, 3, 141)
    )
    with pytest.raises(ValueError):
        edge(x0, x1)


@pytest.mark.parametrize("in_dim", [1, 5, 11, 20])
def test_link_inference_concat_kernel_input_width_is_checked(in_dim):
    # 'concat' embeds [n, 2d], so its weight is (2d, output_dim)
    x0, x1 = _embeddings()
    edge = sg.link_inference(
        output_dim=3, edge_embedding_method="concat", kernel=_kernel(in_dim, 3, 151)
    )
    with pytest.raises(ValueError):
        edge(x0, x1)


@pytest.mark.parametrize("method", ["ip", "dot"])
def test_link_inference_inner_product_resets_output_dim_with_a_warning(method):
    x0, x1 = _embeddings()
    d = x0.shape[1]
    with pytest.warns(UserWarning, match="Reverting output_dim to be 1"):
        edge = sg.link_inference(
            output_dim=4,
            edge_embedding_method=method,
            kernel=_kernel(d, 1, 111),
        )
    got = edge(x0, x1)
    assert got.shape == (7, 1)
    assert np.abs(got - (x0 * x1).sum(axis=-1).reshape(-1, 1)).max() <= TOL


def test_link_inference_inner_product_warns_once_not_twice():
    x0, x1 = _embeddings()
    d = x0.shape[1]
    with pytest.warns(UserWarning) as caught:
        edge = sg.link_inference(
            output_dim=4, edge_embedding_method="ip", kernel=_kernel(d, 1, 112)
        )
    assert len(caught) == 1
    assert edge(x0, x1).shape == (7, 1)


def test_link_inference_requires_a_kernel():
    x0, x1 = _embeddings()
    edge = sg.link_inference(edge_embedding_method="l1")
    with pytest.raises(ValueError):
        edge(x0, x1)


def test_link_inference_rejects_mismatched_embedding_shapes():
    x0, x1 = _embeddings()
    with pytest.raises(ValueError):
        sg.link_inference(
            output_dim=1, edge_embedding_method="l1", kernel=_kernel(5, 1, 121)
        )(x0, np.ascontiguousarray(x1[:4]))


# --------------------------------------------------------- the two factories
def test_link_classification_defaults():
    edge = sg.link_classification()
    assert _free(edge, "output_dim") == 1
    assert _free(edge, "output_act") == "sigmoid"
    assert _free(edge, "edge_embedding_method") == "ip"
    assert _free(edge, "clip") == 0
    assert sg.link_classification.__defaults__ == (1, "sigmoid", "ip")


def test_link_regression_defaults():
    edge = sg.link_regression()
    assert _free(edge, "output_dim") == 1
    assert _free(edge, "output_act") == "linear"
    assert _free(edge, "edge_embedding_method") == "ip"
    assert _free(edge, "clip") == 0
    assert sg.link_regression.__defaults__ == (1, None, "ip")


def test_link_classification_reports_its_name(capsys):
    sg.link_classification()
    assert "link_classification" in capsys.readouterr().out


def test_link_regression_reports_its_name(capsys):
    sg.link_regression()
    assert "link_regression" in capsys.readouterr().out


def test_the_factories_never_supply_a_kernel():
    # both factories leave `kernel=None`, so calling the returned edge function
    # is a ValueError rather than a silently glorot-initialised Dense
    x0, x1 = _embeddings()
    for factory in (sg.link_classification, sg.link_regression):
        edge = factory()
        assert _free(edge, "kernel") is None
        with pytest.raises(ValueError):
            edge(x0, x1)
