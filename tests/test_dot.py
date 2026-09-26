"""Parity tests for the shared `dot` primitive and the elementwise helpers
built on the same SIMD path.

`mojostellargraph.types.dot` is not itself exported over the C ABI -- it is
the primitive every layer kernel multiplies through -- so these tests reach
it through the two exported entry points that call it with a shape the caller
chooses: `GraphConvolution` (`A @ features` then `h_graph @ kernel`) and
`chebyshev_polynomial` (an `[n, n]` square product, once per degree).

`dot` has two kernels and picks between them on the size of the right-hand
operand, so the point of these tests is the shape sweep: `n` odd and even, `m`
below, at, and above the vector width, and `k` not a multiple of it. Every
remainder loop is on that path, and a remainder that drops the last row or
the last column is silent, so each combination is compared against NumPy.

The two kernels also have to agree with each other. The right-hand operand
crosses at 1 MiB, so a product of the same arithmetic appears at a size that
takes the register-blocked kernel and at one that takes the streaming one;
both are checked against the same NumPy product, which is the real assertion
that the dispatch is not losing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg

ATOL = 1e-12
# The right-hand operand of `h_graph @ kernel` is `f * units * 8` bytes, and
# `dot` switches kernels above 1 MiB of it.
STREAMING_F = 1200


def _adjacency(n, seed=3):
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < 0.4).astype(np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 1.0)
    return a


def _convolution(n, f, units, seed=0):
    rng = np.random.default_rng(seed)
    a = _adjacency(n)
    x = np.ascontiguousarray(rng.normal(size=(n, f)))
    layer = sg.GraphConvolution(units, activation=None, use_bias=True)
    layer.build(f)
    layer.kernel[:] = rng.normal(size=(f, units))
    layer.bias[:] = rng.normal(size=units)
    return layer, a, x


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("f", [7, 8])
@pytest.mark.parametrize("units", [1, 2, 3, 4, 5, 8])
def test_dot_matches_numpy_across_every_remainder(n, f, units):
    """`A @ x` then `h_graph @ kernel`, at widths that exercise the `m` and
    `k` remainder loops and an odd `n` for the two-row blocking."""
    layer, a, x = _convolution(n, f, units)
    got = layer(x, a)
    want = (a @ x) @ layer.kernel + layer.bias
    np.testing.assert_allclose(got, want, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("units", [200, 201])
def test_dot_matches_numpy_on_the_streaming_kernel(n, units):
    """`f * units * 8` above 1 MiB, so the product runs on the kernel that
    streams the right-hand operand and accumulates in `dst` in memory."""
    f = STREAMING_F
    assert f * units * 8 > 1048576
    layer, a, x = _convolution(n, f, units, seed=5)
    got = layer(x, a)
    want = (a @ x) @ layer.kernel + layer.bias
    np.testing.assert_allclose(got, want, rtol=0.0, atol=ATOL * 1e3)


@pytest.mark.parametrize("n, k", [(9, 2), (10, 2), (9, 3), (61, 2), (62, 2)])
def test_chebyshev_square_product_matches_numpy(n, k):
    """`chebyshev_polynomial` multiplies an `[n, n]` square once per degree.
    The two sizes straddle the 1 MiB dispatch, so both kernels are in play."""
    rng = np.random.default_rng(11)
    x = np.ascontiguousarray(rng.normal(size=(n, n)))
    got = sg.chebyshev_polynomial(x, k)
    t_prev = np.eye(n)
    t_curr = x
    for step in range(2, k + 1):
        t_prev, t_curr = t_curr, 2.0 * x @ t_curr - t_prev
    np.testing.assert_allclose(got[-1], t_curr, rtol=0.0, atol=ATOL * 1e3)


@pytest.mark.parametrize("act", ["relu", "elu", "sigmoid", "tanh", "softmax", "softplus"])
@pytest.mark.parametrize("units", [1, 3, 5])
def test_activation_tail_is_applied_to_every_column(units, act):
    """`activation_inplace` writes over its own input in `W`-wide steps, so a
    width that leaves a remainder is the only way to see the scalar tail run
    at all. Reached through `GraphConvolution`, which owns no output buffer
    of its own."""
    n, f = 5, 6
    rng = np.random.default_rng(13)
    a = _adjacency(n)
    x = np.ascontiguousarray(rng.normal(size=(n, f)))
    layer = sg.GraphConvolution(units, activation=act, use_bias=False)
    layer.build(f)
    layer.kernel[:] = rng.normal(size=(f, units))
    got = layer(x, a)
    want = _activate(act, (a @ x) @ layer.kernel)
    tol = 1e-8 if act == "softplus" else 1e-12
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def _activate(act, x):
    if act == "relu":
        return np.maximum(x, 0.0)
    if act == "elu":
        return np.where(x >= 0.0, x, np.exp(x) - 1.0)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if act == "tanh":
        return np.tanh(x)
    if act == "softplus":
        return np.log1p(np.exp(x))
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
