"""Parity tests for `mojostellargraph.core_utils` and `GraphPreProcessingLayer`.

Upstream `stellargraph` 0.8.1 needs Python < 3.9 and TensorFlow 2.1, so it
cannot be installed here and `tests/upstream_reference.py` -- a line-by-line
NumPy transliteration of the upstream functions -- is the oracle. Every test
below therefore checks the Mojo port against that oracle, and separately pins
a closed form or a structural identity so agreement with the oracle is not the
only evidence that the kernel is right.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg
import upstream_reference as ref

ATOL = 1e-12
EIG_ATOL = 1e-9
PI_ATOL = 1e-4


def _asymmetric(adj: np.ndarray) -> np.ndarray:
    """A weighted, provably non-symmetric copy: `out[0, 1] - out[1, 0] == 2.0`."""
    out = np.array(adj, dtype=np.float64, copy=True)
    out[0, 1] += 2.0
    return out


@pytest.mark.parametrize("symmetric", [True, False])
def test_normalize_adj_matches_reference(adj_with_loops, sym_graph, symmetric):
    for adj in (adj_with_loops, sym_graph):
        got = sg.normalize_adj(adj, symmetric=symmetric)
        np.testing.assert_allclose(
            got, ref._normalize_adj(adj, symmetric), atol=ATOL, rtol=0.0
        )


def test_normalize_adj_symmetric_closed_form(adj_with_loops):
    # D = diag(adj.sum(1)); upstream returns adj.dot(D^-1/2).T.dot(D^-1/2).
    adj = _asymmetric(adj_with_loops)
    d_inv_sqrt = np.power(adj.sum(1), -0.5)
    closed = (d_inv_sqrt[:, None] * adj.T) * d_inv_sqrt[None, :]
    np.testing.assert_allclose(
        sg.normalize_adj(adj, True), closed, atol=ATOL, rtol=0.0
    )


def test_normalize_adj_asymmetric_closed_form(adj_with_loops):
    # The `symmetric=False` branch is D^-1 A, so every row sums to exactly one.
    got = sg.normalize_adj(adj_with_loops, False)
    d_inv = np.float_power(adj_with_loops.sum(1), -1)
    np.testing.assert_allclose(got, d_inv[:, None] * adj_with_loops, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(got.sum(1), np.ones(adj_with_loops.shape[0]), atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0, 10.0])
@pytest.mark.parametrize("symmetric", [True, False])
def test_normalize_adj_is_invariant_to_scaling_the_adjacency(
    adj_with_loops, scale, symmetric
):
    # The degrees and the `** -0.5` (or `** -1`) both gain the scale factor, so
    # it cancels: a weighted graph normalizes to the same matrix at any scale.
    np.testing.assert_allclose(
        sg.normalize_adj(scale * adj_with_loops, symmetric),
        sg.normalize_adj(adj_with_loops, symmetric),
        atol=ATOL,
        rtol=0.0,
    )


def test_normalize_adj_on_a_weighted_graph_matches_reference(rng):
    w = rng.uniform(0.5, 2.5, size=(9, 9))
    w = np.maximum(w, w.T)
    np.fill_diagonal(w, 0.0)
    for symmetric in (True, False):
        np.testing.assert_allclose(
            sg.normalize_adj(w, symmetric),
            ref._normalize_adj(w, symmetric),
            atol=ATOL,
            rtol=0.0,
        )


# ------------------------------------------------------------ normalized_laplacian
@pytest.mark.parametrize("symmetric", [True, False])
def test_normalized_laplacian_matches_reference(adj_with_loops, sym_graph, symmetric):
    for adj in (adj_with_loops, sym_graph):
        np.testing.assert_allclose(
            sg.normalized_laplacian(adj, symmetric),
            ref._normalized_laplacian(adj, symmetric),
            atol=ATOL,
            rtol=0.0,
        )


@pytest.mark.parametrize("symmetric", [True, False])
def test_normalized_laplacian_is_identity_minus_normalize_adj(
    adj_with_loops, symmetric
):
    lap = sg.normalized_laplacian(adj_with_loops, symmetric)
    np.testing.assert_allclose(
        lap,
        np.eye(adj_with_loops.shape[0]) - sg.normalize_adj(adj_with_loops, symmetric),
        atol=ATOL,
        rtol=0.0,
    )



def test_normalized_laplacian_diagonal_closed_form(adj_with_loops):
    deg = adj_with_loops.sum(1)
    np.testing.assert_allclose(
        np.diag(sg.normalized_laplacian(adj_with_loops)),
        1.0 - np.diag(adj_with_loops) / deg,
        atol=ATOL,
        rtol=0.0,
    )


def test_normalized_laplacian_spectrum_lies_in_zero_to_two(adj_with_loops):
    ev = np.linalg.eigvalsh(sg.normalized_laplacian(adj_with_loops))
    assert ev.min() > -EIG_ATOL
    assert ev.max() < 2.0 + EIG_ATOL


def test_normalized_laplacian_is_symmetric(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    np.testing.assert_array_equal(lap, lap.T)


# ---------------------------------------------------------------- power_iteration
def test_power_iteration_matches_largest_eigenvalue(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    assert abs(sg.power_iteration(lap) - np.linalg.eigvalsh(lap)[-1]) < PI_ATOL


def test_power_iteration_converges_with_a_tighter_tolerance(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    tight = sg.power_iteration(lap, max_iter=1000, tol=1e-14)
    assert abs(tight - np.linalg.eigvalsh(lap)[-1]) < 1e-10


@pytest.mark.parametrize(
    "diag",
    [
        [3.0, 1.0, -2.0],
        [-7.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 5.5, -1.25],
        [2.0],
    ],
)
def test_power_iteration_on_a_diagonal_is_its_largest_magnitude(diag):
    # Closed form: on a diagonal matrix the dominant eigenvalue is the entry of
    # largest magnitude, and power iteration must find it in a single step.
    m = np.diag(np.asarray(diag, dtype=np.float64))
    assert abs(sg.power_iteration(m) - np.abs(diag).max()) < PI_ATOL


@pytest.mark.parametrize("n", [1, 2, 5, 11])
def test_power_iteration_on_the_identity_is_one(n):
    assert abs(sg.power_iteration(np.eye(n)) - 1.0) < 1e-9


def test_power_iteration_returns_a_python_float(adj_with_loops):
    value = sg.power_iteration(sg.normalized_laplacian(adj_with_loops))
    assert isinstance(value, float)
    assert value > 0.0


# -------------------------------------------------------------- rescale_laplacian
def test_rescale_laplacian_matches_reference(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    largest = float(np.linalg.eigvalsh(lap)[-1])
    np.testing.assert_allclose(
        sg.rescale_laplacian(lap, largest), ref._rescale_laplacian(lap, largest), atol=ATOL, rtol=0.0
    )


def test_rescale_laplacian_closed_form(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    largest = 1.5
    closed = (2.0 / largest) * lap - np.eye(lap.shape[0])
    np.testing.assert_allclose(
        sg.rescale_laplacian(lap, largest), closed, atol=ATOL, rtol=0.0
    )


def test_rescale_laplacian_spectrum_lies_in_minus_one_to_one(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    scaled = sg.rescale_laplacian(lap, float(np.linalg.eigvalsh(lap)[-1]))
    ev = np.linalg.eigvalsh(scaled)
    assert ev.min() > -1.0 - EIG_ATOL
    assert ev.max() < 1.0 + EIG_ATOL


def test_rescale_laplacian_default_eigenvalue_is_power_iteration(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    np.testing.assert_array_equal(
        sg.rescale_laplacian(lap), sg.rescale_laplacian(lap, sg.power_iteration(lap))
    )


def test_rescale_laplacian_default_eigenvalue_tracks_the_reference(adj_with_loops):
    lap = sg.normalized_laplacian(adj_with_loops)
    largest = float(np.linalg.eigvalsh(lap)[-1])
    np.testing.assert_allclose(
        sg.rescale_laplacian(lap), ref._rescale_laplacian(lap, largest), atol=EIG_ATOL, rtol=0.0
    )


# ------------------------------------------------------------ chebyshev_polynomial
@pytest.mark.parametrize("k", [1, 2, 3, 4, 6])
def test_chebyshev_polynomial_matches_reference(adj_with_loops, k):
    lap = sg.normalized_laplacian(adj_with_loops)
    scaled = sg.rescale_laplacian(lap, float(np.linalg.eigvalsh(lap)[-1]))
    got = sg.chebyshev_polynomial(scaled, k)
    want = ref._chebyshev_polynomial(scaled, k)
    assert len(got) == k + 1
    for i, (a, b) in enumerate(zip(got, want)):
        np.testing.assert_allclose(a, b, atol=ATOL, rtol=0.0, err_msg="T_{}".format(i))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6])
def test_chebyshev_polynomial_leading_terms(adj_with_loops, k):
    lap = sg.normalized_laplacian(adj_with_loops)
    scaled = sg.rescale_laplacian(lap, float(np.linalg.eigvalsh(lap)[-1]))
    got = sg.chebyshev_polynomial(scaled, k)
    np.testing.assert_array_equal(got[0], np.eye(scaled.shape[0]))
    np.testing.assert_array_equal(got[1], scaled)


@pytest.mark.parametrize("k", [0, 1])
def test_chebyshev_polynomial_of_degree_zero_is_the_identity(adj_with_loops, k):
    # upstream seeds the list as [I, X] and loops from 2, so it hands back at
    # least two matrices whatever k is
    lap = sg.normalized_laplacian(adj_with_loops)
    got = sg.chebyshev_polynomial(lap, k)
    assert len(got) == 2
    np.testing.assert_array_equal(got[0], np.eye(lap.shape[0]))
    np.testing.assert_array_equal(got[1], lap)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_chebyshev_recurrence(adj_with_loops, k):
    # T_k = 2 X T_{k-1} - T_{k-2}, the recurrence upstream loops on.
    lap = sg.normalized_laplacian(adj_with_loops)
    x = sg.rescale_laplacian(lap, float(np.linalg.eigvalsh(lap)[-1]))
    got = sg.chebyshev_polynomial(x, k)
    for i in range(2, k + 1):
        np.testing.assert_allclose(
            got[i], 2.0 * x @ got[i - 1] - got[i - 2], atol=ATOL, rtol=0.0
        )


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6])
def test_chebyshev_polynomials_reproduce_cosine_on_a_diagonal(k):
    # Published identity: T_k(cos t) = cos(k t), and cos(k * arccos(x)) is
    # therefore T_k applied to a matrix whose diagonal holds x.
    x = np.array([0.3, -0.7, 1.0, 0.0, -1.0, 0.5])
    got = sg.chebyshev_polynomial(np.diag(x), k)
    np.testing.assert_allclose(
        np.diag(got[k]), np.cos(k * np.arccos(x)), atol=ATOL, rtol=0.0
    )


def test_chebyshev_polynomials_commute(rng):
    # Every T_k is a polynomial in X, so any two of them commute.
    x = rng.normal(size=(7, 7))
    x = (x + x.T) / 2.0
    ts = sg.chebyshev_polynomial(x, 4)
    np.testing.assert_allclose(ts[3] @ ts[1], ts[1] @ ts[3], atol=1e-11, rtol=0.0)


def test_chebyshev_polynomial_parity(rng):
    # T_k(-X) = (-1)^k T_k(X).
    x = rng.normal(size=(6, 6))
    plus = sg.chebyshev_polynomial(x, 5)
    minus = sg.chebyshev_polynomial(-x, 5)
    for k in range(6):
        np.testing.assert_allclose(minus[k], ((-1) ** k) * plus[k], atol=ATOL, rtol=0.0)


# ---------------------------------------------------------------------- invert
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invert_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(8, 8)) + 3.0 * np.eye(8)
    np.testing.assert_allclose(sg.invert(m), np.linalg.inv(m), atol=ATOL, rtol=0.0)


def test_invert_of_a_symmetric_matrix(adj_with_loops):
    m = adj_with_loops + 0.5 * np.eye(adj_with_loops.shape[0])
    np.testing.assert_allclose(sg.invert(m), np.linalg.inv(m), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(sg.invert(m), sg.invert(m).T, atol=1e-11, rtol=0.0)



def test_invert_is_the_inverse(rng):
    m = rng.normal(size=(6, 6)) + 4.0 * np.eye(6)
    np.testing.assert_allclose(m @ sg.invert(m), np.eye(6), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(sg.invert(m) @ m, np.eye(6), atol=ATOL, rtol=0.0)


def test_invert_of_a_diagonal_matrix_is_the_reciprocal():
    m = np.diag(np.array([2.0, -4.0, 0.5, 8.0]))
    np.testing.assert_allclose(
        sg.invert(m), np.diag(1.0 / np.array([2.0, -4.0, 0.5, 8.0])), atol=ATOL, rtol=0.0
    )


def test_invert_of_the_identity_is_the_identity():
    np.testing.assert_array_equal(sg.invert(np.eye(4)), np.eye(4))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 3)),
        np.array([[1.0, 2.0], [2.0, 4.0]]),
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.ones((4, 4)),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_invert_singular_raises_linalg_error(bad):
    with pytest.raises(np.linalg.LinAlgError):
        sg.invert(bad)


def test_invert_singular_integer_matrix_raises_linalg_error():
    bad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.raises(np.linalg.LinAlgError):
        sg.invert(bad)



# ------------------------------------------------------------- PPNP_Aadj_feats_op
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_ppnp_matches_reference(adj_with_loops, alpha):
    _, got = sg.PPNP_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, teleport_probability=alpha)
    np.testing.assert_allclose(
        got, ref.ppnp_aadj_feats_op(adj_with_loops, alpha), atol=EIG_ATOL, rtol=0.0
    )


@pytest.mark.parametrize("alpha", [0.0, 0.15, 0.4])
def test_ppnp_closed_form(adj_with_loops, features, alpha):
    # alpha * inv(I - (1 - alpha) * normalize_adj(A + I)).
    a_norm = sg.normalize_adj(adj_with_loops, True)
    eye = np.eye(adj_with_loops.shape[0])
    closed = alpha * sg.invert(eye - (1.0 - alpha) * a_norm)
    got_features, got = sg.PPNP_Aadj_feats_op(features, adj_with_loops, teleport_probability=alpha)
    np.testing.assert_allclose(got, closed, atol=EIG_ATOL, rtol=0.0)
    np.testing.assert_array_equal(got_features, features)


def test_ppnp_at_zero_is_the_zero_matrix(adj_with_loops):
    _, got = sg.PPNP_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, teleport_probability=0.0)
    np.testing.assert_array_equal(got, np.zeros_like(got))


def test_ppnp_at_one_is_the_identity(adj_with_loops):
    _, got = sg.PPNP_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, teleport_probability=1.0)
    np.testing.assert_array_equal(got, np.eye(adj_with_loops.shape[0]))


def test_ppnp_result_is_symmetric(adj_with_loops):
    _, got = sg.PPNP_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops)
    np.testing.assert_allclose(got, got.T, atol=EIG_ATOL, rtol=0.0)


@pytest.mark.parametrize("alpha", [-0.001, -0.1, 1.0001, 1.5, 2.0])
def test_ppnp_rejects_out_of_range_teleport_probability(adj_with_loops, alpha):
    with pytest.raises(ValueError):
        sg.PPNP_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, teleport_probability=alpha)


# -------------------------------------------------------------- GCN_Aadj_feats_op
@pytest.mark.parametrize("k", [1, 2])
def test_gcn_gcn_matches_reference(adj_with_loops, k):
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, k=k, method="gcn")
    np.testing.assert_allclose(
        got, ref.gcn_aadj_feats_op(adj_with_loops, k, "gcn"), atol=ATOL, rtol=0.0
    )


@pytest.mark.parametrize("with_loops", [True, False])
def test_gcn_gcn_diagonal_closed_form(adj_with_loops, sym_graph, with_loops):
    # preprocess_adj normalizes `symmetrize(A) + I - diag(symmetrize(A))`, so
    # the diagonal of the output is a'_ii / d'_i on the looped degrees d'.
    adj = adj_with_loops if with_loops else sym_graph
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj, method="gcn")
    sym = np.maximum(adj, adj.T)
    looped = sym + np.diag(1.0 - np.diag(sym))
    deg = looped.sum(1)
    np.testing.assert_allclose(np.diag(got), np.diag(looped) / deg, atol=ATOL, rtol=0.0)
    assert np.all(np.diag(got) > 0.0), "preprocess_adj adds a self loop everywhere"


def test_gcn_gcn_adds_self_loops_where_the_graph_has_none(sym_graph):
    # A loop-free input still gets a full self loop on every node, so the
    # output diagonal is 1 / d_i rather than zero; contrast `method="none"`.
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), sym_graph, method="gcn")
    deg = (sym_graph + np.eye(sym_graph.shape[0])).sum(1)
    np.testing.assert_allclose(np.diag(got), 1.0 / deg, atol=ATOL, rtol=0.0)
    _, raw = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), sym_graph, method="none")
    np.testing.assert_array_equal(np.diag(raw), np.zeros(sym_graph.shape[0]))


def test_gcn_gcn_is_symmetric(adj_with_loops):
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, method="gcn")
    np.testing.assert_array_equal(got, got.T)



@pytest.mark.parametrize("k", [1, 2])
def test_gcn_sgc_matches_reference(adj_with_loops, k):
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, k=k, method="sgc")
    np.testing.assert_allclose(
        got, ref.gcn_aadj_feats_op(adj_with_loops, k, "sgc"), atol=ATOL, rtol=0.0
    )


@pytest.mark.parametrize("k", [1, 2])
def test_gcn_sgc_is_the_kth_power_of_the_normalized_adjacency(adj_with_loops, k):
    a_norm = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, method="gcn")[1]
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, k=k, method="sgc")
    np.testing.assert_allclose(got, np.linalg.matrix_power(a_norm, k), atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_gcn_sgc_kth_power(adj_with_loops, k):
    """Upstream is `A = A ** k` after `preprocess_adj`, not repeated squaring."""
    a_norm = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, method="gcn")[1]
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, k=k, method="sgc")
    np.testing.assert_allclose(got, np.linalg.matrix_power(a_norm, k), atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_gcn_chebyshev_stack_matches_reference(adj_with_loops, k):
    stack, out = sg.GCN_Aadj_feats_op(
        np.zeros((1, 1)), adj_with_loops, k=k, method="chebyshev"
    )
    # upstream returns `[features] + T_k` and leaves the adjacency alone.
    assert len(stack) == k + 2
    np.testing.assert_array_equal(out, np.maximum(adj_with_loops, adj_with_loops.T))
    for i, want in enumerate(ref.gcn_aadj_feats_op(adj_with_loops, k, "chebyshev")):
        np.testing.assert_allclose(stack[i + 1], want, atol=EIG_ATOL, rtol=0.0)


def test_gcn_chebyshev_leading_entries(adj_with_loops, features):
    stack, out = sg.GCN_Aadj_feats_op(features, adj_with_loops, k=2, method="chebyshev")
    # upstream returns `[features] + T_k`; the adjacency stays the symmetrized A.
    np.testing.assert_array_equal(stack[0], features)
    np.testing.assert_array_equal(stack[1], np.eye(adj_with_loops.shape[0]))
    np.testing.assert_array_equal(out, np.maximum(adj_with_loops, adj_with_loops.T))


def test_gcn_chebyshev_first_block_is_the_rescaled_laplacian(adj_with_loops):
    stack, _ = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, k=2, method="chebyshev")
    lap = sg.normalized_laplacian(adj_with_loops)
    np.testing.assert_allclose(
        stack[2], sg.rescale_laplacian(lap, sg.power_iteration(lap)), atol=EIG_ATOL, rtol=0.0
    )


@pytest.mark.parametrize("method", ["gcn", "sgc", "none", None])
def test_gcn_returns_features_unchanged(adj_with_loops, features, method):
    got, _ = sg.GCN_Aadj_feats_op(features, adj_with_loops, k=2, method=method)
    np.testing.assert_array_equal(got, features)


def test_gcn_none_returns_the_symmetrized_adjacency(adj_with_loops):
    # `_symmetrize` keeps the larger of the two mirrored entries, which for a
    # 0/1 adjacency is exactly elementwise `maximum`.
    adj = _asymmetric(adj_with_loops)
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj, method="none")
    np.testing.assert_array_equal(got, np.maximum(adj, adj.T))
    np.testing.assert_allclose(got, ref._symmetrize(adj), atol=ATOL, rtol=0.0)


def test_gcn_none_does_not_normalize_or_loop(adj_with_loops):
    _, got = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, method="none")
    assert np.array_equal(got != 0.0, adj_with_loops != 0.0)
    np.testing.assert_array_equal(got, adj_with_loops)


def test_gcn_default_method_is_gcn(adj_with_loops):
    _, default = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops)
    _, explicit = sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, method="gcn")
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize(
    "method,k",
    [
        ("chebyshev", 1),
        ("chebyshev", 0),
        ("chebyshev", -2),
        ("sgc", 0),
        ("sgc", -1),
        ("bogus", 1),
        ("GCN", 1),
        ("gat", 1),
    ],
)
def test_gcn_rejects_bad_method_or_k(adj_with_loops, method, k):
    with pytest.raises(ValueError):
        sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, k=k, method=method)


# --------------------------------------------------------- GraphPreProcessingLayer
@pytest.mark.parametrize("use_loops", [True, False])
def test_preprocessing_layer_matches_reference(adj_with_loops, sym_graph, use_loops):
    adj = adj_with_loops if use_loops else sym_graph
    layer = sg.GraphPreProcessingLayer(adj.shape[0])
    np.testing.assert_allclose(
        layer(adj), ref.graph_pre_processing_layer(adj), atol=ATOL, rtol=0.0
    )


def test_preprocessing_layer_closed_form(adj_with_loops):
    adj = _asymmetric(adj_with_loops)
    sym = np.maximum(adj, adj.T)
    looped = sym + np.diag(1.0 - np.diag(sym))
    d = 1.0 / np.sqrt(looped.sum(1))
    np.testing.assert_allclose(
        sg.GraphPreProcessingLayer(adj.shape[0])(adj),
        (d[:, None] * looped) * d[None, :],
        atol=ATOL,
        rtol=0.0,
    )


@pytest.mark.parametrize("k", [1, 2])
def test_preprocessing_layer_equals_the_gcn_op(adj_with_loops, k):
    n = adj_with_loops.shape[0]
    layer = sg.GraphPreProcessingLayer(n)
    np.testing.assert_array_equal(
        layer(adj_with_loops), sg.GCN_Aadj_feats_op(np.zeros((1, 1)), adj_with_loops, method="gcn")[1]
    )


def test_preprocessing_layer_is_a_no_op_on_an_already_looped_adjacency(adj_with_loops):
    n = adj_with_loops.shape[0]
    layer = sg.GraphPreProcessingLayer(n)
    np.testing.assert_array_equal(
        layer(adj_with_loops), sg.normalize_adj(adj_with_loops, True)
    )


def test_preprocessing_layer_output_dims_and_shape(adj_with_loops):
    layer = sg.GraphPreProcessingLayer(24)
    assert layer.output_dims == (24, 24)
    got = layer(adj_with_loops)
    assert got.shape == (24, 24)
    np.testing.assert_array_equal(got, got.T)


def test_preprocessing_layer_row_sums_closed_form(adj_with_loops):
    # Row i of D^-1/2 (A + I) D^-1/2 sums to
    # d_i^-1/2 * sum_j a_ij d_j^-1/2, which is 1 exactly for a regular graph.
    n = adj_with_loops.shape[0]
    got = sg.GraphPreProcessingLayer(n)(adj_with_loops)
    d = adj_with_loops.sum(1)
    expected = (adj_with_loops / np.sqrt(d)[None, :]).sum(1) / np.sqrt(d)
    np.testing.assert_allclose(got.sum(1), expected, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(got.sum(1), got.T.sum(0), atol=ATOL, rtol=0.0)


def test_preprocessing_layer_call_and_dunder_agree(adj_with_loops):
    layer = sg.GraphPreProcessingLayer(adj_with_loops.shape[0])
    np.testing.assert_array_equal(layer(adj_with_loops), layer.call(adj_with_loops))


# ------------------------------------------------- degenerate-input guards


def test_normalize_adj_leaves_an_isolated_node_at_zero():
    """`np.float_power(0, -0.5)` is `inf`, and `inf * 0` is NaN. SciPy's
    `diags(...) .dot(adj)` only visits stored entries, so an isolated node
    leaves a zero row and a zero column, not a NaN one."""
    adj = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    got = sg.normalize_adj(adj, symmetric=True)
    assert np.isfinite(got).all()
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got[3], 0.0, atol=0.0)
    np.testing.assert_allclose(got[:, 3], 0.0, atol=0.0)
    # the rest of the matrix is unchanged by the isolated node
    np.testing.assert_allclose(
        got[:3, :3], sg.normalize_adj(adj[:3, :3], symmetric=True), atol=ATOL
    )


def test_normalize_adj_left_only_leaves_an_isolated_node_at_zero():
    adj = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    got = sg.normalize_adj(adj, symmetric=False)
    assert np.isfinite(got).all()


def test_chebyshev_T0_is_the_identity_whatever_the_buffer_held():
    """`T_0` is `sp.eye(n)`, so every off-diagonal entry is written, not just
    the diagonal: `result` belongs to the caller."""
    from mojostellargraph._lib import addr, f64, lib

    n, k = 4, 3
    x = np.random.default_rng(1).normal(size=(n, n))
    x = np.ascontiguousarray((x + x.T) / 2.0)
    # the polynomials land consecutively: `k + 1` blocks of `n * n`
    xf = f64(x)
    rf = np.full((k + 1) * n * n, -7.0)
    wf = np.zeros(n * n)
    lib().msg_chebyshev_polynomial(addr(xf), addr(rf), addr(wf), n, k)
    t0 = rf[: n * n].reshape(n, n)
    np.testing.assert_array_equal(t0, np.eye(n))


def test_rescale_laplacian_survives_a_failed_eigensolve():
    """Upstream substitutes `largest_eigval = 2` when ARPACK does not
    converge, which is a scale of 1.0. `power_iteration` reports failure as
    0.0, and `2.0 / 0.0` would be an all-infinite matrix."""
    lap = np.array([[1.0, -1.0], [-1.0, 1.0]])
    got = sg.rescale_laplacian(lap, largest_eigval=0.0)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, sg.rescale_laplacian(lap, largest_eigval=2.0))


def test_gcn_aadj_feats_op_chebyshev_is_finite_on_a_two_node_graph():
    """A two-node graph's normalized Laplacian is regular, which is the case
    a constant power-iteration start vector stalls on."""
    adj = np.array([[0.0, 1.0], [1.0, 0.0]])
    feats = np.arange(4.0).reshape(2, 2)
    out, _ = sg.GCN_Aadj_feats_op(feats, adj, method="chebyshev", k=2)
    assert all(np.isfinite(t).all() for t in out)
