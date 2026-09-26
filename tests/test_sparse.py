"""Parity tests for `mojostellargraph.sparse`, the `SparseTensor` layout the
two upstream TensorFlow sparse ops are expressed in.

`tf.sparse.softmax` and `tf.sparse.matmul` both require row-major canonical
order and both canonicalize on construction, so the port canonicalizes once,
in the constructor, exactly where TensorFlow does. The oracle is the dense
equivalent, since a `tf.SparseTensor` product is the dense product of the
matrix it denotes.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojostellargraph as sg

ATOL = 1e-12


def _sp(a):
    return sg.SparseTensor.from_dense(np.asarray(a, dtype=np.float64))


def test_from_dense_round_trips():
    a = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    t = _sp(a)
    assert len(t) == 3
    assert t.dense_shape == (3, 3)
    np.testing.assert_array_equal(t.toarray(), a)
    np.testing.assert_array_equal(t.indptr, [0, 2, 3, 3])
    np.testing.assert_array_equal(t.colind, [0, 2, 1])


def test_unsorted_indices_are_canonicalized_row_major():
    """`tf.SparseTensor` does not require sorted indices, but both sparse ops
    do, so the constructor does the conversion once."""
    t = sg.SparseTensor(
        np.array([[2, 1], [0, 3], [2, 0], [0, 1]], dtype=np.int32),
        np.array([4.0, 2.0, 5.0, 1.0]),
        (3, 4),
    )
    np.testing.assert_array_equal(t.sorted_indices, [[0, 1], [0, 3], [2, 0], [2, 1]])
    np.testing.assert_array_equal(t.indptr, [0, 2, 2, 4])
    np.testing.assert_array_equal(t.colind, [1, 3, 0, 1])
    np.testing.assert_allclose(t.canonical_values, [1.0, 2.0, 5.0, 4.0])


def test_duplicate_entries_sum():
    """A `tf.SparseTensor` product sums duplicate `(row, col)` entries."""
    t = sg.SparseTensor(
        np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int32),
        np.array([1.5, 2.5, 3.0]),
        (2, 2),
    )
    np.testing.assert_allclose(t.toarray(), [[0.0, 4.0], [3.0, 0.0]], atol=0.0)


def test_sparse_matmul_dense_matches_the_dense_product():
    r = np.random.default_rng(3)
    a = r.normal(size=(9, 9))
    a[a < 0.6] = 0.0
    dense = r.normal(size=(9, 5))
    t = _sp(a)
    np.testing.assert_allclose(
        sg.sparse_matmul_dense(t, dense), a @ dense, rtol=0, atol=ATOL
    )


def test_sparse_matmul_dense_handles_an_empty_matrix():
    t = _sp(np.zeros((4, 4)))
    out = sg.sparse_matmul_dense(t, np.ones((4, 2)))
    np.testing.assert_array_equal(out, np.zeros((4, 2)))


def test_sparse_matmul_dense_is_not_confused_by_duplicate_edges():
    """The kernel reads the canonical form, so a repeated edge is summed once
    into the CSR rather than being walked twice."""
    t = sg.SparseTensor(
        np.array([[0, 0], [0, 0], [1, 2]], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]),
        (2, 3),
    )
    dense = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        sg.sparse_matmul_dense(t, dense), np.array([[3.0, 3.0], [3.0, 3.0]]), atol=0.0
    )


@pytest.mark.parametrize("bad", [[[0, 3]], [[-1, 0]], [[2, 0]]])
def test_an_index_past_the_dense_shape_is_rejected(bad):
    with pytest.raises(IndexError):
        sg.SparseTensor(np.array(bad, dtype=np.int32), np.array([1.0]), (2, 2))
