"""Sparse adjacency helpers, over the same CSR layout the Mojo kernels take.

`stellargraph` builds a SciPy COO adjacency and hands it to `tf.SparseTensor`.
`tf.sparse.softmax` and `tf.sparse.matmul` both require row-major canonical
order, so the conversion happens here, once, where upstream TensorFlow does it
inside the two ops.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i32, lib


class SparseTensor:
    """A `[n, n]` sparse matrix in the `(indices, values, dense_shape)` form
    `tf.SparseTensor` takes, with the canonical CSR ordering the two sparse ops
    need alongside it.

    Attributes:
        indices: `[e, 2]` int32 row/column pairs, in the order given.
        rows, cols: `indices[:, 0]` and `indices[:, 1]` as contiguous int32
            runs, in canonical order.
        values: `[e]` float64, parallel to `indices`.
        dense_shape: `(n, n)`.
        sorted_indices: `indices` in canonical (row-major, then column) order,
            which is the order `tf.sparse.softmax` and `tf.sparse.matmul` see.
        indptr: `[n + 1]` int32 row pointer of the canonical ordering.
        colind: `[e]` int32 column indices in canonical ordering.
        canonical_values: `[e]` float64 values in canonical ordering.
    """

    def __init__(self, indices, values, dense_shape):
        raw = i32(indices).reshape(-1, 2)
        self.dense_shape = tuple(int(x) for x in dense_shape)
        n = self.dense_shape[0]
        ncols = self.dense_shape[1] if len(self.dense_shape) > 1 else n
        vals = f64(values).reshape(-1)
        e = raw.shape[0]
        order = np.lexsort((raw[:, 1], raw[:, 0])) if e else np.zeros(0, np.int64)
        self.sorted_indices = np.ascontiguousarray(raw[order])
        # The two columns are split into their own contiguous buffers: a
        # column of an [e, 2] array is strided, and the kernel reads it as a
        # flat int32 run.
        self.rows = np.ascontiguousarray(self.sorted_indices[:, 0])
        self.cols = np.ascontiguousarray(self.sorted_indices[:, 1])

        self.indptr = np.zeros(n + 1, dtype=np.int32)
        self.colind = np.zeros(max(e, 1), dtype=np.int32)
        self.canonical_values = np.zeros(max(e, 1), dtype=np.float64)
        cursor = np.zeros(n + 1, dtype=np.int32)
        # the reordered values are a temporary: bind them to a local so the
        # buffer outlives the FFI call
        canonical = np.ascontiguousarray(vals[order])
        rc = lib().msg_coo_to_csr(
            addr(self.rows), addr(self.cols), addr(canonical),
            addr(self.indptr), addr(self.colind), addr(self.canonical_values),
            addr(cursor), e, n, ncols,
        )
        if rc == 0 and e:
            raise IndexError(
                "sparse index out of range for a matrix of shape {}".format(
                    self.dense_shape
                )
            )

    @property
    def indices(self) -> np.ndarray:
        return self.sorted_indices

    @classmethod
    def from_dense(cls, a):
        a = f64(a)
        rows, cols = np.nonzero(a)
        return cls(np.stack([rows, cols], axis=1), a[rows, cols], a.shape)

    @classmethod
    def from_scipy(cls, a):
        a = a.tocoo()
        return cls(np.stack([a.row, a.col], axis=1), a.data, a.shape)

    def toarray(self) -> np.ndarray:
        # duplicate (row, col) entries sum, as a tf.SparseTensor product does
        out = np.zeros(self.dense_shape, dtype=np.float64)
        np.add.at(out, (self.indices[:, 0], self.indices[:, 1]), self.canonical_values)
        return out

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __repr__(self) -> str:
        return "SparseTensor(edges={}, dense_shape={})".format(
            len(self), self.dense_shape
        )


def sparse_matmul_dense(a: SparseTensor, dense: np.ndarray) -> np.ndarray:
    """`tf.sparse.matmul(a, dense)` for a dense `[a.shape[1], d]` right-hand side."""
    dense = f64(dense)
    rows, cols = a.dense_shape
    if dense.shape[0] != cols:
        raise ValueError(
            "right-hand side has {} rows, the matrix has {} columns".format(
                dense.shape[0], cols
            )
        )
    d = dense.shape[1]
    dst = np.zeros((rows, d), dtype=np.float64)
    rc = lib().msg_sparse_dense_matmul(
        addr(a.indptr), addr(a.colind), addr(a.canonical_values),
        addr(dense), addr(dst), rows, d, dense.shape[0],
    )
    if rc == 0:
        raise IndexError(
            "sparse column index outside the {} rows of the right-hand side".format(
                dense.shape[0]
            )
        )
    return dst
