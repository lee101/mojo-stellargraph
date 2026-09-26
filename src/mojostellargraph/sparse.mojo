"""Sparse kernels the layer code needs, over the CSR layout.

`stellargraph` hands its sparse adjacency around as a `tf.SparseTensor`
(`indices`, `values`, `dense_shape`) and lets `tf.sparse.softmax` and
`tf.sparse.matmul` do the work. TensorFlow canonicalizes to row-major inside
those two ops, so the same canonical form is what these kernels take, and the
Python side does the reordering once.

There is no `stellargraph` module of its own here: this file exists so the
layer ports can be read side by side with the Keras originals, and every
function in it is the array backend behind one upstream TensorFlow call.
"""

from std.math import exp

from mojostellargraph.types import FPtr, IPtr, iget, iput


def sparse_softmax(indptr: IPtr, values: FPtr, dst: FPtr, n: Int):
    """`tf.sparse.softmax` on a row-major `SparseTensor`: softmax down each
    row over the stored entries only."""
    for r in range(n):
        var lo = Int(iget(indptr, r))
        var hi = Int(iget(indptr, r + 1))
        var m = -1.7976931348623157e308
        for k in range(lo, hi):
            m = max(m, values.unsafe_load(k))
        var s = 0.0
        for k in range(lo, hi):
            s += exp(values.unsafe_load(k) - m)
        for k in range(lo, hi):
            dst.unsafe_store(k, exp(values.unsafe_load(k) - m) / s)


def sparse_dense_matmul(
    indptr: IPtr,
    colind: IPtr,
    values: FPtr,
    dense: FPtr,
    dst: FPtr,
    n: Int,
    d: Int,
    dense_rows: Int,
):
    """`tf.sparse.matmul(a, b)` with `a` in row-major `SparseTensor` form,
    `b` dense `[dense_rows, d]` and a result dense `[n, d]`.

    `dense_rows` is `a`'s column count, which is the row stride of `b` and is
    not `d` unless the product happens to be square."""
    for r in range(n):
        for c in range(d):
            var acc = 0.0
            for k in range(
                Int(iget(indptr, r)), Int(iget(indptr, r + 1))
            ):
                acc += values.unsafe_load(k) * dense.unsafe_load(
                    Int(iget(colind, k)) * dense_rows + c
                )
            dst.unsafe_store(r * d + c, acc)


def sparse_dense_matvec(
    indptr: IPtr, colind: IPtr, values: FPtr, x: FPtr, dst: FPtr, n: Int
):
    """`tf.sparse.matmul(a, x)` for a dense rank-1 `x`."""
    for r in range(n):
        var acc = 0.0
        for k in range(
            Int(iget(indptr, r)), Int(iget(indptr, r + 1))
        ):
            acc += values.unsafe_load(k) * x.unsafe_load(Int(iget(colind, k)))
        dst.unsafe_store(r, acc)


def coo_to_csr(
    rows: IPtr,
    cols: IPtr,
    values: FPtr,
    indptr: IPtr,
    colind: IPtr,
    result: FPtr,
    cursor: IPtr,
    e: Int,
    n: Int,
    ncols: Int,
) -> Int:
    """Counting sort of `[e, 2]` row-major COO into canonical row-major CSR.

    `tf.SparseTensor` does not require its `indices` to be sorted, and both
    `tf.sparse.softmax` and `tf.sparse.matmul` do, so this is the work
    `SqueezedSparseConversion` and the two sparse ops do between them. `cursor`
    is `n + 1` scratch. The placement pass walks the input in order, so
    duplicate `(row, col)` pairs keep their relative order exactly as a
    canonicalizing `tf.SparseTensor` would.

    Returns 0 if an index is outside the dense shape, and 1 otherwise.
    """
    var i = 0
    while i <= n:
        iput(indptr, i, 0)
        i += 1
    # A row index goes straight into `indptr[r + 1]`, and a column index is a
    # node the sparse kernels gather, so both are range-checked here, in the
    # counting pass: it is the only pass that must finish before any pointer
    # is used.
    for k in range(e):
        var r = Int(rows.unsafe_load(k))
        var c = Int(cols.unsafe_load(k))
        if r < 0 or r >= n or c < 0 or c >= ncols:
            return 0
        iput(indptr, r + 1, iget(indptr, r + 1) + 1)
    for c in range(1, n + 1):
        iput(indptr, c, iget(indptr, c) + iget(indptr, c - 1))

    var j = 0
    while j <= n:
        iput(cursor, j, iget(indptr, j))
        j += 1
    for k in range(e):
        var r = Int(rows.unsafe_load(k))
        var at = iget(cursor, r)
        iput(colind, at, Int(cols.unsafe_load(k)))
        result.unsafe_store(at, values.unsafe_load(k))
        iput(cursor, r, at + 1)
    return 1


def csr_row_sums(indptr: IPtr, colind: IPtr, values: FPtr, dst: FPtr, n: Int):
    """`A.sum(1)` for a CSR matrix, as a dense `[n]` vector."""
    for r in range(n):
        var acc = 0.0
        for k in range(
            Int(iget(indptr, r)), Int(iget(indptr, r + 1))
        ):
            acc += values.unsafe_load(k)
        dst.unsafe_store(r, acc)
