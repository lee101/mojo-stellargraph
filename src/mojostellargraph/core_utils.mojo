"""Port of `stellargraph/core/utils.py` (v0.8.1).

This is the GCN-ing / normalisation module: the sparse-matrix transformations
upstream applies to an adjacency before it reaches a GCN, SGC, Chebyshev or
PPNP model. Upstream is SciPy; here the same arithmetic runs over a dense
row-major `[n, n]` float64 block, because that is the layout the C ABI can
carry without a copy.

Upstream signatures are kept. `symmetric=True` becomes an `Int` flag because a
C ABI call cannot carry a Python `bool` alongside the buffers; the branch
itself is unchanged.

`rescale_laplacian` is the one function whose numerics diverge: upstream calls
`scipy.sparse.linalg.eigsh(laplacian, 1, which="LM")` and falls back to
`largest_eigval = 2` on `ArpackNoConvergence`. The eigenvalue is therefore an
argument here and `power_iteration` supplies it.
"""

from std.math import sin, sqrt

from mojostellargraph.types import FPtr, Vec, W, dot

# Upstream dispatches on the `method` string; the C ABI carries a code and
# python/mojostellargraph/core_utils.py maps the string onto it.
comptime METHOD_GCN = 0
comptime METHOD_CHEBYSHEV = 1
comptime METHOD_SGC = 2
comptime METHOD_NONE = 3


def symmetrize(adj: FPtr, dst: FPtr, n: Int):
    """`A + A.T.multiply(A.T > A) - A.multiply(A.T > A)`.

    The idiom upstream uses for "make this matrix symmetric" appears three
    times: in `PPNP_Aadj_feats_op`, in `GCN_Aadj_feats_op` and in
    `GraphPreProcessingLayer.call`. It keeps `max(A, A.T)` in every position
    and is symmetric by construction, so no explicit transpose is needed.
    """
    for i in range(n):
        for j in range(n):
            var a = adj.unsafe_load(i * n + j)
            var t = adj.unsafe_load(j * n + i)
            var keep = 1.0 if t > a else 0.0
            dst.unsafe_store(i * n + j, a + t * keep - a * keep)


def add_self_loops(adj: FPtr, dst: FPtr, n: Int):
    """`adj + sp.diags(np.ones(n) - adj.diagonal())`.

    Upstream spells this out in three places; the arithmetic is the same and is
    factored out here.
    """
    for i in range(n):
        for j in range(n):
            var v = adj.unsafe_load(i * n + j)
            if i == j:
                v += 1.0 - v
            dst.unsafe_store(i * n + j, v)



def _inv_sqrt(x: Float64) -> Float64:
    """`x ** -0.5` with upstream's sparse result for an isolated node: SciPy's
    `diags` product only visits stored entries, so a zero row sum leaves a zero
    row and a zero column, not an `inf * 0 = NaN` row."""
    return 1.0 / sqrt(x) if x > 0.0 else 0.0


def _inv(x: Float64) -> Float64:
    """`x ** -1`, with the same treatment of a zero row sum."""
    return 1.0 / x if x > 0.0 else 0.0


def normalize_adj(adj: FPtr, dst: FPtr, d: FPtr, n: Int, symmetric: Int):
    """Upstream `normalize_adj(adj, symmetric=True)`.

    ```
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    ```

    `adj.sum(1)` is the row sum. The left diagonal scales rows and the right
    one columns, and the transpose in the symmetric branch swaps them, so
    `a_norm[i, j] = adj[j, i] / d[i] / d[j]`. `d` is `n` scratch.
    """
    for i in range(n):
        var s = 0.0
        for j in range(n):
            s += adj.unsafe_load(i * n + j)
        d.unsafe_store(i, s)
    if symmetric:
        for i in range(n):
            var di = _inv_sqrt(d.unsafe_load(i))
            for j in range(n):
                dst.unsafe_store(
                    i * n + j,
                    adj.unsafe_load(j * n + i) * di * _inv_sqrt(d.unsafe_load(j)),
                )
    else:
        for i in range(n):
            var di = _inv(d.unsafe_load(i))
            for j in range(n):
                dst.unsafe_store(i * n + j, di * adj.unsafe_load(i * n + j))


def normalized_laplacian(adj: FPtr, laplacian: FPtr, d: FPtr, n: Int, symmetric: Int):
    """Upstream `normalized_laplacian(adj, symmetric=True)`: `I - A_norm`."""
    normalize_adj(adj, laplacian, d, n, symmetric)
    for i in range(n):
        for j in range(n):
            laplacian.unsafe_store(
                i * n + j,
                (1.0 if i == j else 0.0) - laplacian.unsafe_load(i * n + j),
            )


def preprocess_adj(adj: FPtr, dst: FPtr, work: FPtr, d: FPtr, n: Int, symmetric: Int):
    """The nested `preprocess_adj` in `GCN_Aadj_feats_op`: self loops first,
    then normalization. `work` is the `n * n` intermediate."""
    add_self_loops(adj, work, n)
    normalize_adj(work, dst, d, n, symmetric)


def rescale_laplacian(laplacian: FPtr, dst: FPtr, n: Int, largest_eigval: Float64):
    """Upstream `rescale_laplacian(laplacian)`: `(2 / lambda) * L - I`.

    `largest_eigval` is a parameter here; upstream computes it with
    `eigsh(laplacian, 1, which="LM")` and substitutes `2.0` when ARPACK does
    not converge. Use `power_iteration` for the same quantity.
    """
    # upstream substitutes `largest_eigval = 2` when ARPACK fails to converge,
    # and `power_iteration` reports failure as 0.0; both give a scale of 1.0
    var scale = 2.0 / largest_eigval if largest_eigval > 0.0 else 1.0
    for i in range(n):
        for j in range(n):
            var v = scale * laplacian.unsafe_load(i * n + j)
            if i == j:
                v -= 1.0
            dst.unsafe_store(i * n + j, v)


def power_iteration(
    a: FPtr, work: FPtr, vec: FPtr, n: Int, max_iter: Int, tol: Float64
) -> Float64:
    """Largest-magnitude eigenvalue of a symmetric `a`, by power iteration.

    Stands in for the `eigsh(laplacian, 1, which="LM")` call inside
    `rescale_laplacian`. `work` is `n` scratch. Power iteration is a different
    algorithm from ARPACK, so this is a documented numerical divergence rather
    than a like-for-like port; the eigenvalue it returns is the same one.
    """
    # A constant start vector is an exact null vector of the normalized
    # Laplacian of a regular graph, which would stall the iteration at 0.0.
    for i in range(n):
        vec.unsafe_store(i, sin(Float64(i) + 1.0) / sqrt(Float64(n)))
    var eigval = 0.0
    for _ in range(max_iter):
        dot(a, vec, work, n, n, 1)
        var norm = 0.0
        for i in range(n):
            norm += work.unsafe_load(i) * work.unsafe_load(i)
        norm = sqrt(norm)
        if norm == 0.0:
            return 0.0
        eigval = norm
        var diff = 0.0
        for i in range(n):
            var u = work.unsafe_load(i) / norm
            diff += abs(u - vec.unsafe_load(i))
            vec.unsafe_store(i, u)
        if diff < tol:
            break
    return eigval


def chebyshev_polynomial(x: FPtr, result: FPtr, work: FPtr, n: Int, k: Int):
    """Upstream `chebyshev_polynomial(X, k)`.

    ```
    T_k = [sp.eye(n), X]
    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        return 2 * X.copy().dot(T_k_minus_one) - T_k_minus_two
    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
    ```

    Upstream returns a list of sparse matrices; a C ABI call cannot, so the
    polynomials land consecutively in `out` as `k + 1` blocks of `n * n` and
    the Python side hands them back as a list of views. `work` is one `n * n`
    block used for the `X_.dot(...)` product.
    """
    # `sp.eye(n)`, not a bare diagonal write: `result` is the caller's buffer.
    var z = 0
    while z < n * n:
        result.unsafe_store(z, 0.0)
        z += 1
    for i in range(n):
        result.unsafe_store(i * n + i, 1.0)
    var j = 0
    while j < n * n:
        result.unsafe_store(n * n + j, x.unsafe_load(j))
        j += 1

    for step in range(2, k + 1):
        var prev = (step - 2) * n * n
        var cur = (step - 1) * n * n
        var next = step * n * n
        # `T_k = 2 * X.dot(T_k_minus_one) - T_k_minus_two`, as the shared
        # `dot` and one more pass over the `[n, n]` block
        dot(x, result.unsafe_offset(cur), work, n, n, n)
        for r in range(n):
            var wrow = work.unsafe_offset(r * n)
            var prow = result.unsafe_offset(prev + r * n)
            var nrow = result.unsafe_offset(next + r * n)
            var j = 0
            while j + W <= n:
                nrow.unsafe_store(
                    j,
                    Vec(2.0) * wrow.unsafe_load[width=W](j)
                    - prow.unsafe_load[width=W](j),
                )
                j += W
            while j < n:
                nrow.unsafe_store(j, 2.0 * wrow.unsafe_load(j) - prow.unsafe_load(j))
                j += 1


def invert(a: FPtr, dst: FPtr, work: FPtr, n: Int) -> Int:
    """`np.linalg.inv(a)` for a dense `a`, by Gauss-Jordan with partial
    pivoting on `[a | I]`.

    Stands in for the `np.linalg.inv` inside upstream's `PPNP_Aadj_feats_op`.
    `work` is `2 * n * n` scratch. Returns 0 when a pivot is negligible against
    the largest entry of `a`, which the Python wrapper turns into the
    `LinAlgError` NumPy raises. The test is relative rather than exact because
    elimination leaves a rounding residue where LAPACK gets an exact zero; a
    documented divergence, and a stricter one than `np.linalg.inv` on a
    borderline matrix.
    """
    var w = 2 * n
    for i in range(n):
        for j in range(n):
            work.unsafe_store(i * w + j, a.unsafe_load(i * n + j))
            work.unsafe_store(i * w + n + j, 1.0 if i == j else 0.0)

    var scale = 0.0
    for i in range(n * n):
        scale = max(scale, abs(a.unsafe_load(i)))
    var tol = 1e-15 * scale

    for col in range(n):
        var piv = col
        var best = abs(work.unsafe_load(col * w + col))
        for r in range(col + 1, n):
            var v = abs(work.unsafe_load(r * w + col))
            if v > best:
                best = v
                piv = r
        if best <= tol:
            return 0
        if piv != col:
            for j in range(w):
                var t = work.unsafe_load(col * w + j)
                work.unsafe_store(col * w + j, work.unsafe_load(piv * w + j))
                work.unsafe_store(piv * w + j, t)
        var d = work.unsafe_load(col * w + col)
        var crow = work.unsafe_offset(col * w)
        var j = 0
        while j + W <= w:
            crow.unsafe_store(j, crow.unsafe_load[width=W](j) / d)
            j += W
        while j < w:
            crow.unsafe_store(j, crow.unsafe_load(j) / d)
            j += 1
        for r in range(n):
            if r == col:
                continue
            var f = work.unsafe_load(r * w + col)
            if f == 0.0:
                continue
            var rrow = work.unsafe_offset(r * w)
            j = 0
            while j + W <= w:
                rrow.unsafe_store(
                    j, rrow.unsafe_load[width=W](j) - f * crow.unsafe_load[width=W](j)
                )
                j += W
            while j < w:
                rrow.unsafe_store(j, rrow.unsafe_load(j) - f * crow.unsafe_load(j))
                j += 1

    for i in range(n):
        for j in range(n):
            dst.unsafe_store(i * n + j, work.unsafe_load(i * w + n + j))
    return 1


def PPNP_Aadj_feats_op(
    adj: FPtr,
    result: FPtr,
    tmp: FPtr,
    tmp2: FPtr,
    work: FPtr,
    n: Int,
    teleport_probability: Float64,
) -> Int:
    """Upstream `PPNP_Aadj_feats_op(features, A, teleport_probability=0.1)`.

    ```
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A = A + sp.diags(np.ones(n) - A.diagonal())
    A = normalize_adj(A, symmetric=True)
    A = A.toarray()
    A = teleport_probability * np.linalg.inv(np.eye(n) - (1 - teleport_probability) * A)
    ```

    The features pass through untouched upstream, so only the adjacency comes
    back; the Python wrapper returns `(features, A)` unchanged from the caller's
    point of view.

    Returns 0 when `teleport_probability` is out of range (upstream raises
    `ValueError`) or when the inner matrix is singular (upstream's `np.linalg.inv`
    raises `LinAlgError`).
    """
    if teleport_probability > 1.0 or teleport_probability < 0.0:
        return 0

    symmetrize(adj, tmp, n)
    add_self_loops(tmp, result, n)
    normalize_adj(result, tmp, work, n, 1)
    # tmp = eye(n) - (1 - teleport_probability) * A_norm
    for i in range(n):
        for j in range(n):
            var v = -(1.0 - teleport_probability) * tmp.unsafe_load(i * n + j)
            if i == j:
                v += 1.0
            tmp2.unsafe_store(i * n + j, v)
    if invert(tmp2, result, work, n) == 0:
        return 0
    for i in range(n * n):
        result.unsafe_store(i, teleport_probability * result.unsafe_load(i))
    return 1


def GCN_Aadj_feats_op(
    adj: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    scratch: FPtr,
    cheb_out: FPtr,
    n: Int,
    k: Int,
    method: Int,
) -> Int:
    """Upstream `GCN_Aadj_feats_op(features, A, k=1, method="gcn")`.

    ```
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    if method == "gcn":       A = preprocess_adj(A)
    elif method == "chebyshev": T_k = chebyshev_polynomial(rescale_laplacian(normalized_laplacian(A)), k)
    elif method == "sgc":      A = preprocess_adj(A) ** k
    return features, A
    ```

    `cheb_out` receives `k + 1` blocks of `n * n` for the Chebyshev case; the
    Python side turns that into the `[features] + T_k` list upstream returns.
    Returns 0 for an undefined method (upstream raises `ValueError`).
    """
    symmetrize(adj, result, n)

    if method == METHOD_GCN:
        preprocess_adj(result, work, work2, scratch, n, 1)
        var i = 0
        while i < n * n:
            result.unsafe_store(i, work.unsafe_load(i))
            i += 1
        return 1

    if method == METHOD_CHEBYSHEV:
        if k < 2:
            return 0
        normalized_laplacian(result, work, scratch, n, 1)
        rescale_laplacian(work, work2, n, power_iteration(work, scratch, work2, n, 100, 1e-10))
        chebyshev_polynomial(work2, cheb_out, work, n, k)
        return 1

    if method == METHOD_SGC:
        if k <= 0:
            return 0
        # A = A ** k
        preprocess_adj(result, work2, work, scratch, n, 1)
        var i = 0
        while i < n * n:
            result.unsafe_store(i, work2.unsafe_load(i))
            i += 1
        var step = 1
        while step < k:
            dot(work2, result, work, n, n, n)
            i = 0
            while i < n * n:
                result.unsafe_store(i, work.unsafe_load(i))
                i += 1
            step += 1
        return 1

    if method == METHOD_NONE:
        return 1

    return 0
