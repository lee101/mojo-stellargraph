"""Port of `stellargraph/core/utils.py` (upstream 1.2.1).

This is the GCN-ing / normalisation module: the sparse-matrix transformations
upstream applies to an adjacency before it reaches a GCN, SGC or PPNP model.
Upstream is SciPy; here the same arithmetic runs over a dense row-major
`[n, n]` float64 block, because that is the layout the C ABI can carry without
a copy.

Upstream signatures are kept. `symmetric=True` becomes an `Int` flag because a
C ABI call cannot carry a Python `bool` alongside the buffers; the branch
itself is unchanged.

`chebyshev_polynomial` and `GCN_Aadj_feats_op(method="chebyshev")` existed in
v0.8.1 but were removed upstream in 1.2.1, which raises `ValueError` for that
method, so neither is ported.

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
comptime METHOD_SGC = 2
comptime METHOD_NONE = 3

# Columns of a row are gathered `TB` at a time wherever the kernel reads
# `adj[j, i]`: one column is `W` separate cache lines, so a block of `TB`
# is `TB * W` lines and only `W` of them are touched per step. A scalar
# walk down one column at a time misses on every element.
comptime TB = 32


def symmetrize(adj: FPtr, dst: FPtr, n: Int):
    """`A + A.T.multiply(A.T > A) - A.multiply(A.T > A)`.

    The idiom upstream uses for "make this matrix symmetric" appears three
    times: in `PPNP_Aadj_feats_op`, in `GCN_Aadj_feats_op` and in
    `GraphPreProcessingLayer.call`. It keeps `max(A, A.T)` in every position
    and is symmetric by construction, so no explicit transpose is needed.

    Rows are held `W` at a time: `adj[i, j]` is then a contiguous vector load
    and `adj[j, i]` a strided gather, so the per-element `keep` upstream needs
    becomes a free `max`.
    """
    var i = 0
    while i + W <= n:
        _symmetrize_block[W](adj, dst, i, n)
        i += W
    while i < n:
        _symmetrize_block[1](adj, dst, i, n)
        i += 1


def _symmetrize_block[rows: Int](adj: FPtr, dst: FPtr, i0: Int, n: Int):
    """`symmetrize` for the `rows` rows starting at row `i0`, held at once.

    `rows` is a compile-time width so the `adj[i, j]` side is a vector load
    and the `adj[j, i]` side a gather, which is what turns the per-element
    `keep` upstream needs into a free `max`.
    """
    var r = 0
    while r < rows:
        var arow = adj.unsafe_offset((i0 + r) * n)
        var drow = dst.unsafe_offset((i0 + r) * n)
        # dst[i0 + r, j] = max(adj[i0 + r, j], adj[j, i0 + r]) for every j
        var j = 0
        while j + rows <= n:
            var tv = SIMD[DType.float64, rows]()
            var t = 0
            while t < rows:
                tv[t] = adj.unsafe_load((j + t) * n + i0 + r)
                t += 1
            drow.unsafe_store(j, max(arow.unsafe_load[width=rows](j), tv))
            j += rows
        while j < n:
            drow.unsafe_store(j, max(arow.unsafe_load(j), adj.unsafe_load(j * n + i0 + r)))
            j += 1
        r += 1


def add_self_loops(adj: FPtr, dst: FPtr, n: Int):
    """`adj + sp.diags(np.ones(n) - adj.diagonal())`.

    Upstream spells this out in three places; the arithmetic is the same and is
    factored out here. Only the `n` diagonal entries change, so each row is a
    copy and the diagonal is overwritten rather than a conditional add per
    element.
    """
    for i in range(n):
        var arow = adj.unsafe_offset(i * n)
        var drow = dst.unsafe_offset(i * n)
        var j = 0
        while j + W <= n:
            drow.unsafe_store(j, arow.unsafe_load[width=W](j))
            j += W
        while j < n:
            drow.unsafe_store(j, arow.unsafe_load(j))
            j += 1
    var j = 0
    while j < n:
        dst.unsafe_store(j * n + j, 1.0)
        j += 1



def _inv_sqrt(x: Float64) -> Float64:
    """`x ** -0.5` with upstream's sparse result for an isolated node: SciPy's
    `diags` product only visits stored entries, so a zero row sum leaves a zero
    row and a zero column, not an `inf * 0 = NaN` row."""
    return 1.0 / sqrt(x) if x > 0.0 else 0.0


def _inv(x: Float64) -> Float64:
    """`x ** -1`, with the same treatment of a zero row sum."""
    return 1.0 / x if x > 0.0 else 0.0


def _sub_identity(dst: FPtr, n: Int):
    """`I - dst`, in place: `normalized_laplacian`'s `1 - A_norm`.

    Every entry is negated and the diagonal gets `+1`. It is stepped one
    element at a time because the `n` diagonal entries are `n` doubles apart,
    and a `W`-wide store at `k * n + k` would run on into the next row.
    """
    var i = 0
    while i < n:
        var drow = dst.unsafe_offset(i * n)
        var j = 0
        while j + W <= n:
            drow.unsafe_store(j, drow.unsafe_load[width=W](j) * Vec(-1.0))
            j += W
        while j < n:
            drow.unsafe_store(j, -drow.unsafe_load(j))
            j += 1
        dst.unsafe_store(i * n + i, dst.unsafe_load(i * n + i) + 1.0)
        i += 1


def normalize_adj(adj: FPtr, dst: FPtr, d: FPtr, n: Int, symmetric: Int, sub_identity: Int = 0):
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
    `a_norm[i, j] = adj[j, i] / d[i] / d[j]`. `d` is `n` scratch and is
    overwritten with the per-node scale once the sums are in it.

    `sub_identity` subtracts the identity on the way out, which is
    `normalized_laplacian`'s `I - A_norm` folded into the same pass.
    """
    for i in range(n):
        var s = Vec(0.0)
        var j = 0
        while j + W <= n:
            s += adj.unsafe_load[width=W](i * n + j)
            j += W
        # `s` is a W-wide register, so the tail goes into its own scalar sum:
        # adding a `Float64` to a `SIMD` broadcasts it and counts it W times.
        var tail = s.reduce_add()
        while j < n:
            tail += adj.unsafe_load(i * n + j)
            j += 1
        d.unsafe_store(i, tail)
    if symmetric:
        # `d ** -0.5`, one square root per node rather than one per element.
        for i in range(n):
            d.unsafe_store(i, _inv_sqrt(d.unsafe_load(i)))
        # a_norm[i, j] = adj[j, i] * d[i] * d[j], with `1 - a_norm` folded in
        var jb = 0
        while jb + TB <= n:
            _normalize_block[TB](adj, dst, d, jb, n, sub_identity)
            jb += TB
        var rest = n - jb
        if rest > 0:
            _normalize_tail(adj, dst, d, jb, rest, n, sub_identity)
    else:
        for i in range(n):
            d.unsafe_store(i, _inv(d.unsafe_load(i)))
        for i in range(n):
            var arow = adj.unsafe_offset(i * n)
            var drow = dst.unsafe_offset(i * n)
            var di = Vec(d.unsafe_load(i))
            var j = 0
            while j + W <= n:
                drow.unsafe_store(j, arow.unsafe_load[width=W](j) * di)
                j += W
            while j < n:
                drow.unsafe_store(j, arow.unsafe_load(j) * d.unsafe_load(i))
                j += 1
        if sub_identity:
            _sub_identity(dst, n)


def _normalize_block[cols: Int](
    adj: FPtr, dst: FPtr, d: FPtr, jb: Int, n: Int, sub_identity: Int
):
    """One `cols`-wide column block of `normalize_adj`'s symmetric branch.

    `dst[i, jb + j] = adj[jb + j, i] * d[i] * d[jb + j]` for `j` in `[0, cols)`,
    with `d` already holding the `** -0.5` scales. `sub_identity` negates the
    result and puts `1` on the diagonal, which is `normalized_laplacian`'s
    `I - A_norm`; doing it here saves the second read and write of `n * n`
    doubles that a separate pass would cost.

    The transposed read `adj[jb + j, i]` is a gather with stride `n`, so there
    is nothing contiguous to widen: what this buys over a plain double loop is
    the precomputed scale vector and the absent per-element `sqrt`.
    """
    var i = 0
    while i < n:
        var drow = dst.unsafe_offset(i * n + jb)
        var di = d.unsafe_load(i)
        var j = 0
        while j < cols:
            var v = adj.unsafe_load((jb + j) * n + i) * di * d.unsafe_load(jb + j)
            if sub_identity:
                v = -v
                if i == jb + j:
                    v += 1.0
            drow.unsafe_store(j, v)
            j += 1
        i += 1


def _normalize_tail(
    adj: FPtr, dst: FPtr, d: FPtr, jb: Int, rest: Int, n: Int, sub_identity: Int
):
    """The fewer-than-TB columns at the right edge."""
    var i = 0
    while i < n:
        var drow = dst.unsafe_offset(i * n + jb)
        var di = d.unsafe_load(i)
        var j = 0
        while j < rest:
            var v = adj.unsafe_load((jb + j) * n + i) * di * d.unsafe_load(jb + j)
            if sub_identity:
                v = -v
                if i == jb + j:
                    v += 1.0
            drow.unsafe_store(j, v)
            j += 1
        i += 1


def normalized_laplacian(adj: FPtr, laplacian: FPtr, d: FPtr, n: Int, symmetric: Int):
    """Upstream `normalized_laplacian(adj, symmetric=True)`: `I - A_norm`.

    The `1 - A_norm` is fused into `normalize_adj`'s output rather than run as a
    second pass over the whole block: `A_norm` is not read anywhere else, so
    storing it and reading it back is `n * n` doubles of traffic for nothing.
    """
    normalize_adj(adj, laplacian, d, n, symmetric, 1)


def calculate_laplacian(adj: FPtr, dst: FPtr, d: FPtr, n: Int):
    """Upstream `calculate_laplacian(adj)`.

    ```
    D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
    adj = np.dot(D, np.dot(adj, D))
    ```

    The degree is the COLUMN sum here (`sum(axis=0)`), where
    `normalize_adj` uses the row sum (`sum(1)`). The two differ for a
    non-symmetric adjacency, which is why this is a separate function upstream
    and not a call to `normalize_adj`.
    """
    # `adj.sum(axis=0)` is the column sum. Summing `W` columns into one register
    # and reducing it would add the columns together, not accumulate each
    # column down its rows, so the column sums stay a scalar walk: `adj[i, j]`
    # for consecutive `i` strides by `n`. The `** -0.5` and the scaling pass
    # below are what this kernel spends its time on, and both vectorize.
    for j in range(n):
        var t = 0.0
        var r = 0
        while r < n:
            t += adj.unsafe_load(r * n + j)
            r += 1
        d.unsafe_store(j, t)
    for i in range(n):
        d.unsafe_store(i, _inv_sqrt(d.unsafe_load(i)))
    for i in range(n):
        var arow = adj.unsafe_offset(i * n)
        var drow = dst.unsafe_offset(i * n)
        var di = Vec(d.unsafe_load(i))
        var j = 0
        while j + W <= n:
            drow.unsafe_store(
                j, arow.unsafe_load[width=W](j) * d.unsafe_load[width=W](j) * di
            )
            j += W
        while j < n:
            drow.unsafe_store(j, arow.unsafe_load(j) * d.unsafe_load(j) * d.unsafe_load(i))
            j += 1


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
        var lrow = laplacian.unsafe_offset(i * n)
        var drow = dst.unsafe_offset(i * n)
        var j = 0
        while j + W <= n:
            drow.unsafe_store(j, lrow.unsafe_load[width=W](j) * Vec(scale))
            j += W
        while j < n:
            drow.unsafe_store(j, lrow.unsafe_load(j) * scale)
            j += 1
        # the `I` of `(2 / lambda) * L - I` touches only this row's diagonal,
        # so it is applied here rather than in a second pass over the block
        drow.unsafe_store(i, drow.unsafe_load(i) - 1.0)


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
    _scale_block(result, n * n, teleport_probability)
    return 1


def _scale_block(x: FPtr, count: Int, f: Float64):
    """`x[:count] *= f`, in place."""
    var i = 0
    while i + W <= count:
        x.unsafe_store(i, x.unsafe_load[width=W](i) * Vec(f))
        i += W
    while i < count:
        x.unsafe_store(i, x.unsafe_load(i) * f)
        i += 1


def GCN_Aadj_feats_op(
    adj: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    scratch: FPtr,
    n: Int,
    k: Int,
    method: Int,
) -> Int:
    """Upstream `GCN_Aadj_feats_op(features, A, k=1, method="gcn")`.

    ```
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    if method == "gcn":       A = preprocess_adj(A)
    elif method == "chebyshev": raise ValueError(...)
    elif method == "sgc":      A = preprocess_adj(A) ** k
    return features, A
    ```

    Upstream 1.2.1 removed the `chebyshev` branch outright, so there is no
    `cheb_out` buffer here: the Python wrapper raises the same `ValueError`
    before the call.

    Returns 0 for an undefined method (upstream raises `ValueError`) or for a
    non-positive `k` under `sgc`.
    """
    symmetrize(adj, work, n)

    if method == METHOD_GCN:
        # `preprocess_adj` is `normalize_adj(add_self_loops(A))`. `add_self_loops`
        # leaves every off-diagonal entry alone and sets the diagonal to 1, so
        # it runs directly into `work2`, and `normalize_adj` then writes the
        # normalized adjacency straight into `result` -- no `n * n` copy back
        # at the end, which is what the original ordering paid.
        add_self_loops(work, work2, n)
        normalize_adj(work2, result, scratch, n, 1)
        return 1

    if method == METHOD_SGC:
        if k <= 0:
            return 0
        # A = A ** k
        add_self_loops(work, work2, n)
        normalize_adj(work2, result, scratch, n, 1)
        _copy_block(result, work2, n)
        var step = 1
        while step < k:
            dot(work2, result, work, n, n, n)
            _copy_block(work, result, n)
            step += 1
        return 1

    if method == METHOD_NONE:
        _copy_block(work, result, n)
        return 1

    return 0


def _copy_block(src: FPtr, dst: FPtr, n: Int):
    """`dst[:] = src` over an `n * n` block, one row at a time."""
    for i in range(n):
        var srow = src.unsafe_offset(i * n)
        var drow = dst.unsafe_offset(i * n)
        var j = 0
        while j + W <= n:
            drow.unsafe_store(j, srow.unsafe_load[width=W](j))
            j += W
        while j < n:
            drow.unsafe_store(j, srow.unsafe_load(j))
            j += 1
