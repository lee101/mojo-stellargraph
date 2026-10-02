"""Port of `stellargraph/core/utils.py` (upstream 1.2.1).

Upstream is SciPy over sparse matrices; these functions take and return dense
`float64` arrays, which is what crosses the C ABI. Names, argument order and
defaults are upstream's.

Two upstream operations have no dense-kernel counterpart and are called out in
the README:

* `rescale_laplacian` computes its eigenvalue with
  `scipy.sparse.linalg.eigsh(laplacian, 1, which="LM")`, falling back to `2` on
  `ArpackNoConvergence`. Here the eigenvalue is an argument, and
  `power_iteration` supplies one when the caller does not.
* `calculate_laplacian` is `D.dot(adj).dot(D)` with `D = diag(ravel(adj.sum(0))
  ** -0.5)`; it is implemented over the same dense layout.

`method="chebyshev"` was removed upstream in 1.2.1 (`GCN_Aadj_feats_op` raises
`ValueError` for it), so it raises the same error here rather than computing a
polynomial nothing calls.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib

#: Upstream dispatches on the `method` string; the C ABI carries a code.
METHODS = {"gcn": 0, "sgc": 2, "none": 3, None: 3}


def _n(adj) -> int:
    return int(adj.shape[0])


def _as_dense(adj) -> np.ndarray:
    """Upstream takes a SciPy sparse matrix; a dense array is taken as-is and a
    sparse one is densified, so both upstream inputs work unchanged."""
    if hasattr(adj, "toarray"):
        return f64(adj.toarray())
    return f64(adj)


def add_self_loops(adj) -> np.ndarray:
    """`adj + sp.diags(np.ones(n) - adj.diagonal())`, the expression upstream
    writes out in `normalize_adj`, `GCN_Aadj_feats_op` and
    `FullBatchGenerator.__init__`."""
    adj = _as_dense(adj)
    n = _n(adj)
    dst = np.zeros((n, n), dtype=np.float64)
    lib().msg_self_loops(addr(adj), addr(dst), n)
    return dst


def normalize_adj(adj, symmetric: bool = True, add_self_loops: bool = False):
    """Upstream `normalize_adj(adj, symmetric=True, add_self_loops=False)`.

    ```
    if add_self_loops:
        adj = adj + sp.diags(np.ones(adj.shape[0]) - adj.diagonal())
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    ```
    """
    adj = _as_dense(adj)
    n = _n(adj)
    if add_self_loops:
        # upstream mutates `adj` before the branch, so the row sums below are
        # taken over the looped matrix. The flag shadows the module-level
        # `add_self_loops`, so the function is reached through the module.
        adj = globals()["add_self_loops"](adj)
    dst = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    lib().msg_normalize_adj(addr(adj), addr(dst), addr(d), n, 1 if symmetric else 0)
    return dst


def normalized_laplacian(adj, symmetric: bool = True) -> np.ndarray:
    """Upstream `normalized_laplacian(adj, symmetric=True)`: `I - A_norm`."""
    adj = _as_dense(adj)
    n = _n(adj)
    dst = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    lib().msg_normalized_laplacian(addr(adj), addr(dst), addr(d), n, 1 if symmetric else 0)
    return dst


def calculate_laplacian(adj) -> np.ndarray:
    """Upstream `calculate_laplacian(adj)`:

    ```
    D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
    adj = np.dot(D, np.dot(adj, D))
    ```
    """
    adj = _as_dense(adj)
    n = _n(adj)
    dst = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    lib().msg_calculate_laplacian(addr(adj), addr(dst), addr(d), n)
    return dst


def power_iteration(a, max_iter: int = 100, tol: float = 1e-10) -> float:
    """Largest-magnitude eigenvalue of a symmetric `a`.

    Stands in for the `eigsh(laplacian, 1, which="LM")` inside upstream's
    `rescale_laplacian`. This is a different algorithm, not a like-for-like
    port; see the README.
    """
    a = _as_dense(a)
    n = _n(a)
    work = np.zeros(n, dtype=np.float64)
    vec = np.zeros(n, dtype=np.float64)
    return lib().msg_power_iteration(
        addr(a), addr(work), addr(vec), n, int(max_iter), float(tol)
    )


def rescale_laplacian(laplacian, largest_eigval: float | None = None) -> np.ndarray:
    """Upstream `rescale_laplacian(laplacian)`: `(2 / lambda) * L - I`.

    Upstream computes `lambda` with `eigsh` and falls back to `2` on
    `ArpackNoConvergence`. Here it is an argument; passing `None` runs
    `power_iteration` over the Laplacian upstream scales.
    """
    laplacian = _as_dense(laplacian)
    n = _n(laplacian)
    if largest_eigval is None:
        # upstream's fallback for a non-converging `eigsh` is 2.0
        largest_eigval = power_iteration(laplacian) or 2.0
    dst = np.zeros((n, n), dtype=np.float64)
    lib().msg_rescale_laplacian(addr(laplacian), addr(dst), n, float(largest_eigval))
    return dst


def invert(a) -> np.ndarray:
    """`np.linalg.inv(a)` for a dense `a`, by Gauss-Jordan with partial
    pivoting.

    Stands in for the `np.linalg.inv` inside upstream's `PPNP_Aadj_feats_op`.
    Raises `np.linalg.LinAlgError` on a singular matrix, exactly as NumPy does.
    """
    a = _as_dense(a)
    n = _n(a)
    out = np.zeros((n, n), dtype=np.float64)
    work = np.zeros((n, 2 * n), dtype=np.float64)
    if lib().msg_invert(addr(a), addr(out), addr(work), n) == 0:
        raise np.linalg.LinAlgError("Singular matrix")
    return out


def PPNP_Aadj_feats_op(
    features: np.ndarray, A, teleport_probability: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream `PPNP_Aadj_feats_op(features, A, teleport_probability=0.1)`.

    Symmetrizes, adds self loops, symmetrically normalizes, then returns
    `teleport_probability * inv(I - (1 - teleport_probability) * A_norm)`.
    Upstream returns `(features, A)`; `features` is passed through untouched.
    """
    if teleport_probability > 1.0 or teleport_probability < 0.0:
        raise ValueError(
            "teleport_probability should be between 0.0 and 1.0 (inclusive)"
        )
    A = _as_dense(A)
    n = _n(A)
    out = np.zeros((n, n), dtype=np.float64)
    tmp = np.zeros((n, n), dtype=np.float64)
    tmp2 = np.zeros((n, n), dtype=np.float64)
    work = np.zeros((n, 2 * n), dtype=np.float64)
    ok = lib().msg_PPNP_Aadj_feats_op(
        addr(A), addr(out), addr(tmp), addr(tmp2), addr(work), n,
        float(teleport_probability),
    )
    if ok == 0:
        raise np.linalg.LinAlgError("Singular matrix")
    return features, out


def GCN_Aadj_feats_op(
    features: np.ndarray, A, k: int = 1, method: str = "gcn"
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream `GCN_Aadj_feats_op(features, A, k=1, method="gcn")`.

    ```
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    if method == "gcn":    A = preprocess_adj(A)
    elif method == "sgc":  A = preprocess_adj(A) ** k
    elif method == "chebyshev": raise ValueError(...)
    ```

    `method="chebyshev"` was removed in 1.2.1 and raises here too.
    """
    if method == "chebyshev":
        # upstream 1.2.1 removed this branch outright
        raise ValueError("method 'chebyshev' did not behave correctly and has been removed")
    if method not in METHODS:
        raise ValueError(
            "Undefined method for adjacency matrix transformation. "
            "Accepted: 'gcn' (default), 'sgc', and 'self_loops'."
        )
    A = _as_dense(A)
    n = _n(A)
    code = METHODS[method]
    k = int(k)
    out = np.zeros((n, n), dtype=np.float64)
    work = np.zeros((n, n), dtype=np.float64)
    work2 = np.zeros((n, n), dtype=np.float64)
    scratch = np.zeros(n, dtype=np.float64)
    ok = lib().msg_GCN_Aadj_feats_op(
        addr(A), addr(out), addr(work), addr(work2), addr(scratch), n, k, code,
    )
    if ok == 0:
        if method == "sgc":
            raise ValueError(
                "k should be positive integer for method='sgcn'; but received "
                "type {} with value {}.".format(type(k).__name__, k)
            )
        raise ValueError("undefined method")
    return features, out