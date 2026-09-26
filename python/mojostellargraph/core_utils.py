"""Port of `stellargraph/core/utils.py` (v0.8.1).

Upstream is SciPy over sparse matrices; these functions take and return dense
`float64` arrays, which is what crosses the C ABI. Names, argument order and
defaults are upstream's. The one place a SciPy call has no Mojo counterpart is
`eigsh` in `rescale_laplacian`, and there the eigenvalue is an argument.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib

METHODS = {"gcn": 0, "chebyshev": 1, "sgc": 2, "none": 3, None: 3}


def _n(adj) -> int:
    return int(adj.shape[0])


def normalize_adj(adj, symmetric: bool = True) -> np.ndarray:
    """Upstream `normalize_adj(adj, symmetric=True)`.

    ```
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm
    """
    adj = f64(adj)
    n = _n(adj)
    dst = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    lib().msg_normalize_adj(addr(adj), addr(dst), addr(d), n, 1 if symmetric else 0)
    return dst


def normalized_laplacian(adj, symmetric: bool = True) -> np.ndarray:
    """Upstream `normalized_laplacian(adj, symmetric=True)`: `I - A_norm`."""
    adj = f64(adj)
    n = _n(adj)
    dst = np.zeros((n, n), dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    lib().msg_normalized_laplacian(addr(adj), addr(dst), addr(d), n, 1 if symmetric else 0)
    return dst


def power_iteration(a, max_iter: int = 100, tol: float = 1e-10) -> float:
    """Largest-magnitude eigenvalue of a symmetric `a`.

    Stands in for the `eigsh(laplacian, 1, which="LM")` inside upstream's
    `rescale_laplacian`. See the README: this is a different algorithm, not a
    like-for-like port.
    """
    a = f64(a)
    n = _n(a)
    work = np.zeros(n, dtype=np.float64)
    vec = np.zeros(n, dtype=np.float64)
    return lib().msg_power_iteration(
        addr(a), addr(work), addr(vec), n, max_iter, tol
    )


def rescale_laplacian(laplacian, largest_eigval: float | None = None) -> np.ndarray:
    """Upstream `rescale_laplacian(laplacian)`: `(2 / lambda) * L - I`.

    Upstream computes `lambda` with `eigsh` and falls back to `2` on
    `ArpackNoConvergence`. Here it is an argument; passing `None` runs
    `power_iteration` over the same normalized Laplacian upstream scales.
    """
    laplacian = f64(laplacian)
    n = _n(laplacian)
    if largest_eigval is None:
        largest_eigval = power_iteration(laplacian) or 2.0
    dst = np.zeros((n, n), dtype=np.float64)
    lib().msg_rescale_laplacian(addr(laplacian), addr(dst), n, float(largest_eigval))
    return dst


def chebyshev_polynomial(X, k: int) -> list[np.ndarray]:
    """Upstream `chebyshev_polynomial(X, k)`: `[I, X, 2 X T1 - T0, ...]`.

    Upstream returns a list of sparse matrices built by the recurrence
    `T_k = 2 * X.dot(T_{k-1}) - T_{k-2}`. The Mojo kernel writes the same
    polynomials consecutively into one buffer; each is handed back as its own
    array here.
    """
    X = f64(X)
    n = _n(X)
    k = int(k)
    # upstream seeds the list as `[sp.eye(n), X]` and then appends, so it holds
    # at least two matrices whatever k is
    blocks = max(k + 1, 2)
    result = np.zeros((blocks * n, n), dtype=np.float64)
    work = np.zeros((n, n), dtype=np.float64)
    lib().msg_chebyshev_polynomial(addr(X), addr(result), addr(work), n, k)
    return [result[i * n : (i + 1) * n] for i in range(blocks)]


def invert(a) -> np.ndarray:
    """`np.linalg.inv(a)` for a dense `a`, by Gauss-Jordan with partial
    pivoting.

    Stands in for the `np.linalg.inv` inside upstream's `PPNP_Aadj_feats_op`.
    Raises `np.linalg.LinAlgError` on a singular matrix, exactly as NumPy does.
    """
    a = f64(a)
    n = _n(a)
    out = np.zeros((n, n), dtype=np.float64)
    work = np.zeros((n, 2 * n), dtype=np.float64)
    if lib().msg_invert(addr(a), addr(out), addr(work), n) == 0:
        raise np.linalg.LinAlgError("Singular matrix")
    return out


def PPNP_Aadj_feats_op(
    features: np.ndarray, A: np.ndarray, teleport_probability: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream `PPNP_Aadj_feats_op(features, A, teleport_probability=0.1)`.

    Symmetrizes, adds self loops, symmetrically normalizes, then returns
    `teleport_probability * inv(I - (1 - teleport_probability) * A_norm)`.
    """
    if teleport_probability > 1.0 or teleport_probability < 0.0:
        raise ValueError(
            "teleport_probability should be between 0.0 and 1.0 (inclusive)"
        )
    A = f64(A)
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
    features: np.ndarray, A: np.ndarray, k: int = 1, method: str = "gcn"
) -> tuple:
    """Upstream `GCN_Aadj_feats_op(features, A, k=1, method="gcn")`.

    ```
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    if method == "gcn":       A = preprocess_adj(A)
    elif method == "chebyshev": T_k = chebyshev_polynomial(rescale_laplacian(normalized_laplacian(A)), k)
    elif method == "sgc":      A = preprocess_adj(A) ** k
    return features, A
    ```

    For `method="chebyshev"` upstream returns `([features] + T_k, A)`; this
    returns the same pair, with the Chebyshev stack as a list of dense blocks.
    """
    if method not in METHODS:
        raise ValueError(
            "Undefined method for adjacency matrix transformation. "
            "Accepted: 'gcn' (default), 'chebyshev','sgc', and 'self_loops'."
        )
    A = f64(A)
    n = _n(A)
    code = METHODS[method]
    k = int(k)
    out = np.zeros((n, n), dtype=np.float64)
    work = np.zeros((n, n), dtype=np.float64)
    work2 = np.zeros((n, n), dtype=np.float64)
    scratch = np.zeros(n, dtype=np.float64)
    cheb = np.zeros(((k + 1) * n, n), dtype=np.float64)
    ok = lib().msg_GCN_Aadj_feats_op(
        addr(A), addr(out), addr(work), addr(work2), addr(scratch), addr(cheb),
        n, k, code,
    )
    if ok == 0:
        if method == "chebyshev":
            raise ValueError(
                "max_degree should be positive integer of value at least 2 for "
                "method='chebyshev'; but received value {}.".format(k)
            )
        if method == "sgc":
            raise ValueError(
                "k should be positive integer for method='sgcn'; but received "
                "value {}.".format(k)
            )
        raise ValueError("undefined method")
    if method == "chebyshev":
        return [features] + [cheb[i * n : (i + 1) * n] for i in range(k + 1)], out
    return features, out
