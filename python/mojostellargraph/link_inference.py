"""Port of `stellargraph/layer/link_inference.py` (v0.8.1):
`LeakyClippedLinear` and the `link_inference` / `link_classification` /
`link_regression` factories.

Upstream `link_inference(...)` returns a closure over a Keras graph. Here it
returns a callable that runs the same branch, with the `Dense` that follows it
and the optional `LeakyClippedLinear` on top.
"""

from __future__ import annotations

import numpy as np

from . import activations as _acts
from ._lib import addr, f64, lib

METHODS = {
    "ip": 0,
    "dot": 1,
    "l1": 2,
    "l2": 3,
    "mul": 4,
    "hadamard": 4,
    "concat": 5,
    "avg": 6,
}

_WIDE = {"concat"}


class LeakyClippedLinear:
    """
    Leaky Clipped Linear Unit.

    ```
    LeakyClippedLinear(low=1.0, high=5.0, alpha=0.1)
    ```
    """

    def __init__(self, low: float = 1.0, high: float = 5.0, alpha: float = 0.1, **kwargs):
        self.gamma = 1.0 - alpha
        self.lo = float(low)
        self.hi = float(high)
        self.alpha = float(alpha)

    def call(self, x) -> np.ndarray:
        """`x + gamma * relu(lo - x) - gamma * relu(x - hi)`."""
        x = f64(x)
        flat = x.reshape(-1)
        dst = np.zeros(flat.shape, dtype=np.float64)
        lib().msg_LeakyClippedLinear_call(
            addr(flat), addr(dst), flat.size, self.lo, self.hi, self.alpha
        )
        return dst.reshape(x.shape)

    def __call__(self, x) -> np.ndarray:
        return self.call(x)


def link_inference(
    output_dim: int = 1,
    output_act: str = "linear",
    edge_embedding_method: str = "ip",
    clip_limits=None,
    name: str = "link_inference",
    kernel: np.ndarray | None = None,
    bias: np.ndarray | None = None,
):
    """
    Defines an edge inference function that takes source, destination node
    embeddings as input and returns a numeric vector of `output_dim` size.

    Args:
        output_dim (int): number of predictor's output units.
        output_act (str): activation applied to the output.
        edge_embedding_method (str): one of 'concat', 'ip'/'dot', 'mul'/'hadamard',
            'l1', 'l2', 'avg'.
        clip_limits (tuple): lower and upper thresholds for the
            `LeakyClippedLinear` unit on top; `None` skips it.
        name (str): used for error logging.
        kernel, bias: the `Dense` weights that every branch except 'ip'/'dot'
            applies. Upstream builds them with `Dense`; they are passed in here
            so the caller owns them.

    Returns:
        A function taking `[n, d]` source and destination embeddings, in that
        order, and returning `[n, output_dim]`.
    """
    if edge_embedding_method not in METHODS:
        raise NotImplementedError(
            "{}: the requested method '{}' is not known/not implemented".format(
                name, edge_embedding_method
            )
        )
    if output_act not in _acts.CODES:
        raise ValueError("unsupported activation {!r}".format(output_act))
    method = METHODS[edge_embedding_method]
    if method in (0, 1) and output_dim != 1:
        import warnings

        warnings.warn(
            "Inner product is a scalar, but output_dim is set to {}. Reverting "
            "output_dim to be 1.".format(output_dim)
        )
        output_dim = 1

    # `ip`/`dot` is the one arm with no Dense upstream, so it needs no kernel.
    needs_dense = method not in (METHODS["ip"], METHODS["dot"])
    if not needs_dense and kernel is None:
        # the kernel pointer is non-nullable at the C ABI, so a one-element
        # placeholder crosses; the `ip`/`dot` arm never reads it
        kernel = np.zeros((1, output_dim))

    kernel_rows = None
    if kernel is None:
        raise ValueError(
            "'{}' needs an explicit kernel: the embedding width is only "
            "known at call time".format(edge_embedding_method)
        )
    kernel = np.atleast_2d(f64(kernel))
    if kernel.ndim != 2 or kernel.shape[1] != output_dim:
        raise ValueError(
            "kernel must have shape (in_dim, output_dim={}), got {}".format(
                output_dim, kernel.shape
            )
        )
    bias = np.zeros(output_dim) if bias is None else f64(bias).reshape(-1)
    kernel_rows = int(kernel.shape[0])
    kernel = np.ascontiguousarray(kernel).reshape(-1)
    clip = 1 if clip_limits is not None else 0
    clip_low = float(clip_limits[0]) if clip_limits is not None else 0.0
    clip_high = float(clip_limits[1]) if clip_limits is not None else 0.0

    def edge_function(x0, x1) -> np.ndarray:
        x0 = f64(x0)
        x1 = f64(x1)
        if x0.shape != x1.shape:
            raise ValueError(
                "source and destination embeddings must have the same shape, "
                "got {} and {}".format(x0.shape, x1.shape)
            )
        n, d = x0.shape
        want = 2 * d if method == METHODS["concat"] else d
        # the `ip`/`dot` arm has no Dense, so its placeholder kernel carries no
        # input-width contract
        if needs_dense and kernel_rows != want:
            # `Dense` rejects the shape at build time; the embedding width is
            # only known here, so the check happens here
            raise ValueError(
                "kernel expects {} input features, the embeddings have {}".format(
                    kernel_rows, want
                )
            )
        # `le` is the `2 * d` concat buffer, and the activation and the
        # LeakyClippedLinear run over the `n * output_dim` result through it, so
        # it has to be at least that wide
        le = np.zeros((n, max(2 * d, output_dim, 1)), dtype=np.float64)
        result = np.zeros((n, max(output_dim, 1)), dtype=np.float64)
        width = lib().msg_link_inference_edge_function(
            addr(x0), addr(x1), addr(le), addr(kernel), addr(bias), addr(result),
            n, d, output_dim, method, _acts.code(output_act), _acts.alpha(output_act),
            1 if bias is not None else 0, clip, clip_low, clip_high,
        )
        return result.reshape(n, int(width))

    print(
        "{}: using '{}' method to combine node embeddings into edge "
        "embeddings".format(name, edge_embedding_method)
    )
    return edge_function


def link_classification(
    output_dim: int = 1, output_act: str = "sigmoid", edge_embedding_method: str = "ip"
):
    """
    Defines a function that predicts a binary or multi-class edge
    classification output from (source, destination) node embeddings.
    """
    return link_inference(
        output_dim=output_dim,
        output_act=output_act,
        edge_embedding_method=edge_embedding_method,
        name="link_classification",
    )


def link_regression(
    output_dim: int = 1, clip_limits=None, edge_embedding_method: str = "ip"
):
    """
    Defines a function that predicts a numeric edge regression output vector or
    scalar from (source, destination) node embeddings.
    """
    return link_inference(
        output_dim=output_dim,
        output_act="linear",
        edge_embedding_method=edge_embedding_method,
        clip_limits=clip_limits,
        name="link_regression",
    )
