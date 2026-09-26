"""Port of `stellargraph/layer/graph_attention.py` (v0.8.1):
`GraphAttention` and `GraphAttentionSparse`.

Upstream `GraphAttention.call` is a `for head in range(self.attn_heads)` loop
over the projection, the attention scores, the masked softmax, the neighbour
combination and the bias. That loop is the Mojo kernel; this module is the
weight container and the argument marshalling.
"""

from __future__ import annotations

import numpy as np

from . import activations as _acts
from ._lib import addr, f64, i32, lib
from .sparse import SparseTensor

HEADS_REDUCTION = {"concat": 0, "average": 1}


class GraphAttention:
    """Graph Attention (GAT) layer, from
    `stellargraph/layer/graph_attention.py`.

    ```
    GraphAttention(units, attn_heads=1, attn_heads_reduction="concat",
                   in_dropout_rate=0.0, attn_dropout_rate=0.0,
                   activation="relu", use_bias=True, final_layer=False,
                   saliency_map_support=False)
    ```

    This does not add self loops to the adjacency matrix; the caller must
    preprocess it, as upstream documents.
    """

    def __init__(
        self,
        units,
        attn_heads: int = 1,
        attn_heads_reduction: str = "concat",
        in_dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        activation: str = "relu",
        use_bias: bool = True,
        final_layer: bool = False,
        saliency_map_support: bool = False,
        **kwargs,
    ):
        if attn_heads_reduction not in HEADS_REDUCTION:
            raise ValueError(
                "GraphAttention: Possible heads reduction methods: concat, "
                "average; received {}".format(attn_heads_reduction)
            )
        if activation not in _acts.CODES:
            raise ValueError("unsupported activation {!r}".format(activation))
        self.units = int(units)
        self.attn_heads = int(attn_heads)
        self.attn_heads_reduction = attn_heads_reduction
        self.in_dropout_rate = in_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.final_layer = bool(final_layer)
        self.saliency_map_support = bool(saliency_map_support)

        self.output_dim = (
            self.units * self.attn_heads
            if attn_heads_reduction == "concat"
            else self.units
        )
        self.kernels = None
        self.biases = None
        self.attn_kernels = None
        # Upstream adds these two scalars for the saliency branch.
        self.delta = 1.0
        self.non_exist_edge = 0.0

    def build(self, input_dim: int) -> "GraphAttention":
        """One `(input_dim, units)` kernel, one `(units,)` bias and one
        `(2 * units,)` attention kernel per head, glorot-uniform as upstream's
        `kernel_initializer="glorot_uniform"` default."""
        rng = np.random.default_rng(0)
        d, u, k = int(input_dim), self.units, self.attn_heads
        self.kernels = np.zeros((k, d, u), dtype=np.float64)
        self.biases = np.zeros((k, u), dtype=np.float64) if self.use_bias else None
        self.attn_kernels = np.zeros((k, 2 * u), dtype=np.float64)
        for head in range(k):
            lim = np.sqrt(6.0 / (d + u))
            self.kernels[head] = rng.uniform(-lim, lim, size=(d, u))
            lim = np.sqrt(6.0 / (u + 1))
            self.attn_kernels[head] = rng.uniform(-lim, lim, size=2 * u)
        return self

    def call(self, X, A, out_indices=None) -> np.ndarray:
        """`call(inputs)` with `X` `[n, f]`, `A` `[n, n]`, `out_indices` `[m]`."""
        X = f64(X)
        A = f64(A)
        n, f = X.shape
        if self.kernels is None:
            self.build(f)
        # `out_indices=None` means no gather at all, so a final layer keeps
        # every node rather than gathering node 0
        if out_indices is None:
            out_indices = np.zeros(0, dtype=np.int32)
        out_indices = i32(np.asarray(out_indices).reshape(-1))
        m = int(out_indices.shape[0])

        result = np.zeros((n, self.output_dim), dtype=np.float64)
        # work holds the projected features, the dense [n, n] attention block
        # and two [n] scratch vectors; it is reused for the output activation.
        work = np.zeros(
            max(n * self.units + 2 * n * n + n, n * self.output_dim), dtype=np.float64
        )
        work2 = np.zeros(self.attn_heads * n * self.units, dtype=np.float64)
        bias = (
            self.biases.reshape(-1)
            if self.use_bias
            else np.zeros(max(self.attn_heads * self.units, 1))
        )
        lib().msg_GraphAttention_call(
            addr(X), addr(A), addr(self.kernels.reshape(-1)),
            addr(self.attn_kernels.reshape(-1)), addr(bias), addr(result),
            addr(work), addr(work2), addr(out_indices),
            n, m, f, self.units, self.attn_heads,
            1 if self.use_bias else 0,
            HEADS_REDUCTION[self.attn_heads_reduction],
            _acts.code(self.activation), _acts.alpha(self.activation),
            1 if self.final_layer else 0,
            1 if self.saliency_map_support else 0,
            float(self.delta), float(self.non_exist_edge),
        )
        return result[:m] if self.final_layer and m else result

    def __call__(self, X, A, out_indices=None) -> np.ndarray:
        return self.call(X, A, out_indices)


class GraphAttentionSparse(GraphAttention):
    """The sparse variant, from `stellargraph/layer/graph_attention.py`.

    Same signature as `GraphAttention`; `A` is a `SparseTensor` (or anything
    convertible to one) instead of a dense matrix.
    """

    def call(self, X, A, out_indices=None) -> np.ndarray:
        if not isinstance(A, SparseTensor):
            A = SparseTensor.from_dense(A)
        X = f64(X)
        n, f = X.shape
        if self.kernels is None:
            self.build(f)
        e = len(A)
        # `out_indices=None` means no gather at all, so a final layer keeps
        # every node rather than gathering node 0
        if out_indices is None:
            out_indices = np.zeros(0, dtype=np.int32)
        out_indices = i32(np.asarray(out_indices).reshape(-1))
        m = int(out_indices.shape[0])

        result = np.zeros((n, self.output_dim), dtype=np.float64)
        # the kernel's scratch is [n*units features | n attn_self |
        # n attn_neighs | e attn_values | e attn_norm], then the gathered rows
        work = np.zeros(
            max(n * self.units + 2 * n + 2 * e, n * self.output_dim),
            dtype=np.float64,
        )
        work2 = np.zeros(self.attn_heads * n * self.units, dtype=np.float64)
        bias = (
            self.biases.reshape(-1)
            if self.use_bias
            else np.zeros(max(self.attn_heads * self.units, 1))
        )
        lib().msg_GraphAttentionSparse_call(
            addr(X), addr(A.rows), addr(A.cols), addr(A.indptr),
            addr(self.kernels.reshape(-1)), addr(self.attn_kernels.reshape(-1)),
            addr(bias), addr(result), addr(work), addr(work2), addr(out_indices),
            n, e, m, f, self.units, self.attn_heads,
            1 if self.use_bias else 0,
            HEADS_REDUCTION[self.attn_heads_reduction],
            _acts.code(self.activation), _acts.alpha(self.activation),
            1 if self.final_layer else 0,
        )
        return result[:m] if self.final_layer and m else result
