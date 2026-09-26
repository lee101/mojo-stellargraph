"""Port of `stellargraph/layer/ppnp.py` (v0.8.1):
`PPNPPropagationLayer` and `PPNP`.

PPNP puts a stack of fully connected layers in front of one propagation
through the personalized-PageRank matrix, which the generator precomputes with
`method="ppnp"`.
"""

from __future__ import annotations

import numpy as np

from . import activations as _acts
from ._lib import addr, f64, lib, node_indices


class PPNPPropagationLayer:
    """
    Personalized Propagation of Neural Predictions, as in
    https://arxiv.org/abs/1810.05997.

    ```
    PPNPPropagationLayer(units, final_layer=False)
    ```

    `call(features, A, out_indices=None)` is `K.dot(A, features)`, optionally
    gathered.
    """

    def __init__(self, units, final_layer: bool = False, **kwargs):
        self.units = int(units)
        self.final_layer = bool(final_layer)

    def call(self, features, A, out_indices=None) -> np.ndarray:
        """`PPNPPropagationLayer.call`."""
        features = f64(features)
        A = f64(A)
        n, f = features.shape
        if A.shape != (n, n):
            raise ValueError(
                "adjacency must be ({}, {}), got {}".format(n, n, A.shape)
            )
        # `out_indices=None` means no gather at all, so a final layer keeps
        # every node rather than gathering node 0
        if out_indices is None:
            out_indices = np.zeros(0, dtype=np.int32)
        out_indices = node_indices(out_indices, n)
        m = int(out_indices.shape[0])
        result = np.zeros((n, f), dtype=np.float64)
        gathered = np.zeros((max(m, 1), f), dtype=np.float64)
        lib().msg_PPNPPropagationLayer_call(
            addr(features), addr(A), addr(result), addr(gathered),
            addr(out_indices), n, m, f, 1 if self.final_layer else 0,
        )
        return gathered if self.final_layer and m else result

    def __call__(self, features, A, out_indices=None) -> np.ndarray:
        return self.call(features, A, out_indices)


class PPNP:
    """
    PPNP, as in https://arxiv.org/abs/1810.05997.

    ```
    PPNP(layer_sizes, activations, generator, bias=True, dropout=0.0,
         kernel_regularizer=None)
    ```

    The model is a stack of fully connected layers followed by one
    `PPNPPropagationLayer`, which is what `PPNP.__call__` evaluates.
    """

    def __init__(self, layer_sizes, activations, generator, bias: bool = True,
                 dropout: float = 0.0, kernel_regularizer=None):
        from .node_mappers import FullBatchNodeGenerator

        if not isinstance(generator, FullBatchNodeGenerator):
            raise TypeError("Generator should be a instance of FullBatchNodeGenerator")

        if not len(layer_sizes) == len(activations):
            raise ValueError(
                "The number of layers should equal the number of activations"
            )
        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        self.bias = bool(bias)
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.generator = generator
        self._layers = None

    def build(self, input_dim: int, seed: int = 0):
        """A `Dense`-equivalent kernel per layer, glorot-uniform."""
        rng = np.random.default_rng(seed)
        dim = int(input_dim)
        self._layers = []
        for size, act in zip(self.layer_sizes, self.activations):
            if act not in _acts.CODES:
                raise ValueError("unsupported activation {!r}".format(act))
            lim = np.sqrt(6.0 / (dim + size))
            self._layers.append(
                {
                    "kernel": rng.uniform(-lim, lim, size=(dim, size)),
                    "bias": np.zeros(size) if self.bias else None,
                    "act": act,
                }
            )
            dim = size
        self._layers.append(
            {"kernel": None, "bias": None, "act": None, "propagation": True}
        )
        return self

    def __call__(self, features, A, out_indices=None) -> np.ndarray:
        h = f64(features)
        if self._layers is None:
            self.build(h.shape[1])
        for layer in self._layers[:-1]:
            h = h @ layer["kernel"]
            if layer["bias"] is not None:
                h = h + layer["bias"]
            h = _acts.numpy_activation(layer["act"])(h)
        return PPNPPropagationLayer(h.shape[1], bool(out_indices is not None))(
            h, A, out_indices
        )
