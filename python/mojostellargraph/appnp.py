"""Port of `stellargraph/layer/appnp.py` (v0.8.1):
`APPNPPropagationLayer` and `APPNP`.

APPNP is the approximate half of Klicpera et al. (2018): instead of inverting
`(I - (1 - alpha) A)` up front as PPNP does, it applies
`(1 - alpha) A Z + alpha Z` repeatedly to the base model's output.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i32, lib


class APPNPPropagationLayer:
    """
    Approximate Personalized Propagation of Neural Predictions, as in
    https://arxiv.org/abs/1810.05997.

    ```
    APPNPPropagationLayer(units, teleport_probability=0.1, final_layer=False)
    ```
    """

    def __init__(self, units, teleport_probability: float = 0.1,
                 final_layer: bool = False, **kwargs):
        self.units = int(units)
        self.teleport_probability = float(teleport_probability)
        self.final_layer = bool(final_layer)

    def call(self, propagated_features, features, A, out_indices=None) -> np.ndarray:
        """`APPNPPropagationLayer.call`:
        `(1 - alpha) * K.dot(A, propagated_features) + alpha * features`."""
        propagated_features = f64(propagated_features)
        features = f64(features)
        A = f64(A)
        n, f = propagated_features.shape
        # `out_indices=None` means no gather at all, so a final layer keeps
        # every node rather than gathering node 0
        if out_indices is None:
            out_indices = np.zeros(0, dtype=np.int32)
        out_indices = i32(np.asarray(out_indices).reshape(-1))
        m = int(out_indices.shape[0])
        result = np.zeros((n, f), dtype=np.float64)
        gathered = np.zeros((max(m, 1), f), dtype=np.float64)
        lib().msg_APPNPPropagationLayer_call(
            addr(propagated_features), addr(features), addr(A), addr(result),
            addr(gathered), addr(out_indices), n, m, f,
            self.teleport_probability, 1 if self.final_layer else 0,
        )
        return gathered if self.final_layer and m else result

    def __call__(self, propagated_features, features, A, out_indices=None) -> np.ndarray:
        return self.call(propagated_features, features, A, out_indices)


class APPNP:
    """
    Approximate Personalized Propagation of Neural Predictions.

    ```
    APPNP(layer_sizes, generator, bias=True, dropout=0.0, teleport_probability=0.1)
    ```

    `propagate(x, A, k)` runs the `k`-step power iteration the layer stack
    performs, and `__call__` applies it to a base model's output.
    """

    def __init__(self, layer_sizes, generator, bias: bool = True, dropout: float = 0.0,
                 teleport_probability: float = 0.1):
        self.layer_sizes = list(layer_sizes)
        self.generator = generator
        self.bias = bool(bias)
        self.dropout = dropout
        self.teleport_probability = float(teleport_probability)
        self._layers = []

    def build(self, input_dim: int, seed: int = 0):
        """A `Dense`-equivalent kernel per layer, glorot-uniform."""
        rng = np.random.default_rng(seed)
        dim = int(input_dim)
        self._layers = []
        for size in self.layer_sizes:
            lim = np.sqrt(6.0 / (dim + size))
            self._layers.append(
                {
                    "kernel": rng.uniform(-lim, lim, size=(dim, size)),
                    "bias": np.zeros(size) if self.bias else None,
                }
            )
            dim = size
        return self

    def propagate(self, x, A, k: int) -> np.ndarray:
        """The repeated propagation `APPNP.propagate_model` performs around a
        trained base model, run `k` times."""
        x = f64(x)
        A = f64(A)
        n, f = x.shape
        dst = np.zeros((n, f), dtype=np.float64)
        work = np.zeros((n, f), dtype=np.float64)
        lib().msg_APPNP_propagate(
            addr(x), addr(A), addr(dst), addr(work), n, f,
            self.teleport_probability, int(k),
        )
        return dst

    def __call__(self, x, A, k: int = 1, out_indices=None) -> np.ndarray:
        h = f64(x)
        if not self._layers:
            self.build(h.shape[1])
        for layer in self._layers:
            h = h @ layer["kernel"]
            if layer["bias"] is not None:
                h = h + layer["bias"]
        return self.propagate(h, A, k)
