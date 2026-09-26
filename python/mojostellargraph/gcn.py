"""Port of `stellargraph/layer/gcn.py` (v0.8.1): `GraphConvolution` and `GCN`.

Upstream is Keras, so the layer owns its weights and `call` is the whole
numeric body. Here the layer still owns its weights and `call` takes the same
inputs, minus the batch dimension Keras carries; `build` is explicit because
there is no `add_weight` behind it.
"""

from __future__ import annotations

import numpy as np

from . import activations as _acts
from ._lib import addr, f64, i32, lib


class GraphConvolution:
    """Graph Convolution (GCN) layer, from `stellargraph/layer/gcn.py`.

    ```
    GraphConvolution(units, activation=None, use_bias=True, final_layer=False)
    ```

    This class assumes that the normalized Laplacian matrix is passed to
    `call`, exactly as upstream documents.
    """

    def __init__(
        self, units, activation=None, use_bias: bool = True, final_layer: bool = False
    ):
        if activation not in _acts.CODES:
            raise ValueError("unsupported activation {!r}".format(activation))
        self.units = int(units)
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.final_layer = bool(final_layer)
        self.kernel = None
        self.bias = None

    def build(self, input_dim: int) -> "GraphConvolution":
        """`build(input_shapes)`: an `(input_dim, units)` kernel, plus a
        `(units,)` bias when `use_bias`."""
        self.kernel = np.zeros((int(input_dim), self.units), dtype=np.float64)
        self.bias = (
            np.zeros(self.units, dtype=np.float64) if self.use_bias else None
        )
        return self

    def call(self, features, A, out_indices=None) -> np.ndarray:
        """`K.dot(A, features) @ kernel + bias`, then the activation, then the
        gather when `final_layer`."""
        features = f64(features)
        A = f64(A)
        n, f = features.shape
        if self.kernel is None:
            self.build(f)
        if self.kernel.shape[0] != f:
            raise ValueError(
                "kernel was built for {} input features, got {}".format(
                    self.kernel.shape[0], f
                )
            )
        # `out_indices=None` means no gather at all, so a final layer keeps
        # every node rather than gathering node 0
        if out_indices is None:
            out_indices = np.zeros(0, dtype=np.int32)
        out_indices = i32(np.asarray(out_indices).reshape(-1))
        m = int(out_indices.shape[0])

        result = np.zeros((n, self.units), dtype=np.float64)
        # holds both `A @ features` (n x f) and the gathered rows
        work = np.zeros((n, max(n, f, self.units, 1)), dtype=np.float64)
        bias = self.bias if self.use_bias else np.zeros(max(self.units, 1))
        lib().msg_GraphConvolution_call(
            addr(features), addr(A), addr(self.kernel), addr(bias), addr(result),
            addr(work), addr(out_indices), n, m, f, self.units,
            1 if self.use_bias else 0, _acts.code(self.activation), _acts.alpha(self.activation),
            1 if self.final_layer else 0,
        )
        return result[:m] if self.final_layer and m else result

    def __call__(self, features, A, out_indices=None) -> np.ndarray:
        return self.call(features, A, out_indices)


class GCN:
    """A stack of `GraphConvolution` layers implementing
    https://arxiv.org/abs/1609.02907.

    ```
    GCN(layer_sizes, generator, bias=True, dropout=0.0, activations=None)
    ```

    `__call__` walks `self._layers` the way upstream does: a
    `Dropout(self.dropout)` — the identity at inference, which is the only mode
    implemented — followed by one `GraphConvolution` per layer, with
    `final_layer` set on the last.
    """

    def __init__(self, layer_sizes, generator, bias: bool = True,
                 dropout: float = 0.0, activations=None, **kwargs):
        n_layers = len(layer_sizes)
        self.layer_sizes = list(layer_sizes)
        self.bias = bool(bias)
        self.dropout = dropout
        self.generator = generator
        self.method = getattr(generator, "method", "gcn")
        if activations is None:
            acts = ["relu"] * n_layers
        elif len(activations) != n_layers:
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )
        else:
            acts = list(activations)
        for a in acts:
            if a not in _acts.CODES:
                raise ValueError("unsupported activation {!r}".format(a))
        self.activations = acts

        self._layers = []
        for ii in range(n_layers):
            # Dropout(self.dropout) -- the identity at inference
            self._layers.append(
                GraphConvolution(
                    self.layer_sizes[ii],
                    activation=self.activations[ii],
                    use_bias=self.bias,
                    final_layer=ii == (n_layers - 1),
                )
            )
        self.kernels = None

    def build(self, input_dim: int, seed: int = 0) -> "GCN":
        """Build every layer kernel, glorot-uniform, which is upstream's
        `kernel_initializer="glorot_uniform"` default."""
        rng = np.random.default_rng(seed)
        dim = int(input_dim)
        self.kernels = []
        for layer in self._layers:
            layer.build(dim)
            limit = np.sqrt(6.0 / (dim + layer.units))
            layer.kernel[:] = rng.uniform(-limit, limit, size=layer.kernel.shape)
            if layer.bias is not None:
                layer.bias[:] = 0.0
            self.kernels.append(layer.kernel)
            dim = layer.units
        return self

    def __call__(self, features, A, out_indices=None) -> np.ndarray:
        h_layer = f64(features)
        if self.kernels is None:
            self.build(h_layer.shape[1])
        if self.method == "none":
            # Upstream inserts GraphPreProcessingLayer when the generator did
            # no preprocessing of its own.
            from .preprocessing_layer import GraphPreProcessingLayer

            A = GraphPreProcessingLayer(A.shape[0])(A)
        for layer in self._layers:
            h_layer = layer(h_layer, A, out_indices)
        return h_layer

    def node_model(self):
        """Upstream builds the Keras `(x_inp, x_out)` pair. Here the input
        shape is all that is knowable without a built model, so only that is
        returned."""
        if self.generator is None:
            raise RuntimeError("a generator is needed to infer the input size")
        in_shape = (len(self.generator.node_list), self.generator.features.shape[1])
        return (in_shape,), None
