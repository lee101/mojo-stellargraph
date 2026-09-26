"""Port of `stellargraph/layer/hinsage.py` (v0.8.1):
`MeanHinAggregator.call` and `HinSAGE`.

HinSAGE is GraphSAGE over a heterogeneous schema: the same aggregation runs
once per relation and the per-relation means are averaged, so the self and
neighbour halves of the output are concatenated.
"""

from __future__ import annotations

import numpy as np

from . import activations as _acts
from ._lib import addr, f64, lib


class MeanHinAggregator:
    """
    MeanHinAggregator(output_dim, nr, activation="relu", bias=True)

    Args:
        output_dim (int): the total output size; half of it goes to the self
            features and half to the neighbourhood average, as upstream's
            `half_output_dim` is `output_dim // 2`.
        nr (int): the number of relations (edge types).
    """

    def __init__(self, output_dim, nr, activation: str = "relu", bias: bool = True):
        if activation not in _acts.CODES:
            raise ValueError("unsupported activation {!r}".format(activation))
        self.output_dim = int(output_dim)
        # `assert output_dim % 2 == 0`, upstream: the kernel packs the
        # concatenated self and neighbour halves into `2 * half` columns
        assert self.output_dim % 2 == 0
        self.half_output_dim = self.output_dim // 2
        self.nr = int(nr)
        self.act = activation
        self.has_bias = bool(bias)
        self.bias = np.zeros(self.output_dim) if self.has_bias else None

    def build(self, d_self, d_neigh, seed: int = 0):
        """`w_self` `(d_self, half)`, `w_neigh` `[nr, d_neigh, half]`, and a
        `(output_dim,)` bias, glorot-uniform as upstream's default."""
        rng = np.random.default_rng(seed)
        h = self.half_output_dim
        lim = np.sqrt(6.0 / (int(d_self) + h))
        self.w_self = rng.uniform(-lim, lim, size=(int(d_self), h))
        lim = np.sqrt(6.0 / (int(d_neigh) + h))
        self.w_neigh = rng.uniform(-lim, lim, size=(self.nr, int(d_neigh), h))
        return self

    def call(self, x_self, x_neigh) -> np.ndarray:
        """`MeanHinAggregator.call`.

        `x_self` is `[n_batch, n_head, d_self]`. `x_neigh` is the
        `[nr, n_batch, n_head, n_neighbour, d_neigh]` stack of the per-relation
        neighbour tensors upstream indexes as `x[1 + r]`.
        """
        x_self = f64(x_self)
        x_neigh = f64(x_neigh)
        b, h, d_self = x_self.shape
        nr, _, _, s, d = x_neigh.shape
        if nr != self.nr:
            raise ValueError(
                "expected {} relations, got {}".format(self.nr, nr)
            )
        out = np.zeros((b, h, self.output_dim), dtype=np.float64)
        # work holds the [b*h, d_neigh] neighbour means, then from_self and
        # from_neigh, then the activation
        work = np.zeros(max(b * h * max(self.output_dim, d), 1), dtype=np.float64)
        # upstream's `neigh_agg_by_relation` list, one block per relation
        scratch = np.zeros(max(nr * b * h * self.half_output_dim, 1), dtype=np.float64)
        bias = self.bias if self.has_bias else np.zeros(max(self.output_dim, 1))
        w_self = f64(self.w_self)
        w_neigh = f64(self.w_neigh).reshape(-1)
        lib().msg_MeanHinAggregator_call(
            addr(x_self.reshape(-1)), addr(x_neigh.reshape(-1)),
            addr(w_self), addr(w_neigh),
            addr(bias), addr(out), addr(work), addr(scratch), b, h, nr, s,
            d_self, d,
            self.half_output_dim, 1 if self.has_bias else 0,
            _acts.code(self.act), _acts.alpha(self.act),
        )
        return out

    def __call__(self, x_self, x_neigh) -> np.ndarray:
        return self.call(x_self, x_neigh)


class HinSAGE:
    """
    HinSAGE, the heterogeneous GraphSAGE of Hu et al.

    ```
    HinSAGE(layer_sizes, generator, nr, activation="relu", bias=True,
            normalize="l2", activations=None)
    ```

    `__call__` applies one `MeanHinAggregator` per layer, in the layer order
    `layer_sizes` gives, then upstream's optional `l2` normalization over the
    last axis.
    """

    def __init__(self, layer_sizes, generator, nr, activation: str = "relu",
                 bias: bool = True, normalize: str | None = "l2",
                 activations=None, **kwargs):
        self.layer_sizes = list(layer_sizes)
        self.generator = generator
        self.nr = int(nr)
        self.activation = activation
        self.bias = bool(bias)
        if normalize == "l2":
            self.normalize = "l2"
        elif normalize in (None, "none", "None"):
            self.normalize = None
        else:
            raise ValueError(
                "Normalization should be either 'l2' or 'none'; received "
                "'{}'".format(normalize)
            )
        # `activations = ["relu"] * (n_layers - 1) + ["linear"]` upstream: the
        # last layer is linear, so a plain `activation` string only sets the
        # hidden layers
        if activations is None:
            acts = [activation] * (len(self.layer_sizes) - 1) + ["linear"]
        elif len(activations) != len(self.layer_sizes):
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )
        else:
            acts = list(activations)
        for a in acts:
            if a not in _acts.CODES:
                raise ValueError("unsupported activation {!r}".format(a))
        self.activations = acts
        self._aggs = []

    def build(self, dims, seed: int = 0):
        """`dims[layer]` is `(d_self, d_neigh)` for that layer."""
        self._aggs = [
            MeanHinAggregator(size, self.nr, self.activations[i], self.bias).build(
                dims[i][0], dims[i][1], seed + i
            )
            for i, size in enumerate(self.layer_sizes)
        ]
        return self

    def _normalization(self, x):
        """`K.l2_normalize(x, axis=-1)`, upstream's `self._normalization`."""
        x = np.ascontiguousarray(f64(x))
        if self.normalize is None:
            return x
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        dst = np.zeros_like(flat)
        lib().msg_GraphSAGE_normalization(
            addr(flat), addr(dst), flat.shape[0], flat.shape[1], 1
        )
        return dst.reshape(shape)

    def __call__(self, head, neighbourhoods):
        """`head` is the `[n_batch, n_head, d_self]` self tensor and
        `neighbourhoods[i]` is the `[nr, n_batch, n_head, n_neighbour, d_neigh]`
        stack for layer `i` -- upstream's `x[1 + r]` for each relation `r`.
        """
        if not self._aggs:
            raise RuntimeError("HinSAGE.build must be called before use")
        if len(neighbourhoods) != len(self._aggs):
            raise ValueError(
                "expected {} neighbourhood stacks, got {}".format(
                    len(self._aggs), len(neighbourhoods)
                )
            )
        h = head
        for agg, rel in zip(self._aggs, neighbourhoods):
            h = agg(h, rel)
        return self._normalization(h)
