"""Port of `stellargraph/layer/graphsage.py` (v0.8.1): the aggregators and
`GraphSAGE`.

Upstream inputs are `[n_batch, n_head, n_neighbour, n_feat]` tensors. They are
taken here as plain row-major NumPy arrays of that shape, so nothing is
reshaped on the way in. Group 0 is the head node's own features; groups 1 and
up are the sampled neighbours for successive hops, which is the layout
`GraphSAGE.__call__`'s `Reshape((head_shape, n_samples[i], dims))` produces.
"""

from __future__ import annotations

import numpy as np

from . import activations as _acts
from ._lib import addr, f64, lib


def _check(act):
    if act not in _acts.CODES:
        raise ValueError("unsupported activation {!r}".format(act))


def _as4(x, what="x_group"):
    """Group 0 arrives as `[n_batch, n_head, n_feat]`, which is the
    `[b, h, 1, d]` the kernels want."""
    if x.ndim == 3:
        return x[:, :, None, :]
    return x


def _shape4(x, what="x_group"):
    if x.ndim != 4:
        raise ValueError(
            "expected a [n_batch, n_head, n_neighbour, n_feat] tensor for {}, got "
            "shape {}".format(what, x.shape)
        )
    return x.shape


class _Aggregator:
    """Shared plumbing for the four upstream aggregators: the `(output_dim,
    bias, act)` constructor arguments, the `hidden_act` that two of them set,
    and the `w_group` dict that `_build_group_weights` fills in."""

    def __init__(self, output_dim, bias: bool = True, act: str = "relu"):
        _check(act)
        self.output_dim = int(output_dim)
        self.has_bias = bool(bias)
        self.act = act
        self.bias = np.zeros(self.output_dim) if self.has_bias else None
        self.w_group: dict[int, np.ndarray] = {}
        self.w_pool: dict[int, np.ndarray] = {}
        self.b_pool: dict[int, np.ndarray] = {}

    def _build_group_weights(self, in_dim, group_idx, out_size=None):
        """`_build_group_weights`: one weight of `(in_dim, out_size)` per group,
        glorot-uniform as upstream's `kernel_initializer` default."""
        out_size = self.output_dim if out_size is None else int(out_size)
        in_dim = int(in_dim)
        rng = np.random.default_rng(1000 + group_idx)
        if isinstance(self, _PoolingAggregator) and group_idx > 0:
            # upstream: w_group for a neighbour group is (hidden_dim, out_size)
            self.hidden_dim = self.output_dim
            lim = np.sqrt(6.0 / (self.hidden_dim + out_size))
            w = rng.uniform(-lim, lim, size=(self.hidden_dim, out_size))
            lim2 = np.sqrt(6.0 / (in_dim + self.hidden_dim))
            self.w_pool[group_idx] = rng.uniform(
                -lim2, lim2, size=(in_dim, self.hidden_dim)
            )
            self.b_pool[group_idx] = np.zeros(self.hidden_dim)
        else:
            lim = np.sqrt(6.0 / (in_dim + out_size))
            w = rng.uniform(-lim, lim, size=(in_dim, out_size))
            if isinstance(self, _PoolingAggregator):
                # a head-node group only needs w_group; the pool weights are
                # read for neighbour groups only
                self.w_pool.setdefault(group_idx, np.zeros((in_dim, self.output_dim)))
                self.b_pool.setdefault(group_idx, np.zeros(self.output_dim))
        self.w_group[group_idx] = w
        return w


class MeanAggregator(_Aggregator):
    """
    Mean Aggregator for GraphSAGE.

    ```
    MeanAggregator(output_dim, bias=True, act="relu")
    ```
    """

    def group_aggregate(self, x_group, group_idx) -> np.ndarray:
        """`MeanAggregator.group_aggregate`:
        `K.dot(K.mean(x_group, axis=2), w)` for a neighbour group and
        `K.dot(x_group, w)` for the head node group."""
        x_group = _as4(f64(x_group))
        b, h, s, d = _shape4(x_group)
        w = f64(self.w_group[group_idx])
        result = np.zeros((b * h, self.output_dim), dtype=np.float64)
        work = np.zeros(max(b * h * max(d, 1), 1), dtype=np.float64)
        lib().msg_MeanAggregator_group_aggregate(
            addr(x_group.reshape(-1)), addr(w), addr(result), addr(work),
            b, h, s, d, self.output_dim, group_idx,
        )
        return result.reshape(b, h, self.output_dim)

    def call(self, sources) -> np.ndarray:
        """`GraphSAGEAggregator.call` on the concatenated group outputs."""
        return _aggregator_call(sources, self.bias, self.has_bias, self.act)


class _PoolingAggregator(_Aggregator):
    """Shared body of `MaxPoolingAggregator` and `MeanPoolingAggregator`; the
    two differ only in the reduction over the neighbour axis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: These should be user parameters
        self.hidden_dim = self.output_dim
        self.hidden_act = "relu"

    def _reduce(self, x_group, group_idx, take_max, w, w_pool, b_pool):
        x_group = _as4(f64(x_group))
        b, h, s, d = _shape4(x_group)
        hidden = self.hidden_dim
        out = np.zeros((b * h, self.output_dim), dtype=np.float64)
        scratch = np.zeros(max(b * h * self.output_dim, 1), dtype=np.float64)
        pooled = np.zeros(max(b * h * s * hidden, 1), dtype=np.float64)
        fn = (
            lib().msg_MaxPoolingAggregator_group_aggregate
            if take_max
            else lib().msg_MeanPoolingAggregator_group_aggregate
        )
        wk, wpk, bpk = f64(w), f64(w_pool), f64(b_pool)
        fn(
            addr(x_group.reshape(-1)), addr(wk), addr(wpk),
            addr(bpk), addr(out), addr(scratch), addr(pooled),
            b, h, s, d, hidden, self.output_dim, group_idx,
        )
        return out.reshape(b, h, self.output_dim)

    def call(self, sources) -> np.ndarray:
        return _aggregator_call(sources, self.bias, self.has_bias, self.act)


class MaxPoolingAggregator(_PoolingAggregator):
    """
    Max Pooling Aggregator, Eq. (3) of Hamilton et al. (2017).

    ```
    MaxPoolingAggregator(output_dim, bias=True, act="relu")
    ```
    """

    def group_aggregate(self, x_group, group_idx) -> np.ndarray:
        """`MaxPoolingAggregator.group_aggregate`."""
        if group_idx == 0:
            return _plain_dot(_as4(f64(x_group)), f64(self.w_group[0]))
        return self._reduce(
            x_group, group_idx, 1, self.w_group[group_idx],
            self.w_pool[group_idx], self.b_pool[group_idx],
        )


class MeanPoolingAggregator(_PoolingAggregator):
    """
    Mean Pooling Aggregator, Eq. (3) of Hamilton et al. (2017) with the max
    replaced by the mean.

    ```
    MeanPoolingAggregator(output_dim, bias=True, act="relu")
    ```
    """

    def group_aggregate(self, x_group, group_idx) -> np.ndarray:
        """`MeanPoolingAggregator.group_aggregate`."""
        if group_idx == 0:
            return _plain_dot(_as4(f64(x_group)), f64(self.w_group[0]))
        return self._reduce(
            x_group, group_idx, 0, self.w_group[group_idx],
            self.w_pool[group_idx], self.b_pool[group_idx],
        )


def _plain_dot(x_group, w):
    b, h, s, d = _shape4(x_group)
    return (x_group.reshape(b * h, s * d) @ w).reshape(b, h, w.shape[1])


class AttentionalAggregator(_Aggregator):
    """
    Attentional Aggregator, implementing Veličković et al. "Graph Attention
    Networks", ICLR 2018.

    ```
    AttentionalAggregator(output_dim, bias=True, act="relu")
    ```

    Unlike the other aggregators, this one gives the head node group no weight
    of its own: self and neighbours share one attention distribution per group,
    which is upstream's `calculate_group_sizes` with `weight_dims[0] = 0`.
    """

    def __init__(self, output_dim, bias: bool = True, act: str = "relu"):
        super().__init__(output_dim, bias=bias, act=act)
        # TODO: How can we expose these options to the user?
        self.attn_act = "leaky_relu"

    def _build_group_weights(self, in_dim, group_idx, out_size=None):
        """`_build_group_weights`: `w_g` `(in_dim, out)`, and the two attention
        kernels `w_attn_s` and `w_attn_g`, both `(out, 1)`."""
        out_size = self.output_dim if out_size is None else int(out_size)
        rng = np.random.default_rng(1000 + group_idx)
        lim = np.sqrt(6.0 / (int(in_dim) + out_size))
        w = rng.uniform(-lim, lim, size=(int(in_dim), out_size))
        self.w_group[group_idx] = w
        self.w_attn_s = rng.uniform(-lim, lim, size=(out_size, 1))
        self.w_attn_g = rng.uniform(-lim, lim, size=(out_size, 1))
        return w

    def group_aggregate(self, x_self, x_g, group_idx, out_size=None) -> np.ndarray:
        """One pass of upstream's group loop in `AttentionalAggregator.call`.

        `out_size` is this group's share of `output_dim`; it is
        `self.output_dim` for a single-group call, and `output_dim //
        n_groups` for the multi-group `call` below, which is upstream's
        `calculate_group_sizes` split."""
        out_size = self.output_dim if out_size is None else int(out_size)
        x_self = f64(x_self)
        x_g = f64(x_g)
        b, h, _, d_self = _shape4(_as4(x_self), "x_self")
        _, _, s, d = _shape4(x_g, "x_g")
        # `out` holds the softmax `attn` tensor first and `h_out` second, so it
        # must cover whichever of the two is wider: `attn` is
        # `[b, h, n_neighbour + 1, 1]`, which is upstream's own tensor
        out = np.zeros(max(b * h * out_size, b * h * (s + 1)), dtype=np.float64)
        attn = np.zeros(max(b * h * (s + 1), 1), dtype=np.float64)
        # work2 is [self | neighbours | xw_all], see the kernel's docstring
        pooled = np.zeros(
            max(b * h * out_size * (2 * s + 2), 1), dtype=np.float64
        )
        w_g, w_s, w_attn_g = (
            f64(self.w_group[group_idx]), f64(self.w_attn_s), f64(self.w_attn_g)
        )
        lib().msg_AttentionalAggregator_group_aggregate(
            addr(x_self.reshape(-1)), addr(x_g.reshape(-1)),
            addr(w_g), addr(w_s), addr(w_attn_g),
            addr(out), addr(attn), addr(pooled),
            b, h, s, d_self, d, out_size,
        )
        # `out` doubles as the `attn` staging buffer, so it can be longer than
        # the group output
        return out[: b * h * out_size].reshape(b, h, out_size)

    def call(self, x_self, x_groups) -> np.ndarray:
        """`AttentionalAggregator.call` over every group.

        `x_groups` is the `[n_groups, n_batch, n_head, n_neighbour, n_feat]`
        stack of the `inputs[1:]` tensors upstream loops over.
        """
        x_self = f64(x_self)
        x_groups = f64(x_groups)
        if x_self.ndim == 3:
            x_self = x_self[:, :, None, :]
        b, h, _, d_self = x_self.shape
        n_groups, _, _, s, d = x_groups.shape
        # `if not group_sources: group_sources = [K.dot(x_self, w_group[0])]`,
        # upstream's num_groups == 0 branch, which gives the head group the
        # whole output width when there are no neighbour groups
        if n_groups == 0:
            # `w_group[0]` is only built when there is a head group to run
            if 0 not in self.w_group:
                self._build_group_weights(d_self, 0, self.output_dim)
            wd = self.output_dim
            sources = self.group_aggregate(x_self, _as4(x_self), 0, wd)
            n_groups = 1
        else:
            # upstream's `calculate_group_sizes` splits `output_dim` across the
            # groups, giving the remainder to the first, so the concatenation
            # is `output_dim` wide and the one bias lines up with it
            share = self.output_dim // n_groups
            rem = self.output_dim - share * n_groups
            sources = np.zeros((b, h, self.output_dim), dtype=np.float64)
            at = 0
            for r in range(n_groups):
                w = share + (rem if r == 0 else 0)
                sources[:, :, at : at + w] = self.group_aggregate(
                    x_self, x_groups[r], r + 1, w
                )
                at += w
        out = np.zeros_like(sources)
        bias_arr = f64(self.bias).reshape(-1)
        if self.has_bias and bias_arr.size != sources.shape[2]:
            raise ValueError(
                "bias has {} elements, the concatenated groups have {}".format(
                    bias_arr.size, sources.shape[2]
                )
            )
        lib().msg_AttentionalAggregator_call(
            addr(sources.reshape(-1)), addr(bias_arr),
            addr(out), b, h, sources.shape[2],
            1 if self.has_bias else 0, _acts.code(self.act), _acts.alpha(self.act),
        )
        return out


def _aggregator_call(sources, bias, has_bias, act) -> np.ndarray:
    """`GraphSAGEAggregator.call`: concatenate (already done by the caller),
    add the bias, apply the activation."""
    sources = f64(sources)
    b, h, d = sources.shape
    out = np.zeros((b, h, d), dtype=np.float64)
    bias_arr = f64(bias).reshape(-1)
    if has_bias and bias_arr.size != d:
        raise ValueError(
            "bias has {} elements, the concatenated groups have {}".format(
                bias_arr.size, d
            )
        )
    lib().msg_GraphSAGEAggregator_call(
        addr(sources.reshape(-1)), addr(bias_arr), addr(out),
        b, h, d, 1 if has_bias else 0, _acts.code(act), _acts.alpha(act),
    )
    return out


class GraphSAGE:
    """
    GraphSAGE, see http://snap.stanford.edu/graphsage/

    ```
    GraphSAGE(layer_sizes, generator=None, aggregator=MeanAggregator,
              bias=True, dropout=0.0, normalize="l2", activations=None)
    ```

    `build(group_dims)` takes, per layer, the feature width of each group; the
    weight sizes then follow upstream's `calculate_group_sizes`, which gives the
    remainder of `output_dim // num_groups` to the first enabled neighbour group
    and none to the head node group.
    """

    def __init__(self, layer_sizes, generator=None, aggregator=None, bias: bool = True,
                 dropout: float = 0.0, normalize: str | None = "l2",
                 activations=None, **kwargs):
        self.layer_sizes = list(layer_sizes)
        self.max_hops = len(self.layer_sizes)
        self.bias = bool(bias)
        self.dropout = dropout
        if normalize not in ("l2", None, "none", "None"):
            raise ValueError(
                "unsupported normalize {!r}; GraphSAGE takes 'l2' or None".format(
                    normalize
                )
            )
        self.normalize = normalize
        if activations is None:
            acts = ["relu"] * (self.max_hops - 1) + ["linear"]
        elif len(activations) != self.max_hops:
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )
        else:
            acts = list(activations)
        for a in acts:
            _check(a)
        self.activations = acts
        self.aggregator = aggregator if aggregator is not None else MeanAggregator
        self.generator = generator
        self.n_samples = list(getattr(generator, "num_samples", []) or [])
        self._aggs: list[list] = []

    def build(self, group_dims=None, seed: int = 0) -> "GraphSAGE":
        """One aggregator group set per layer and head, sized as upstream's
        `calculate_group_sizes` prescribes.

        `group_dims[layer][i]` is the `(self_width, neighbour_width)` pair for
        head `i` of `layer`; without it the widths are taken from the first
        call, which is what a Keras model does when it sees the input shape.
        """
        self._aggs = []
        self._seed = seed
        if group_dims is not None:
            for layer, pairs in enumerate(group_dims):
                row = []
                for i, (d_self, d_neigh) in enumerate(pairs):
                    row.append(self._build_layer(layer, d_self, d_neigh))
                self._aggs.append(row)
        return self

    def _build_layer(self, layer, d_self, d_neigh):
        """The aggregator list for one `(layer, head)` pair."""
        out_dim = self.layer_sizes[layer]
        if issubclass(self.aggregator, AttentionalAggregator):
            # upstream's `calculate_group_sizes` gives the head node group no
            # weight of its own: `weight_dims[0] = 0`
            dims = [(1, int(d_neigh))]
        else:
            dims = [(0, int(d_self)), (1, int(d_neigh))]
        # `compute_output_shape` returns `self.output_dim`, so `output_dim` is
        # split across the groups, with the remainder on the first.
        num_groups = len(dims)
        group_output_dim = out_dim // num_groups
        remainder = out_dim - num_groups * group_output_dim
        aggs = []
        for g, (group_idx, in_dim) in enumerate(dims):
            wd = group_output_dim + (remainder if g == 0 else 0)
            agg = self.aggregator(wd, bias=self.bias, act=self.activations[layer])
            agg._build_group_weights(in_dim, group_idx, wd)
            if agg.has_bias:
                # the bias is added to the concatenation, so it spans the
                # layer's whole output width
                agg.bias = np.zeros(out_dim, dtype=np.float64)
            aggs.append(agg)
        return aggs

    def _ensure(self, layer, i, d_self, d_neigh):
        while len(self._aggs) <= layer:
            self._aggs.append([])
        row = self._aggs[layer]
        if i >= len(row):
            row.append(self._build_layer(layer, d_self, d_neigh))
        return row[i]

    def __call__(self, xin):
        """
        ```
        h_layer = xin
        for layer in range(0, self.max_hops):
            h_layer = apply_layer(h_layer, layer)
        h_layer = [Reshape(K.int_shape(x)[2:])(x) if K.int_shape(x)[1] == 1 else x for x in h_layer]
        return self._normalization(h_layer[0]) if len(h_layer) == 1 else [...]
        ```
        """
        if not isinstance(xin, (list, tuple)):
            raise TypeError("Input features to GraphSAGE must be a list")
        if len(xin) != self.max_hops + 1:
            raise ValueError(
                "Length of input features should equal the number of "
                "GraphSAGE layers plus one"
            )

        h_layer = [f64(x) for x in xin]
        for layer in range(self.max_hops):
            layer_out = []
            for i in range(self.max_hops - layer):
                head_shape = h_layer[i].shape[1]
                neigh_in = h_layer[i + 1].reshape(
                    h_layer[i].shape[0], head_shape, -1, h_layer[i + 1].shape[-1]
                )
                # The head-node group is contracted over the feature axis
                # only, so its input width is everything below `n_head`.
                aggs = self._ensure(
                    layer, i, int(np.prod(h_layer[i].shape[2:])),
                    h_layer[i + 1].shape[-1],
                )
                layer_out.append(self._apply_aggregator(aggs, h_layer[i], neigh_in))
            h_layer = layer_out

        outs = []
        for x in h_layer:
            # `Reshape(K.int_shape(x)[2:])(x) if K.int_shape(x)[1] == 1 else x`,
            # with the batch axis the Keras tensor keeps and the squeezed
            # representation does not
            if x.shape[1] == 1:
                x = x.reshape((x.shape[0],) + x.shape[2:])
            outs.append(self._normalization(x))
        return outs[0] if len(outs) == 1 else outs

    def _apply_aggregator(self, aggs, x_self, neigh_in):
        """One pass of upstream's `apply_layer` inner loop: the head node's own
        features and its sampled neighbours go into the layer's aggregator.

        Upstream's group 0 is `K.dot(x[i], w)`, which contracts the feature axis
        and drops one rank. Here the self group is contracted over everything
        below `n_head`, which is the same operation when that axis has size one
        -- the case for head 0, where `neighbourhood_sizes[0] == 1`. See the
        README for the divergence on the other heads."""
        b, h = x_self.shape[0], x_self.shape[1]
        if isinstance(aggs[0], AttentionalAggregator):
            return aggs[0].call(x_self, neigh_in[None, ...])
        # the head group is contracted over everything below `n_head` at once;
        # `_build_layer` sized `w` with that same product
        parts = [aggs[0].group_aggregate(x_self, 0)]
        parts += [
            agg.group_aggregate(neigh_in, g) for g, agg in enumerate(aggs[1:], 1)
        ]
        return aggs[0].call(np.concatenate(parts, axis=2))

    def _normalization(self, x):
        """`self._normalization`, applied as upstream applies it."""
        x = f64(x)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        dst = np.zeros_like(flat)
        lib().msg_GraphSAGE_normalization(
            addr(flat), addr(dst), flat.shape[0], flat.shape[1],
            0 if self.normalize in (None, "none", "None") else 1,
        )
        return dst.reshape(shape)
