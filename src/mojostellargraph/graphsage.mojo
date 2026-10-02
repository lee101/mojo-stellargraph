"""Port of `stellargraph/layer/graphsage.py` (v0.8.1): the aggregators and
`GraphSAGE.__call__`.

Upstream's aggregators are Keras layers, and the numeric body of each is one
`group_aggregate` method. Those are ported one for one below, in the order
upstream defines them:

    MeanAggregator.group_aggregate
    MaxPoolingAggregator.group_aggregate
    MeanPoolingAggregator.group_aggregate
    AttentionalAggregator.call

Tensors are `[n_batch, n_head, n_neighbour, n_feat]`, which is how a Keras
tensor of that shape is already laid out in memory, so no reshuffling happens
on the way in or out. Group 0 is always the head node's own features and is
never aggregated over, which is why each `group_aggregate` branches on
`group_idx == 0` first.
"""

from mojostellargraph.types import (
    FPtr,
    Vec,
    W,
    activation_inplace,
    dot,
    leaky_relu,
    l2_normalize,
    softmax_dim2,
)


def MeanAggregator_group_aggregate(
    x_group: FPtr,
    w: FPtr,
    result: FPtr,
    work: FPtr,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    out_dim: Int,
    group_idx: Int,
):
    """Upstream `MeanAggregator.group_aggregate`.

    ```
    if group_idx == 0:
        x_agg = x_group          # the first group is the self-tensor
    else:
        x_agg = K.mean(x_group, axis=2)
    return K.dot(x_agg, self.w_group[group_idx])
    """
    if group_idx == 0:
        # `K.dot(x_group, self.w_group[0])` contracts everything below
        # `n_head` at once, so the row width is the whole `s * d` block and
        # not just the feature axis
        dot(x_group, w, result, b * h, s * d, out_dim)
    else:
        # x_agg = K.mean(x_group, axis=2): one vector of the feature axis at
        # a time, the first sample copied and the rest added
        var inv = 1.0 / Float64(s)
        for i in range(b * h):
            var drow = work.unsafe_offset(i * d)
            if s > 0:
                var srow = x_group.unsafe_offset(i * s * d)
                var j = 0
                while j + W <= d:
                    drow.unsafe_store(j, srow.unsafe_load[width=W](j))
                    j += W
                while j < d:
                    drow.unsafe_store(j, srow.unsafe_load(j))
                    j += 1
                var t = 1
                while t < s:
                    srow = srow.unsafe_offset(d)
                    j = 0
                    while j + W <= d:
                        drow.unsafe_store(
                            j, drow.unsafe_load[width=W](j) + srow.unsafe_load[width=W](j)
                        )
                        j += W
                    while j < d:
                        drow.unsafe_store(j, drow.unsafe_load(j) + srow.unsafe_load(j))
                        j += 1
                    t += 1
                j = 0
                while j + W <= d:
                    drow.unsafe_store(j, drow.unsafe_load[width=W](j) * inv)
                    j += W
                while j < d:
                    drow.unsafe_store(j, drow.unsafe_load(j) * inv)
                    j += 1
            else:
                # `s == 0` is `0 / 0` upstream too
                var j = 0
                while j + W <= d:
                    drow.unsafe_store(j, Vec(0.0) * inv)
                    j += W
                while j < d:
                    drow.unsafe_store(j, 0.0 * inv)
                    j += 1
        dot(work, w, result, b * h, d, out_dim)


def _pool_group_aggregate(
    x_group: FPtr,
    w_group: FPtr,
    w_pool: FPtr,
    b_pool: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    hidden_dim: Int,
    out_dim: Int,
    group_idx: Int,
    take_max: Int,
):
    """The shared body of `MaxPoolingAggregator` and `MeanPoolingAggregator`.

    Upstream:
    ```
    if group_idx == 0:
        x_agg = K.dot(x_group, self.w_group[0])
    else:
        w_g, w_pool, b_pool = self.w_group[group_idx]
        xw_neigh = self.hidden_act(K.dot(x_group, w_pool) + b_pool)
        x_agg = K.max(xw_neigh, axis=2)      # MeanPoolingAggregator: K.mean
        x_agg = K.dot(x_agg, w_g)
    return x_agg
    ```

    `take_max` picks the reduction; the two aggregators differ only there.
    """
    if group_idx == 0:
        dot(x_group, w_group, result, b * h, d, out_dim)
    else:
        # xw_neigh = self.hidden_act(K.dot(x_group, w_pool) + b_pool)
        dot(x_group, w_pool, work2, b * h * s, d, hidden_dim)
        for i in range(b * h * s):
            var row = work2.unsafe_offset(i * hidden_dim)
            var j = 0
            while j + W <= hidden_dim:
                row.unsafe_store(
                    j,
                    max(
                        row.unsafe_load[width=W](j) + b_pool.unsafe_load[width=W](j),
                        Vec(0.0),
                    ),
                )
                j += W
            while j < hidden_dim:
                row.unsafe_store(
                    j, max(row.unsafe_load(j) + b_pool.unsafe_load(j), 0.0)
                )
                j += 1
        # x_agg = K.max(xw_neigh, axis=2)   /   K.mean(xw_neigh, axis=2)
        # `s` is the axis being reduced, so the feature axis is the one
        # that vectorizes: `s` rows `hidden_dim` apart per group.
        for i in range(b * h):
            var drow = result.unsafe_offset(i * hidden_dim)
            if s == 0:
                # no neighbours to reduce over: upstream's `K.max` of nothing
                # is `-inf` clamped to its `best = 0.0` seed, and `K.mean` of
                # nothing is `0 / 0`
                var j = 0
                while j < hidden_dim:
                    drow.unsafe_store(j, 0.0 if take_max else 0.0 / 0.0)
                    j += 1
                continue
            var row = work2.unsafe_offset(i * s * hidden_dim)
            var j = 0
            while j + W <= hidden_dim:
                drow.unsafe_store(j, row.unsafe_load[width=W](j))
                j += W
            while j < hidden_dim:
                drow.unsafe_store(j, row.unsafe_load(j))
                j += 1
            var t = 1
            while t < s:
                row = row.unsafe_offset(hidden_dim)
                j = 0
                while j + W <= hidden_dim:
                    var v = row.unsafe_load[width=W](j)
                    var acc = drow.unsafe_load[width=W](j)
                    drow.unsafe_store(j, max(acc, v) if take_max else acc + v)
                    j += W
                while j < hidden_dim:
                    var v = row.unsafe_load(j)
                    var acc = drow.unsafe_load(j)
                    drow.unsafe_store(j, max(acc, v) if take_max else acc + v)
                    j += 1
                t += 1
            if not take_max:
                j = 0
                while j + W <= hidden_dim:
                    drow.unsafe_store(j, drow.unsafe_load[width=W](j) / Float64(s))
                    j += W
                while j < hidden_dim:
                    drow.unsafe_store(j, drow.unsafe_load(j) / Float64(s))
                    j += 1
        # x_agg = K.dot(x_agg, w_g)
        dot(result, w_group, work, b * h, hidden_dim, out_dim)
        var k = 0
        while k + W <= b * h * out_dim:
            result.unsafe_store(k, work.unsafe_load[width=W](k))
            k += W
        while k < b * h * out_dim:
            result.unsafe_store(k, work.unsafe_load(k))
            k += 1


def MaxPoolingAggregator_group_aggregate(
    x_group: FPtr,
    w_group: FPtr,
    w_pool: FPtr,
    b_pool: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    hidden_dim: Int,
    out_dim: Int,
    group_idx: Int,
):
    """Upstream `MaxPoolingAggregator.group_aggregate`, Eq. (3) of
    Hamilton et al. (2017)."""
    _pool_group_aggregate(
        x_group, w_group, w_pool, b_pool, result, work, work2, b, h, s, d,
        hidden_dim, out_dim, group_idx, 1,
    )


def MeanPoolingAggregator_group_aggregate(
    x_group: FPtr,
    w_group: FPtr,
    w_pool: FPtr,
    b_pool: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    hidden_dim: Int,
    out_dim: Int,
    group_idx: Int,
):
    """Upstream `MeanPoolingAggregator.group_aggregate`."""
    _pool_group_aggregate(
        x_group, w_group, w_pool, b_pool, result, work, work2, b, h, s, d,
        hidden_dim, out_dim, group_idx, 0,
    )


def AttentionalAggregator_group_aggregate(
    x_self: FPtr,
    x_g: FPtr,
    w_g: FPtr,
    w_attn_s: FPtr,
    w_attn_g: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    b: Int,
    h: Int,
    s: Int,
    d_self: Int,
    d: Int,
    out_dim: Int,
):
    """One pass of upstream `AttentionalAggregator.call`'s group loop.

    ```
    w_g, w_attn_s, w_attn_g = self.w_group[group_idx]
    xw_self  = K.expand_dims(K.dot(x_self, w_g), axis=2)
    xw_neigh = K.dot(x_g, w_g)
    xw_all   = K.concatenate([xw_self, xw_neigh], axis=2)
    attn_self  = K.dot(xw_self, w_attn_s)                       # (n_b, n_h, 1)
    attn_neigh = K.dot(xw_all, w_attn_g)                        # (n_b, n_h, n_neigh+1, 1)
    attn_u = self.attn_act(attn_self + attn_neigh)              # LeakyReLU(0.2)
    attn = K.softmax(attn_u, axis=2)
    h_out = K.sum(attn * xw_all, axis=2)
    ```

    Implements Velickovic et al., ICLR 2018. `work` is `[b, h, s+1]` and
    `work2` is `3 * b * h * out_dim + b * h * s * out_dim`, laid out as
    `[b*h*out_dim self | b*h*s*out_dim neighbours | b*h*(s+1)*out_dim xw_all]`.
    """
    var bh = b * h
    var self_at = 0
    var neigh_at = bh * out_dim
    var all_at = neigh_at + bh * s * out_dim

    # xw_neigh = K.dot(x_g, w_g)
    dot(x_g, w_g, work2.unsafe_offset(neigh_at), bh * s, d, out_dim)
    # xw_self = K.dot(x_self, w_g)
    dot(x_self, w_g, work2.unsafe_offset(self_at), bh, d_self, out_dim)

    # xw_all = K.concatenate([xw_self, xw_neigh], axis=2)
    for i in range(bh):
        for t in range(s + 1):
            var src = work2.unsafe_offset(
                self_at + i * out_dim if t == 0 else neigh_at + (i * s + t - 1) * out_dim
            )
            var dst = work2.unsafe_offset(all_at + (i * (s + 1) + t) * out_dim)
            var j = 0
            while j + W <= out_dim:
                dst.unsafe_store(j, src.unsafe_load[width=W](j))
                j += W
            while j < out_dim:
                dst.unsafe_store(j, src.unsafe_load(j))
                j += 1

    # attn_neigh = K.dot(xw_all, w_attn_g)
    for i in range(bh):
        for t in range(s + 1):
            var a = 0.0
            var row = all_at + (i * (s + 1) + t) * out_dim
            for j in range(out_dim):
                a += work2.unsafe_load(row + j) * w_attn_g.unsafe_load(j)
            work.unsafe_store(i * (s + 1) + t, a)

    # attn_self = K.dot(xw_self, w_attn_s), added by broadcasting
    for i in range(bh):
        var self_a = 0.0
        for j in range(out_dim):
            self_a += work2.unsafe_load(self_at + i * out_dim + j) * w_attn_s.unsafe_load(j)
        for t in range(s + 1):
            # attn_u = self.attn_act(attn_self + attn_neigh)
            work.unsafe_store(
                i * (s + 1) + t,
                leaky_relu(work.unsafe_load(i * (s + 1) + t) + self_a, 0.2),
            )

    # attn = K.softmax(attn_u, axis=2)
    softmax_dim2(work, result, b, h, s + 1)

    # h_out = K.sum(attn * xw_all, axis=2)
    for i in range(bh):
        var drow = work2.unsafe_offset(self_at + i * out_dim)
        var base = work2.unsafe_offset(all_at + i * (s + 1) * out_dim)
        var attn_at = result.unsafe_offset(i * (s + 1))
        var j = 0
        while j + W <= out_dim:
            var acc = attn_at.unsafe_load(0) * base.unsafe_load[width=W](j)
            var t = 1
            while t < s + 1:
                acc = acc + attn_at.unsafe_load(t) * base.unsafe_offset(t * out_dim).unsafe_load[width=W](j)
                t += 1
            drow.unsafe_store(j, acc)
            j += W
        while j < out_dim:
            var acc = 0.0
            var t = 0
            while t < s + 1:
                acc += attn_at.unsafe_load(t) * base.unsafe_offset(t * out_dim).unsafe_load(j)
                t += 1
            drow.unsafe_store(j, acc)
            j += 1
    var k = 0
    while k + W <= bh * out_dim:
        result.unsafe_store(k, work2.unsafe_load[width=W](self_at + k))
        k += W
    while k < bh * out_dim:
        result.unsafe_store(k, work2.unsafe_load(self_at + k))
        k += 1


def _sources_to_output(
    sources: FPtr,
    bias: FPtr,
    result: FPtr,
    b: Int,
    h: Int,
    width: Int,
    has_bias: Int,
    act: Int,
    alpha: Float64,
):
    """The body both aggregator `call`s share.

    ```
    h_out = K.concatenate(sources, axis=2)
    if self.has_bias:
        h_out = h_out + self.bias
    return self.act(h_out)
    ```

    `width` is the whole concatenated row, and upstream's `self.bias` is that
    wide too, so the bias is added by row over `[0, width)` and never read past.
    The activation writes in place, so it needs no second buffer either.
    """
    if has_bias:
        for r in range(b * h):
            var srow = sources.unsafe_offset(r * width)
            var drow = result.unsafe_offset(r * width)
            var j = 0
            while j + W <= width:
                drow.unsafe_store(
                    j, srow.unsafe_load[width=W](j) + bias.unsafe_load[width=W](j)
                )
                j += W
            while j < width:
                drow.unsafe_store(j, srow.unsafe_load(j) + bias.unsafe_load(j))
                j += 1
    else:
        var k = 0
        while k + W <= b * h * width:
            result.unsafe_store(k, sources.unsafe_load[width=W](k))
            k += W
        while k < b * h * width:
            result.unsafe_store(k, sources.unsafe_load(k))
            k += 1
    activation_inplace(result, b * h, width, act, alpha)


def AttentionalAggregator_call(
    sources: FPtr,
    bias: FPtr,
    result: FPtr,
    b: Int,
    h: Int,
    width: Int,
    has_bias: Int,
    act: Int,
    alpha: Float64,
):
    """Upstream `AttentionalAggregator.call`, from the concatenated group
    outputs onwards.

    ```
    h_out = K.concatenate(group_sources, axis=2)
    if self.has_bias:
        h_out = h_out + self.bias
    return self.act(h_out)
    ```

    The group loop above it is `AttentionalAggregator_group_aggregate`, which
    the Python wrapper runs once per relation; the results arrive here already
    concatenated, which is what `K.concatenate(group_sources, axis=2)` sees.
    """
    _sources_to_output(sources, bias, result, b, h, width, has_bias, act, alpha)


def GraphSAGEAggregator_call(
    sources: FPtr,
    bias: FPtr,
    result: FPtr,
    b: Int,
    h: Int,
    width: Int,
    has_bias: Int,
    act: Int,
    alpha: Float64,
):
    """Upstream `GraphSAGEAggregator.call`.

    ```
    sources = [self.group_aggregate(x, group_idx=ii) for each included group]
    h_out = K.concatenate(sources, axis=2)
    if self.has_bias:
        h_out = h_out + self.bias
    return self.act(h_out)
    ```

    `sources` is the concatenation the upstream `K.concatenate` produces; the
    Python wrapper builds it by running each `group_aggregate` in turn.
    """
    _sources_to_output(sources, bias, result, b, h, width, has_bias, act, alpha)


def GraphSAGE_normalization(x: FPtr, dst: FPtr, n: Int, d: Int, normalize: Int):
    """Upstream `GraphSAGE.__init__`'s `self._normalization`, applied as
    `_normalization(x) if len(h_layer) == 1 else [self._normalization(xi) ...]`.

    `normalize` is 0 for the identity (`None` / `"none"` / `"None"`) and 1 for
    `K.l2_normalize(x, axis=-1)`.
    """
    if normalize == 0:
        var k = 0
        while k < n * d:
            dst.unsafe_store(k, x.unsafe_load(k))
            k += 1
    else:
        l2_normalize(x, dst, n, d)
