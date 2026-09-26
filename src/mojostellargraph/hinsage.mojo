"""Port of `stellargraph/layer/hinsage.py` (v0.8.1): `MeanHinAggregator.call`.

HinSAGE is GraphSAGE over a heterogeneous schema, so the same aggregation runs
once per relation (edge type) and the per-relation means are averaged. The
`z.shape[2] > 0` branch — a relation with no sampled neighbours contributes a
synthetic zero vector — is kept.
"""

from mojostellargraph.types import FPtr, W, activation_inplace, dot


def MeanHinAggregator_call(
    x_self: FPtr,
    x_neigh: FPtr,
    w_self: FPtr,
    w_neigh: FPtr,
    bias: FPtr,
    result: FPtr,
    work: FPtr,
    scratch: FPtr,
    b: Int,
    h: Int,
    nr: Int,
    s: Int,
    d_self: Int,
    d: Int,
    half_output_dim: Int,
    has_bias: Int,
    act: Int,
    alpha: Float64,
):
    """Upstream `MeanHinAggregator.call`.

    ```
    neigh_agg_by_relation = []
    for r in range(self.nr):
        z = x[1 + r]
        if z.shape[2] > 0:
            z_agg = K.dot(K.mean(z, axis=2), self.w_neigh[r])
        else:
            z_agg = tf.zeros((z_shape[0], z_shape[1], w_shape))
        neigh_agg_by_relation.append(z_agg)

    from_self  = K.dot(x[0], self.w_self)
    from_neigh = sum(neigh_agg_by_relation) / self.nr
    total      = K.concatenate([from_self, from_neigh], axis=2)
    return self.act((total + self.bias) if self.has_bias else total)
    ```

    `x_neigh` is the `[nr, b, h, s, d]` stack of the per-relation neighbour
    tensors upstream indexes as `x[1 + r]`. `work` is `b * h * output_dim`,
    holding `from_self` then `from_neigh`; `scratch` is the
    `neigh_agg_by_relation` list upstream keeps in Python, `nr * b * h * half`
    entries.
    """
    for r in range(nr):
        var base = r * b * h * s * d
        if s > 0:
            for i in range(b * h):
                var drow = work.unsafe_offset(i * d)
                var nrow = x_neigh.unsafe_offset(base + i * s * d)
                var j = 0
                while j + W <= d:
                    drow.unsafe_store(j, nrow.unsafe_load[width=W](j))
                    j += W
                while j < d:
                    drow.unsafe_store(j, nrow.unsafe_load(j))
                    j += 1
                var t = 1
                while t < s:
                    nrow = nrow.unsafe_offset(d)
                    j = 0
                    while j + W <= d:
                        drow.unsafe_store(
                            j, drow.unsafe_load[width=W](j) + nrow.unsafe_load[width=W](j)
                        )
                        j += W
                    while j < d:
                        drow.unsafe_store(j, drow.unsafe_load(j) + nrow.unsafe_load(j))
                        j += 1
                    t += 1
                j = 0
                while j + W <= d:
                    drow.unsafe_store(j, drow.unsafe_load[width=W](j) / Float64(s))
                    j += W
                while j < d:
                    drow.unsafe_store(j, drow.unsafe_load(j) / Float64(s))
                    j += 1
            dot(
                work,
                w_neigh + r * d * half_output_dim,
                scratch + r * b * h * half_output_dim,
                b * h,
                d,
                half_output_dim,
            )
        else:
            # z_agg = tf.zeros((z_shape[0], z_shape[1], w_shape))
            var z = 0
            while z < b * h * half_output_dim:
                scratch.unsafe_store(r * b * h * half_output_dim + z, 0.0)
                z += 1

    # from_self = K.dot(x[0], self.w_self)
    dot(x_self, w_self, work, b * h, d_self, half_output_dim)

    # from_neigh = sum(neigh_agg_by_relation) / self.nr
    var bh = b * h
    for i in range(bh):
        var drow = work.unsafe_offset(bh * half_output_dim + i * half_output_dim)
        var srow = scratch.unsafe_offset(i * half_output_dim)
        var j = 0
        while j + W <= half_output_dim:
            drow.unsafe_store(j, srow.unsafe_load[width=W](j))
            j += W
        while j < half_output_dim:
            drow.unsafe_store(j, srow.unsafe_load(j))
            j += 1
        var r = 1
        while r < nr:
            srow = srow.unsafe_offset(bh * half_output_dim)
            j = 0
            while j + W <= half_output_dim:
                drow.unsafe_store(
                    j, drow.unsafe_load[width=W](j) + srow.unsafe_load[width=W](j)
                )
                j += W
            while j < half_output_dim:
                drow.unsafe_store(j, drow.unsafe_load(j) + srow.unsafe_load(j))
                j += 1
            r += 1
        j = 0
        while j + W <= half_output_dim:
            drow.unsafe_store(j, drow.unsafe_load[width=W](j) / Float64(nr))
            j += W
        while j < half_output_dim:
            drow.unsafe_store(j, drow.unsafe_load(j) / Float64(nr))
            j += 1

    # total = K.concatenate([from_self, from_neigh], axis=2)
    for i in range(bh):
        var self_row = work.unsafe_offset(i * half_output_dim)
        var neigh_row = work.unsafe_offset(bh * half_output_dim + i * half_output_dim)
        var drow = result.unsafe_offset(i * 2 * half_output_dim)
        var j = 0
        while j + W <= half_output_dim:
            drow.unsafe_store(j, self_row.unsafe_load[width=W](j))
            drow.unsafe_store(j + half_output_dim, neigh_row.unsafe_load[width=W](j))
            j += W
        while j < half_output_dim:
            drow.unsafe_store(j, self_row.unsafe_load(j))
            drow.unsafe_store(j + half_output_dim, neigh_row.unsafe_load(j))
            j += 1

    # return self.act((total + self.bias) if self.has_bias else total)
    if has_bias:
        for r in range(bh):
            var drow = result.unsafe_offset(r * 2 * half_output_dim)
            var j = 0
            while j + 2 * W <= 2 * half_output_dim:
                drow.unsafe_store(
                    j, drow.unsafe_load[width=W](j) + bias.unsafe_load[width=W](j)
                )
                drow.unsafe_store(
                    j + W,
                    drow.unsafe_load[width=W](j + W) + bias.unsafe_load[width=W](j + W),
                )
                j += 2 * W
            while j < 2 * half_output_dim:
                drow.unsafe_store(j, drow.unsafe_load(j) + bias.unsafe_load(j))
                j += 1
    activation_inplace(result, b * h, 2 * half_output_dim, act, alpha)
