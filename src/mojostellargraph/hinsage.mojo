"""Port of `stellargraph/layer/hinsage.py` (v0.8.1): `MeanHinAggregator.call`.

HinSAGE is GraphSAGE over a heterogeneous schema, so the same aggregation runs
once per relation (edge type) and the per-relation means are averaged. The
`z.shape[2] > 0` branch — a relation with no sampled neighbours contributes a
synthetic zero vector — is kept.
"""

from mojostellargraph.types import FPtr, activation, dot


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
                for j in range(d):
                    var acc = 0.0
                    for t in range(s):
                        acc += x_neigh.unsafe_load(base + (i * s + t) * d + j)
                    work.unsafe_store(i * d + j, acc / Float64(s))
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
                scratch.unsafe_store(z, 0.0)
                z += 1

    # from_self = K.dot(x[0], self.w_self)
    dot(x_self, w_self, work, b * h, d_self, half_output_dim)

    # from_neigh = sum(neigh_agg_by_relation) / self.nr
    var bh = b * h
    for i in range(bh):
        for j in range(half_output_dim):
            var acc = 0.0
            for r in range(nr):
                acc += scratch.unsafe_load(
                    r * bh * half_output_dim + i * half_output_dim + j
                )
            work.unsafe_store(
                bh * half_output_dim + i * half_output_dim + j, acc / Float64(nr)
            )

    # total = K.concatenate([from_self, from_neigh], axis=2)
    for i in range(bh):
        for j in range(half_output_dim):
            result.unsafe_store(
                i * 2 * half_output_dim + j,
                work.unsafe_load(i * half_output_dim + j),
            )
            result.unsafe_store(
                i * 2 * half_output_dim + half_output_dim + j,
                work.unsafe_load(bh * half_output_dim + i * half_output_dim + j),
            )

    # return self.act((total + self.bias) if self.has_bias else total)
    if has_bias:
        for i in range(b * h * 2 * half_output_dim):
            result.unsafe_store(i, result.unsafe_load(i) + bias.unsafe_load(i % (2 * half_output_dim)))
    activation(result, work, b * h, 2 * half_output_dim, act, alpha)
    var o = 0
    while o < b * h * 2 * half_output_dim:
        result.unsafe_store(o, work.unsafe_load(o))
        o += 1
