"""Port of `stellargraph/layer/graph_attention.py` (v0.8.1):
`GraphAttention.call` (dense) and `GraphAttentionSparse.call` (sparse).

`GraphAttention.call` is the densest numeric body in `stellargraph` and is
reproduced statement for statement, head loop and all, including the
`saliency_map_support` branch and the `-10e9 * (1 - A)` mask.

Upstream works in TensorFlow float32 and lets Keras carry the batch dimension.
The batch dimension is added back in the Python wrapper; the buffers here are
the squeezed `[n, f]` and `[n, n]` blocks. Everything is float64.
"""

from std.math import exp

from mojostellargraph.sparse import sparse_dense_matmul, sparse_softmax
from mojostellargraph.types import (
    FPtr,
    IPtr,
    activation,
    dot,
    dot_vec,
    iget,
    iput,
    leaky_relu,
    softmax_rows,
)

comptime HEADS_REDUCTION_CONCAT = 0
comptime HEADS_REDUCTION_AVERAGE = 1


def _head_dense(
    x: FPtr,
    a: FPtr,
    kernel: FPtr,
    attn_kernel: FPtr,
    bias: FPtr,
    result: FPtr,
    features: FPtr,
    attn_self: FPtr,
    attn_neighs: FPtr,
    dense: FPtr,
    n: Int,
    f: Int,
    units: Int,
    use_bias: Int,
    saliency_map_support: Int,
    delta: Float64,
    non_exist_edge: Float64,
):
    """One pass of upstream's `for head in range(self.attn_heads)` body in the
    dense `GraphAttention.call`, with the Keras `Dropout`s at inference."""
    # features = K.dot(X, kernel)
    dot(x, kernel, features, n, f, units)

    # attn_for_self  = K.dot(features, attention_kernel[0])   (n x 1)
    # attn_for_neighs = K.dot(features, attention_kernel[1])  (n x 1)
    dot_vec(features, attn_kernel, attn_self, n, units)
    dot_vec(features, attn_kernel + units, attn_neighs, n, units)

    # dense = attn_for_self + K.transpose(attn_for_neighs)    (n x n)
    for i in range(n):
        for j in range(n):
            dense.unsafe_store(
                i * n + j, attn_self.unsafe_load(i) + attn_neighs.unsafe_load(j)
            )

    # dense = LeakyReLU(alpha=0.2)(dense)
    for i in range(n * n):
        dense.unsafe_store(i, leaky_relu(dense.unsafe_load(i), 0.2))

    if not saliency_map_support:
        # mask = -10e9 * (1.0 - A)
        # dense += mask
        # dense = K.softmax(dense)                            Eq. 3
        for i in range(n):
            for j in range(n):
                dense.unsafe_store(
                    i * n + j,
                    dense.unsafe_load(i * n + j)
                    - 1e10 * (1.0 - a.unsafe_load(i * n + j)),
                )
        softmax_rows(dense, n, n)
    else:
        # GAT with support for saliency calculations
        # W = (delta * A) * exp(dense - max(dense, axis=1, keepdims=True))
        #       * (1 - non_exist_edge)
        #     + non_exist_edge * (A + delta * (ones - A) + eye(N))
        #       * exp(dense - max(dense, axis=1, keepdims=True))
        # dense = W / K.sum(W, axis=1, keepdims=True)
        for i in range(n):
            var m = -1.7976931348623157e308
            for j in range(n):
                m = max(m, dense.unsafe_load(i * n + j))
            var s = 0.0
            for j in range(n):
                var av = a.unsafe_load(i * n + j)
                var eye = 1.0 if i == j else 0.0
                var w = (
                    delta
                    * av
                    * exp(dense.unsafe_load(i * n + j) - m)
                    * (1.0 - non_exist_edge)
                    + non_exist_edge
                    * (av + delta * (1.0 - av) + eye)
                    * exp(dense.unsafe_load(i * n + j) - m)
                )
                dense.unsafe_store(i * n + j, w)
                s += w
            for j in range(n):
                dense.unsafe_store(i * n + j, dense.unsafe_load(i * n + j) / s)

    # node_features = K.dot(dropout_attn, dropout_feat)
    dot(dense, features, result, n, n, units)

    # if self.use_bias: node_features = K.bias_add(node_features, self.biases[head])
    if use_bias:
        for i in range(n):
            for j in range(units):
                result.unsafe_store(
                    i * units + j, result.unsafe_load(i * units + j) + bias.unsafe_load(j)
                )


def GraphAttention_call(
    x: FPtr,
    a: FPtr,
    kernel: FPtr,
    attn_kernel: FPtr,
    bias: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    out_indices: IPtr,
    n: Int,
    m: Int,
    f: Int,
    units: Int,
    attn_heads: Int,
    use_bias: Int,
    attn_heads_reduction: Int,
    act: Int,
    alpha: Float64,
    final_layer: Int,
    saliency_map_support: Int,
    delta: Float64,
    non_exist_edge: Float64,
):
    """Upstream `GraphAttention.call` with the batch dimension removed.

    `kernel` is `[heads, f, units]`, `attn_kernel` is `[heads, 2 * units]`
    (upstream's `attn_kernels[head]` is a pair of `(units, 1)` kernels stored
    consecutively), `bias` is `[heads, units]`.

    Upstream's `Dropout` layers are identities at inference, which is the only
    mode this port implements.
    """
    var out_dim = 0
    if attn_heads_reduction == HEADS_REDUCTION_CONCAT:
        out_dim = units * attn_heads
    else:
        out_dim = units

    # Scratch: work is [n*units features | n*n dense | n attn_self |
    # n attn_neighs]; work2 starts the per-head outputs.
    for head in range(attn_heads):
        _head_dense(
            x,
            a,
            kernel + head * f * units,
            attn_kernel + head * 2 * units,
            bias + head * units,
            work2 + head * n * units,
            work,
            work + n * units,
            work + n * units + n * n,
            work + n * units + n * n + n,
            n,
            f,
            units,
            use_bias,
            saliency_map_support,
            delta,
            non_exist_edge,
        )

    # Aggregate the heads' output according to the reduction method
    if attn_heads_reduction == HEADS_REDUCTION_CONCAT:
        for i in range(n):
            for h in range(attn_heads):
                for j in range(units):
                    result.unsafe_store(
                        i * out_dim + h * units + j,
                        work2.unsafe_load(h * n * units + i * units + j),
                    )
    else:
        for i in range(n):
            for j in range(units):
                var acc = 0.0
                for h in range(attn_heads):
                    acc += work2.unsafe_load(h * n * units + i * units + j)
                result.unsafe_store(i * out_dim + j, acc / Float64(attn_heads))

    # output = self.activation(output)
    activation(result, work, n, out_dim, act, alpha)
    var o = 0
    while o < n * out_dim:
        result.unsafe_store(o, work.unsafe_load(o))
        o += 1

    # if self.final_layer: output = K.gather(output, out_indices)
    # The gather is staged in `work` first: `out_indices` need not be sorted,
    # and writing straight into `result` would read a row already overwritten.
    if final_layer:
        for s in range(m):
            var r = Int(iget(out_indices, s))
            for j in range(out_dim):
                work.unsafe_store(s * out_dim + j, result.unsafe_load(r * out_dim + j))
        var g = 0
        while g < m * out_dim:
            result.unsafe_store(g, work.unsafe_load(g))
            g += 1


def _head_sparse(
    x: FPtr,
    a_rows: IPtr,
    a_cols: IPtr,
    a_indptr: IPtr,
    kernel: FPtr,
    attn_kernel: FPtr,
    bias: FPtr,
    result: FPtr,
    features: FPtr,
    attn_self: FPtr,
    attn_neighs: FPtr,
    attn_values: FPtr,
    attn_norm: FPtr,
    n: Int,
    e: Int,
    f: Int,
    units: Int,
    use_bias: Int,
):
    """One head of upstream `GraphAttentionSparse.call`.

    The input is the `tf.SparseTensor` upstream is handed: `A_indices` is
    `[e, 2]` row-major, split here into `a_rows`/`a_cols`; `a_indptr` is the
    row-pointer for the same edges in canonical row-major order, which
    `tf.sparse.softmax` requires. The `tf.SparseTensor` constructor does not
    guarantee that ordering, so the Python side sorts before the call; that is
    the one divergence in this file and it changes no arithmetic.
    """
    # features = K.dot(X, kernel)
    dot(x, kernel, features, n, f, units)

    # attn_for_self / attn_for_neighs = K.dot(features, attention_kernel[k])
    dot_vec(features, attn_kernel, attn_self, n, units)
    dot_vec(features, attn_kernel + units, attn_neighs, n, units)

    # sparse_attn_self  = tf.gather(reshape(attn_for_self, [-1]), A_indices[:, 0])
    # sparse_attn_neighs = tf.gather(reshape(attn_for_neighs, [-1]), A_indices[:, 1])
    # attn_values = sparse_attn_self + sparse_attn_neighs
    for k in range(e):
        attn_values.unsafe_store(
            k,
            attn_self.unsafe_load(Int(iget(a_rows, k)))
            + attn_neighs.unsafe_load(Int(iget(a_cols, k))),
        )

    # attn_values = LeakyReLU(alpha=0.2)(attn_values)
    for k in range(e):
        attn_values.unsafe_store(k, leaky_relu(attn_values.unsafe_load(k), 0.2))

    # sparse_attn = tf.sparse.softmax(sparse_attn)             Eq. 3
    sparse_softmax(a_indptr, attn_values, attn_norm, n)

    # node_features = tf.sparse.matmul(sparse_attn, dropout_feat)
    sparse_dense_matmul(a_indptr, a_cols, attn_norm, features, result, n, units)

    # if self.use_bias: node_features = K.bias_add(node_features, self.biases[head])
    if use_bias:
        for i in range(n):
            for j in range(units):
                result.unsafe_store(
                    i * units + j, result.unsafe_load(i * units + j) + bias.unsafe_load(j)
                )


def GraphAttentionSparse_call(
    x: FPtr,
    a_rows: IPtr,
    a_cols: IPtr,
    a_indptr: IPtr,
    kernel: FPtr,
    attn_kernel: FPtr,
    bias: FPtr,
    result: FPtr,
    work: FPtr,
    work2: FPtr,
    out_indices: IPtr,
    n: Int,
    e: Int,
    m: Int,
    f: Int,
    units: Int,
    attn_heads: Int,
    use_bias: Int,
    attn_heads_reduction: Int,
    act: Int,
    alpha: Float64,
    final_layer: Int,
):
    """Upstream `GraphAttentionSparse.call` with the batch dimension removed."""
    var out_dim = 0
    if attn_heads_reduction == HEADS_REDUCTION_CONCAT:
        out_dim = units * attn_heads
    else:
        out_dim = units

    # Scratch: work holds [n*units features | n attn_self | n attn_neighs |
    # e attn_values | e attn_norm], then the gather; per-head outputs start
    # at work2.
    for head in range(attn_heads):
        _head_sparse(
            x,
            a_rows,
            a_cols,
            a_indptr,
            kernel + head * f * units,
            attn_kernel + head * 2 * units,
            bias + head * units,
            work2 + head * n * units,
            work,
            work + n * units,
            work + n * units + n,
            work + n * units + 2 * n,
            work + n * units + 2 * n + e,
            n,
            e,
            f,
            units,
            use_bias,
        )

    if attn_heads_reduction == HEADS_REDUCTION_CONCAT:
        for i in range(n):
            for h in range(attn_heads):
                for j in range(units):
                    result.unsafe_store(
                        i * out_dim + h * units + j,
                        work2.unsafe_load(h * n * units + i * units + j),
                    )
    else:
        for i in range(n):
            for j in range(units):
                var acc = 0.0
                for h in range(attn_heads):
                    acc += work2.unsafe_load(h * n * units + i * units + j)
                result.unsafe_store(i * out_dim + j, acc / Float64(attn_heads))

    activation(result, work, n, out_dim, act, alpha)
    var o = 0
    while o < n * out_dim:
        result.unsafe_store(o, work.unsafe_load(o))
        o += 1

    # The gather is staged in `work` first: `out_indices` need not be sorted,
    # and writing straight into `result` would read a row already overwritten.
    if final_layer:
        for s in range(m):
            var r = Int(iget(out_indices, s))
            for j in range(out_dim):
                work.unsafe_store(s * out_dim + j, result.unsafe_load(r * out_dim + j))
        var g = 0
        while g < m * out_dim:
            result.unsafe_store(g, work.unsafe_load(g))
            g += 1
