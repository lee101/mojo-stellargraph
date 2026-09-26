"""Port of `stellargraph/layer/preprocessing_layer.py` (v0.8.1):
`GraphPreProcessingLayer.call`.

GCN is documented as taking an already-normalized Laplacian, but
`GCN.__init__` also offers `method="none"`, in which case the graph
normalization is inserted as a layer so that it stays differentiable inside a
Keras model. This is that layer, in the statement order upstream uses.
"""

from std.math import sqrt

from mojostellargraph.core_utils import add_self_loops, symmetrize
from mojostellargraph.types import FPtr


def GraphPreProcessingLayer_call(adj: FPtr, dst: FPtr, work: FPtr, rowsum: FPtr, n: Int):
    """Upstream `GraphPreProcessingLayer.call`.

    ```
    adj_T = tf.transpose(adj)
    adj = adj + tf.multiply(adj_T, tf.where(adj_T > adj, ones, zeros)) \
              - tf.multiply(adj, tf.where(adj_T > adj, ones, zeros))
    adj = adj + tf.linalg.diag(tf.ones(adj.shape[0]) - tf.diag_part(adj))
    rowsum = tf.reduce_sum(adj, 1)
    d_mat_inv_sqrt = tf.diag(tf.rsqrt(rowsum))
    adj_normalized = tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    ```
    """
    symmetrize(adj, work, n)
    add_self_loops(work, dst, n)

    # rowsum = tf.reduce_sum(adj, 1); d_mat_inv_sqrt = tf.diag(tf.rsqrt(rowsum))
    # adj_normalized = tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    for i in range(n):
        var s = 0.0
        for j in range(n):
            s += dst.unsafe_load(i * n + j)
        rowsum.unsafe_store(i, 1.0 / sqrt(s))
    for i in range(n):
        for j in range(n):
            work.unsafe_store(
                i * n + j,
                rowsum.unsafe_load(i) * dst.unsafe_load(i * n + j) * rowsum.unsafe_load(j),
            )
    var k = 0
    while k < n * n:
        dst.unsafe_store(k, work.unsafe_load(k))
        k += 1
