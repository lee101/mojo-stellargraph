"""A NumPy transliteration of the upstream `stellargraph` code.

`stellargraph` 0.8.1 needs Python < 3.9 and TensorFlow 2.1, so it cannot be
installed here. Every function below is the same statements as the upstream
file it names, with `K.dot` as `@`, `tf.sparse.*` as an explicit COO loop and
`sp.*` as dense NumPy. It is the parity oracle: the Mojo port must agree with
it to floating-point tolerance, and the tests check that it is itself right
wherever a closed form or a published vector exists.

The upstream source is at commit `v0.8.1` of
https://github.com/stellargraph/stellargraph.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------- stellargraph/core/utils.py
def _symmetrize(adj: np.ndarray) -> np.ndarray:
    """`A + A.T.multiply(A.T > A) - A.multiply(A.T > A)`."""
    at = adj.T
    return adj + at * (at > adj) - adj * (at > adj)


def _add_self_loops(adj: np.ndarray) -> np.ndarray:
    """`adj + sp.diags(np.ones(n) - adj.diagonal())`."""
    n = adj.shape[0]
    return adj + np.diag(np.ones(n) - np.diag(adj))


def _preprocess_adj(adj: np.ndarray, symmetric: bool = True) -> np.ndarray:
    return _normalize_adj(_add_self_loops(adj), symmetric)


def _normalize_adj(adj: np.ndarray, symmetric: bool = True) -> np.ndarray:
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten())
        return adj.dot(d).T.dot(d)
    d = np.diag(np.float_power(np.array(adj.sum(1)), -1).flatten())
    return d.dot(adj)


def _normalized_laplacian(adj: np.ndarray, symmetric: bool = True) -> np.ndarray:
    return np.eye(adj.shape[0]) - _normalize_adj(adj, symmetric)


def _rescale_laplacian(laplacian: np.ndarray, largest_eigval: float) -> np.ndarray:
    return (2.0 / largest_eigval) * laplacian - np.eye(laplacian.shape[0])


def _chebyshev_polynomial(x: np.ndarray, k: int) -> list[np.ndarray]:
    # upstream returns the whole list, which holds at least `[I, X]` whatever
    # k is, because the loop starts at 2
    ts = [np.eye(x.shape[0]), x]
    for _ in range(2, k + 1):
        ts.append(2 * x.copy().dot(ts[-1]) - ts[-2])
    return ts


def ppnp_aadj_feats_op(A: np.ndarray, teleport_probability: float = 0.1):
    if (teleport_probability > 1.0) or (teleport_probability < 0.0):
        raise ValueError(
            "teleport_probability should be between 0.0 and 1.0 (inclusive)"
        )
    A = _symmetrize(A)
    A = _add_self_loops(A)
    A = _normalize_adj(A, symmetric=True)
    A = A.toarray() if hasattr(A, "toarray") else A
    A = teleport_probability * np.linalg.inv(
        np.eye(A.shape[0]) - ((1 - teleport_probability) * A)
    )
    return A


def gcn_aadj_feats_op(A: np.ndarray, k: int = 1, method: str = "gcn"):
    A = _symmetrize(A)
    if method == "gcn":
        return _preprocess_adj(A)
    if method == "chebyshev":
        if isinstance(k, int) and k >= 2:
            scaled = _rescale_laplacian(
                _normalized_laplacian(A),
                np.linalg.eigvalsh(_normalized_laplacian(A))[-1],
            )
            return _chebyshev_polynomial(scaled, k)
        raise ValueError(
            "max_degree should be positive integer of value at least 2 for "
            "method='chebyshev'"
        )
    if method == "sgc":
        if isinstance(k, int) and k > 0:
            a = _preprocess_adj(A)
            out = np.eye(a.shape[0])
            for _ in range(k):
                out = out.dot(a)
            return out
        raise ValueError("k should be positive integer for method='sgcn'")
    if method in (None, "none"):
        return A
    raise ValueError(
        "Undefined method for adjacency matrix transformation. "
        "Accepted: 'gcn' (default), 'chebyshev','sgc', and 'self_loops'."
    )


# ------------------------------------------- stellargraph/layer/gcn.py, etc.
def graph_convolution_call(features, A, kernel, bias=None, activation=None,
                           out_indices=None):
    h_graph = A @ features
    output = h_graph @ kernel
    if bias is not None:
        output = output + bias
    output = _activation(output, activation)
    if out_indices is not None:
        output = output[out_indices]
    return output


def graph_pre_processing_layer(adj: np.ndarray) -> np.ndarray:
    adj_T = adj.T
    adj = adj + adj_T * (adj_T > adj) - adj * (adj_T > adj)
    adj = adj + np.diag(np.ones(adj.shape[0]) - np.diag(adj))
    rowsum = adj.sum(1)
    d = 1.0 / np.sqrt(rowsum)
    return (d[:, None] * adj) * d[None, :]


def _leaky_relu(x, alpha=0.2):
    return np.where(x >= 0.0, x, alpha * x)


def _softmax_rows(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_dim2(x):
    m = x.max(axis=2, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=2, keepdims=True)


def _activation(x, act, alpha=0.01):
    if act in (None, "linear"):
        return x
    if act == "relu":
        return np.maximum(x, 0.0)
    if act == "elu":
        return np.where(x >= 0, x, np.expm1(x))
    if act == "softmax":
        return _softmax_rows(x)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if act == "tanh":
        return np.tanh(x)
    if act == "softplus":
        return np.log1p(np.exp(x))
    if act == "leaky_relu":
        return np.where(x >= 0, x, alpha * x)
    if act == "hard_sigmoid":
        return np.clip(0.2 * x + 0.5, 0.0, 1.0)
    if act == "exponential":
        return np.exp(x)
    raise ValueError("unsupported activation {!r}".format(act))


# ----------------------------------- stellargraph/layer/graph_attention.py
def graph_attention_call(X, A, kernel, attn_kernel, bias=None, attn_heads=1,
                         attn_heads_reduction="concat", activation="relu",
                         final_layer=False, out_indices=None,
                         saliency_map_support=False, delta=1.0,
                         non_exist_edge=0.0):
    """`GraphAttention.call`, one head at a time, as upstream writes it.

    `attn_kernel` is `[heads, 2, units]` here; the Mojo side flattens the two
    `(units, 1)` kernels of one head into `attn_kernel[head]`.
    """
    n = X.shape[0]
    outputs = []
    for head in range(attn_heads):
        features = X @ kernel[head]
        attn_for_self = features @ attn_kernel[head][0]
        attn_for_neighs = features @ attn_kernel[head][1]
        dense = attn_for_self[:, None] + attn_for_neighs[None, :]
        dense = _leaky_relu(dense, 0.2)
        if not saliency_map_support:
            mask = -10e9 * (1.0 - A)
            dense = dense + mask
            dense = _softmax_rows(dense)
        else:
            W = (delta * A) * np.exp(
                dense - dense.max(axis=1, keepdims=True)
            ) * (1 - non_exist_edge) + non_exist_edge * (
                A
                + delta * (np.ones(shape=[n, n]) - A)
                + np.eye(n)
            ) * np.exp(dense - dense.max(axis=1, keepdims=True))
            dense = W / W.sum(axis=1, keepdims=True)
        node_features = dense @ features
        if bias is not None:
            node_features = node_features + bias[head]
        outputs.append(node_features)

    if attn_heads_reduction == "concat":
        output = np.concatenate(outputs, axis=1)
    else:
        output = np.mean(np.stack(outputs), axis=0)
    output = _activation(output, activation)
    if final_layer and out_indices is not None:
        output = output[out_indices]
    return output


def graph_attention_sparse_call(X, indices, kernel, attn_kernel, bias=None,
                                attn_heads=1, attn_heads_reduction="concat",
                                activation="relu", final_layer=False,
                                out_indices=None):
    """`GraphAttentionSparse.call`.

    `indices` is the `[e, 2]` `tf.SparseTensor.indices`. `tf.sparse.softmax`
    canonicalizes to row-major, which is what `_sparse_softmax` does here.
    """
    n = X.shape[0]
    outputs = []
    for head in range(attn_heads):
        features = X @ kernel[head]
        attn_for_self = features @ attn_kernel[head][0]
        attn_for_neighs = features @ attn_kernel[head][1]
        sparse_attn_self = attn_for_self.reshape(-1)[indices[:, 0]]
        sparse_attn_neighs = attn_for_neighs.reshape(-1)[indices[:, 1]]
        attn_values = sparse_attn_self + sparse_attn_neighs
        attn_values = _leaky_relu(attn_values, 0.2)
        sparse_attn = _sparse_softmax(n, indices, attn_values)
        node_features = _sparse_matmul(n, indices, sparse_attn, features)
        if bias is not None:
            node_features = node_features + bias[head]
        outputs.append(node_features)

    if attn_heads_reduction == "concat":
        output = np.concatenate(outputs, axis=1)
    else:
        output = np.mean(np.stack(outputs), axis=0)
    output = _activation(output, activation)
    if final_layer and out_indices is not None:
        output = output[out_indices]
    return output


def _sparse_softmax(n, indices, values):
    out = np.zeros_like(values)
    for r in range(n):
        m = (indices[:, 0] == r).nonzero()[0]
        if m.size == 0:
            continue
        e = np.exp(values[m] - values[m].max())
        out[m] = e / e.sum()
    return out


def _sparse_matmul(n, indices, values, dense):
    out = np.zeros((n, dense.shape[1]))
    for k in range(indices.shape[0]):
        r, c = int(indices[k, 0]), int(indices[k, 1])
        out[r] += values[k] * dense[c]
    return out


# ------------------------------------- stellargraph/layer/graphsage.py
def mean_aggregator_group_aggregate(x_group, w, group_idx=0):
    if x_group.ndim == 3:
        b, h, _ = x_group.shape
        x_group = x_group[:, :, None, :]
    if group_idx == 0:
        b, h, s, d = x_group.shape
        return (x_group.reshape(b * h, s * d) @ w).reshape(b, h, w.shape[1])
    return (x_group.mean(axis=2) @ w).reshape(x_group.shape[0], x_group.shape[1],
                                             w.shape[1])


def _pool_group_aggregate(x_group, w_group, w_pool, b_pool, group_idx, take_max):
    if x_group.ndim == 3:
        x_group = x_group[:, :, None, :]
    if group_idx == 0:
        b, h, s, d = x_group.shape
        return (x_group.reshape(b * h, s * d) @ w_group).reshape(b, h, w_group.shape[1])
    xw = np.maximum(x_group @ w_pool + b_pool, 0.0)
    pooled = xw.max(axis=2) if take_max else xw.mean(axis=2)
    return pooled @ w_group


def max_pooling_aggregator_group_aggregate(x_group, w_group, w_pool, b_pool,
                                           group_idx=0):
    return _pool_group_aggregate(x_group, w_group, w_pool, b_pool, group_idx, True)


def mean_pooling_aggregator_group_aggregate(x_group, w_group, w_pool, b_pool,
                                            group_idx=0):
    return _pool_group_aggregate(x_group, w_group, w_pool, b_pool, group_idx, False)


def attentional_aggregator_group_aggregate(x_self, x_g, w_g, w_attn_s, w_attn_g):
    """One pass of upstream's `AttentionalAggregator.call` group loop."""
    b, h, d_self = x_self.shape
    _, _, s, d = x_g.shape
    xw_self = (x_self @ w_g)[:, :, None, :]
    xw_neigh = x_g @ w_g
    xw_all = np.concatenate([xw_self, xw_neigh], axis=2)
    attn_self = xw_self @ w_attn_s
    attn_neigh = xw_all @ w_attn_g
    attn_u = _leaky_relu(attn_self + attn_neigh, 0.2)
    attn = _softmax_dim2(attn_u)
    return (attn * xw_all).sum(axis=2)


def graphsage_aggregator_call(sources, bias=None, act="relu"):
    h_out = sources
    if bias is not None:
        h_out = h_out + bias
    return _activation(h_out, act)


def graphsage_normalization(x, normalize="l2"):
    if normalize in (None, "none", "None"):
        return x
    return x / np.maximum(np.sqrt((x ** 2).sum(axis=-1, keepdims=True)), 1e-7)


# --------------------------------------- stellargraph/layer/hinsage.py
def mean_hin_aggregator_call(x_self, x_neigh, w_self, w_neigh, bias=None,
                             act="relu"):
    """`MeanHinAggregator.call`. `x_neigh` is `[nr, b, h, s, d]`."""
    per_rel = []
    for r in range(x_neigh.shape[0]):
        z = x_neigh[r]
        if z.shape[2] > 0:
            per_rel.append(z.mean(axis=2) @ w_neigh[r])
        else:
            per_rel.append(np.zeros((z.shape[0], z.shape[1], w_neigh[r].shape[1])))
    from_self = x_self @ w_self
    from_neigh = sum(per_rel) / len(per_rel)
    total = np.concatenate([from_self, from_neigh], axis=2)
    if bias is not None:
        total = total + bias
    return _activation(total, act)


def hinsage_model_call(head, neighbourhoods, aggs, normalize="l2"):
    """`HinSAGE.__call__`: one aggregator per layer in order, then
    `self._normalization`, which is `K.l2_normalize(x, axis=-1)` for
    `normalize="l2"` and the identity otherwise."""
    h = head
    for agg, rel in zip(aggs, neighbourhoods):
        h = mean_hin_aggregator_call(
            h, rel, agg.w_self, agg.w_neigh, agg.bias, agg.act
        )
    if normalize is None or normalize == "none":
        return h
    # `K.l2_normalize(x, axis=-1)`, with TensorFlow's 1e-7 clamp on the norm
    flat = h.reshape(-1, h.shape[-1])
    norms = np.maximum(np.sqrt((flat ** 2).sum(axis=1, keepdims=True)), 1e-7)
    return (flat / norms).reshape(h.shape)


# ------------------------------------------- stellargraph/layer/ppnp.py
def ppnp_propagation_layer_call(features, A, out_indices=None):
    output = A @ features
    if out_indices is not None:
        output = output[out_indices]
    return output


# ------------------------------------------ stellargraph/layer/appnp.py
def appnp_propagation_layer_call(propagated, features, A, teleport_probability=0.1,
                                 out_indices=None):
    output = (1 - teleport_probability) * (A @ propagated) + teleport_probability * features
    if out_indices is not None:
        output = output[out_indices]
    return output


def appnp_propagate(x, A, k, teleport_probability=0.1):
    z = x.copy()
    for _ in range(k):
        z = (1 - teleport_probability) * (A @ z) + teleport_probability * x
    return z


def _dense_stack(h, layers):
    """The `Dense(l, activation=a, use_bias=b)` stack both models prepend, as
    upstream's `__init__` builds it: one `Dropout` and one `Dense` per entry,
    the activation inside the `Dense` and the bias optional."""
    for kernel, bias, act in layers:
        h = h @ kernel
        if bias is not None:
            h = h + bias
        h = _activation(h, act)
    return h


def ppnp_model_call(features, A, layers, out_indices=None):
    """`PPNP.__call__`: the Dense stack, then one `PPNPPropagationLayer` with
    `final_layer=True`."""
    h = _dense_stack(features, layers)
    return ppnp_propagation_layer_call(h, A, out_indices)


def appnp_model_call(features, A, layers, teleport_probability=0.1,
                     approx_iter=10, out_indices=None):
    """`APPNP.__call__`: the Dense stack, then `approx_iter`
    `APPNPPropagationLayer`s of which the last carries
    `final_layer=(ii == approx_iter - 1)`."""
    h = _dense_stack(features, layers)
    feature_layer = h
    for ii in range(approx_iter):
        h = appnp_propagation_layer_call(
            h, feature_layer, A, teleport_probability,
            out_indices if ii == approx_iter - 1 else None,
        )
    return h


# ---------------------------------- stellargraph/layer/link_inference.py
def leaky_clipped_linear(x, low=1.0, high=5.0, alpha=0.1):
    gamma = 1.0 - alpha
    x_lo = np.maximum(low - x, 0.0)
    x_hi = np.maximum(x - high, 0.0)
    return x + gamma * x_lo - gamma * x_hi


def link_inference(x0, x1, kernel, bias=None, output_dim=1, output_act="linear",
                   edge_embedding_method="ip", clip_limits=None):
    """The `edge_function` closure inside upstream `link_inference`."""
    if edge_embedding_method in ("ip", "dot"):
        out = (x0 * x1).sum(axis=-1)
        out = _activation(out, output_act)
        out = out.reshape(-1, 1)
    elif edge_embedding_method == "l1":
        le = np.abs(x0 - x1)
        out = _activation(le @ kernel + (bias if bias is not None else 0.0), output_act)
    elif edge_embedding_method == "l2":
        le = (x0 - x1) ** 2
        out = _activation(le @ kernel + (bias if bias is not None else 0.0), output_act)
    elif edge_embedding_method in ("mul", "hadamard"):
        le = x0 * x1
        out = _activation(le @ kernel + (bias if bias is not None else 0.0), output_act)
    elif edge_embedding_method == "concat":
        le = np.concatenate([x0, x1], axis=-1)
        out = _activation(le @ kernel + (bias if bias is not None else 0.0), output_act)
    elif edge_embedding_method == "avg":
        le = (x0 + x1) / 2
        out = _activation(le @ kernel + (bias if bias is not None else 0.0), output_act)
    else:
        raise NotImplementedError(edge_embedding_method)
    if clip_limits:
        out = leaky_clipped_linear(out, clip_limits[0], clip_limits[1], 0.1)
    return out


# ------------------------------------- stellargraph/data/explorer.py
_MASK = (1 << 64) - 1
_MULT = 6364136223846793005
_INCR = 1442695040888963407


class LCG:
    """The generator the Mojo walkers use, mirrored so the tests can check the
    walk logic rather than the random numbers."""

    def __init__(self, seed):
        self.state = (int(seed) + _INCR) & _MASK

    def next(self) -> int:
        self.state = (self.state * _MULT + _INCR) & _MASK
        return self.state

    def bounded(self, bound: int) -> int:
        """Rejection sample the top 32 bits against the largest multiple of
        `bound` below 2**32 -- the same arithmetic the Mojo kernel does."""
        if bound <= 1:
            return 0
        s = self.state
        threshold = ((1 << 32) // bound) * bound
        while True:
            s = (s * _MULT + _INCR) & _MASK
            if (s >> 32) < threshold:
                return (s >> 32) % bound

    def uniform(self) -> float:
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)


def shuffle(items, rs: LCG):
    """`rs.shuffle(neighbours)`: Fisher-Yates, as NumPy does it."""
    for i in range(len(items) - 1, 0, -1):
        j = rs.bounded(i + 1)
        rs.next()
        items[i], items[j] = items[j], items[i]
    return items


def uniform_random_walk(indptr, colind, roots, n=1, length=5, seed=0):
    """`UniformRandomWalk.run`, with the LCG in place of `RandomState`."""
    rs = LCG(seed)
    walks = []
    for node in roots:
        for _ in range(n):
            walk = []
            current = int(node)
            for _ in range(length):
                walk.append(current)
                neighbours = [
                    int(colind[k])
                    for k in range(int(indptr[current]), int(indptr[current + 1]))
                ]
                if not neighbours:
                    break
                shuffle(neighbours, rs)
                current = neighbours[0]
            walks.append(walk)
    return walks


def biased_random_walk(indptr, colind, roots, n=1, p=1.0, q=1.0, length=5, seed=0):
    """`BiasedRandomWalk.run`, unweighted, with the LCG in place of
    `RandomState`."""
    rs = LCG(seed)
    ip, iq = 1.0 / p, 1.0 / q
    walks = []

    def nbrs(node):
        return [
            int(colind[k]) for k in range(int(indptr[node]), int(indptr[node + 1]))
        ]

    for node in roots:
        for _ in range(n):
            walk = [int(node)]
            current = int(node)
            previous = int(node)
            previous_neighbours = nbrs(current)
            if previous_neighbours:
                current = previous_neighbours[rs.bounded(len(previous_neighbours))]
                rs.next()
                for _ in range(length - 1):
                    walk.append(current)
                    neighbours = nbrs(current)
                    if not neighbours:
                        break
                    weights = []
                    for nn in neighbours:
                        if nn == previous:
                            weights.append(ip)
                        elif nn in previous_neighbours:
                            weights.append(1.0)
                        else:
                            weights.append(iq)
                    choice = naive_weighted_choices(weights, rs)
                    rs.next()
                    previous = current
                    previous_neighbours = neighbours
                    current = neighbours[choice]
            walks.append(walk)
    return walks


def naive_weighted_choices(weights, rs: LCG) -> int:
    """Upstream `naive_weighted_choices(rs, weights)`."""
    ends = []
    running_total = 0.0
    for w in weights:
        if w < 0:
            raise ValueError("Detected negative weight: {}".format(w))
        running_total += w
        ends.append(running_total)
    x = rs.uniform() * running_total
    for idx, end in enumerate(ends):
        if x < end:
            return idx
    return len(ends) - 1
