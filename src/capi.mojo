"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address: `@export` refuses parametric
functions, and a pointer with an inferred origin is parametric, so the address
is rebuilt inside the wrapper. The callers own all memory including scratch,
so nothing here allocates and nothing here can leak.

The body of each wrapper is the upstream call and nothing else. The branch
structure, the argument order and the arithmetic all live in
`mojostellargraph/`, where they can be read next to the Keras originals.
"""

from mojostellargraph.appnp import (
    APPNP_propagate,
    APPNPPropagationLayer_call,
)
from mojostellargraph.core_utils import (
    GCN_Aadj_feats_op,
    METHOD_CHEBYSHEV,
    METHOD_GCN,
    METHOD_NONE,
    METHOD_SGC,
    PPNP_Aadj_feats_op,
    chebyshev_polynomial,
    invert,
    normalized_laplacian,
    normalize_adj,
    power_iteration,
    rescale_laplacian,
)
from mojostellargraph.explorer import (
    biased_random_walk,
    naive_weighted_choices,
    uniform_random_walk,
)
from mojostellargraph.gcn import GraphConvolution_call
from mojostellargraph.graph_attention import (
    GraphAttention_call,
    GraphAttentionSparse_call,
)
from mojostellargraph.graphsage import (
    AttentionalAggregator_call,
    AttentionalAggregator_group_aggregate,
    GraphSAGE_normalization,
    GraphSAGEAggregator_call,
    MaxPoolingAggregator_group_aggregate,
    MeanAggregator_group_aggregate,
    MeanPoolingAggregator_group_aggregate,
)
from mojostellargraph.hinsage import MeanHinAggregator_call
from mojostellargraph.link_inference import (
    LeakyClippedLinear_call,
    link_inference_edge_function,
)
from mojostellargraph.node_mappers import self_loops
from mojostellargraph.ppnp import PPNPPropagationLayer_call
from mojostellargraph.preprocessing_layer import GraphPreProcessingLayer_call
from mojostellargraph.sparse import coo_to_csr, sparse_dense_matmul
from mojostellargraph.types import FPtr, IPtr


def f(a: Int) -> FPtr:
    return FPtr(unsafe_from_address=a)


def i(a: Int) -> IPtr:
    return IPtr(unsafe_from_address=a)


# --------------------------------------------- stellargraph/core/utils.py
@export("msg_normalize_adj")
def msg_normalize_adj(adj: Int, dst: Int, d: Int, n: Int, symmetric: Int) abi("C"):
    normalize_adj(f(adj), f(dst), f(d), n, symmetric)


@export("msg_normalized_laplacian")
def msg_normalized_laplacian(adj: Int, lap: Int, d: Int, n: Int, symmetric: Int) abi("C"):
    normalized_laplacian(f(adj), f(lap), f(d), n, symmetric)


@export("msg_rescale_laplacian")
def msg_rescale_laplacian(lap: Int, dst: Int, n: Int, eigval: Float64) abi("C"):
    rescale_laplacian(f(lap), f(dst), n, eigval)


@export("msg_power_iteration")
def msg_power_iteration(
    a: Int, work: Int, vec: Int, n: Int, max_iter: Int, tol: Float64
) abi("C") -> Float64:
    return power_iteration(f(a), f(work), f(vec), n, max_iter, tol)


@export("msg_chebyshev_polynomial")
def msg_chebyshev_polynomial(x: Int, result: Int, work: Int, n: Int, k: Int) abi("C"):
    chebyshev_polynomial(f(x), f(result), f(work), n, k)


@export("msg_PPNP_Aadj_feats_op")
def msg_PPNP_Aadj_feats_op(
    adj: Int, result: Int, tmp: Int, tmp2: Int, scratch: Int, n: Int, tp: Float64
) abi("C") -> Int:
    return PPNP_Aadj_feats_op(f(adj), f(result), f(tmp), f(tmp2), f(scratch), n, tp)


@export("msg_GCN_Aadj_feats_op")
def msg_GCN_Aadj_feats_op(
    adj: Int,
    result: Int,
    work: Int,
    work2: Int,
    scratch: Int,
    cheb: Int,
    n: Int,
    k: Int,
    method: Int,
) abi("C") -> Int:
    return GCN_Aadj_feats_op(
        f(adj), f(result), f(work), f(work2), f(scratch), f(cheb), n, k, method
    )


@export("msg_invert")
def msg_invert(a: Int, result: Int, work: Int, n: Int) abi("C") -> Int:
    return invert(f(a), f(result), f(work), n)


# ------------------------------------- stellargraph/mapper/node_mappers.py
@export("msg_self_loops")
def msg_self_loops(adj: Int, dst: Int, n: Int) abi("C"):
    self_loops(f(adj), f(dst), n)


# -------------------------------------- stellargraph/layer/gcn.py, and
# ------------------- stellargraph/layer/preprocessing_layer.py
@export("msg_GraphConvolution_call")
def msg_GraphConvolution_call(
    features: Int,
    adj: Int,
    kernel: Int,
    bias: Int,
    result: Int,
    work: Int,
    out_indices: Int,
    n: Int,
    m: Int,
    fdim: Int,
    units: Int,
    use_bias: Int,
    act: Int,
    alpha: Float64,
    final_layer: Int,
) abi("C"):
    GraphConvolution_call(
        f(features),
        f(adj),
        f(kernel),
        f(bias),
        f(result),
        f(work),
        i(out_indices),
        n,
        m,
        fdim,
        units,
        use_bias,
        act,
        alpha,
        final_layer,
    )


@export("msg_GraphPreProcessingLayer_call")
def msg_GraphPreProcessingLayer_call(
    adj: Int, dst: Int, work: Int, rowsum: Int, n: Int
) abi("C"):
    GraphPreProcessingLayer_call(f(adj), f(dst), f(work), f(rowsum), n)


# --------------------------- stellargraph/layer/graph_attention.py
@export("msg_GraphAttention_call")
def msg_GraphAttention_call(
    x: Int,
    a: Int,
    kernel: Int,
    attn_kernel: Int,
    bias: Int,
    result: Int,
    work: Int,
    work2: Int,
    out_indices: Int,
    n: Int,
    m: Int,
    fdim: Int,
    units: Int,
    attn_heads: Int,
    use_bias: Int,
    heads_reduction: Int,
    act: Int,
    alpha: Float64,
    final_layer: Int,
    saliency_map_support: Int,
    delta: Float64,
    non_exist_edge: Float64,
) abi("C"):
    GraphAttention_call(
        f(x),
        f(a),
        f(kernel),
        f(attn_kernel),
        f(bias),
        f(result),
        f(work),
        f(work2),
        i(out_indices),
        n,
        m,
        fdim,
        units,
        attn_heads,
        use_bias,
        heads_reduction,
        act,
        alpha,
        final_layer,
        saliency_map_support,
        delta,
        non_exist_edge,
    )


@export("msg_GraphAttentionSparse_call")
def msg_GraphAttentionSparse_call(
    x: Int,
    a_rows: Int,
    a_cols: Int,
    a_indptr: Int,
    kernel: Int,
    attn_kernel: Int,
    bias: Int,
    result: Int,
    work: Int,
    work2: Int,
    out_indices: Int,
    n: Int,
    e: Int,
    m: Int,
    fdim: Int,
    units: Int,
    attn_heads: Int,
    use_bias: Int,
    heads_reduction: Int,
    act: Int,
    alpha: Float64,
    final_layer: Int,
) abi("C"):
    GraphAttentionSparse_call(
        f(x),
        i(a_rows),
        i(a_cols),
        i(a_indptr),
        f(kernel),
        f(attn_kernel),
        f(bias),
        f(result),
        f(work),
        f(work2),
        i(out_indices),
        n,
        e,
        m,
        fdim,
        units,
        attn_heads,
        use_bias,
        heads_reduction,
        act,
        alpha,
        final_layer,
    )


# ------------------------------------ stellargraph/layer/graphsage.py
@export("msg_MeanAggregator_group_aggregate")
def msg_MeanAggregator_group_aggregate(
    x: Int,
    w: Int,
    result: Int,
    work: Int,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    out_dim: Int,
    group_idx: Int,
) abi("C"):
    MeanAggregator_group_aggregate(
        f(x), f(w), f(result), f(work), b, h, s, d, out_dim, group_idx
    )


@export("msg_MaxPoolingAggregator_group_aggregate")
def msg_MaxPoolingAggregator_group_aggregate(
    x: Int,
    w_group: Int,
    w_pool: Int,
    b_pool: Int,
    result: Int,
    work: Int,
    work2: Int,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    hidden_dim: Int,
    out_dim: Int,
    group_idx: Int,
) abi("C"):
    MaxPoolingAggregator_group_aggregate(
        f(x),
        f(w_group),
        f(w_pool),
        f(b_pool),
        f(result),
        f(work),
        f(work2),
        b,
        h,
        s,
        d,
        hidden_dim,
        out_dim,
        group_idx,
    )


@export("msg_MeanPoolingAggregator_group_aggregate")
def msg_MeanPoolingAggregator_group_aggregate(
    x: Int,
    w_group: Int,
    w_pool: Int,
    b_pool: Int,
    result: Int,
    work: Int,
    work2: Int,
    b: Int,
    h: Int,
    s: Int,
    d: Int,
    hidden_dim: Int,
    out_dim: Int,
    group_idx: Int,
) abi("C"):
    MeanPoolingAggregator_group_aggregate(
        f(x),
        f(w_group),
        f(w_pool),
        f(b_pool),
        f(result),
        f(work),
        f(work2),
        b,
        h,
        s,
        d,
        hidden_dim,
        out_dim,
        group_idx,
    )


@export("msg_AttentionalAggregator_group_aggregate")
def msg_AttentionalAggregator_group_aggregate(
    x_self: Int,
    x_g: Int,
    w_g: Int,
    w_attn_s: Int,
    w_attn_g: Int,
    result: Int,
    work: Int,
    work2: Int,
    b: Int,
    h: Int,
    s: Int,
    d_self: Int,
    d: Int,
    out_dim: Int,
) abi("C"):
    AttentionalAggregator_group_aggregate(
        f(x_self),
        f(x_g),
        f(w_g),
        f(w_attn_s),
        f(w_attn_g),
        f(result),
        f(work),
        f(work2),
        b,
        h,
        s,
        d_self,
        d,
        out_dim,
    )


@export("msg_AttentionalAggregator_call")
def msg_AttentionalAggregator_call(
    sources: Int,
    bias: Int,
    result: Int,
    work: Int,
    b: Int,
    h: Int,
    n_groups: Int,
    out_dim: Int,
    has_bias: Int,
    act: Int,
    alpha: Float64,
) abi("C"):
    AttentionalAggregator_call(
        f(sources), f(bias), f(result), f(work), b, h, n_groups, out_dim,
        has_bias, act, alpha,
    )


@export("msg_GraphSAGEAggregator_call")
def msg_GraphSAGEAggregator_call(
    sources: Int,
    bias: Int,
    result: Int,
    work: Int,
    b: Int,
    h: Int,
    n_groups: Int,
    out_dim: Int,
    has_bias: Int,
    act: Int,
    alpha: Float64,
) abi("C"):
    GraphSAGEAggregator_call(
        f(sources), f(bias), f(result), f(work), b, h, n_groups, out_dim, has_bias, act, alpha
    )


@export("msg_GraphSAGE_normalization")
def msg_GraphSAGE_normalization(
    x: Int, dst: Int, n: Int, d: Int, normalize: Int
) abi("C"):
    GraphSAGE_normalization(f(x), f(dst), n, d, normalize)


# --------------------------------------- stellargraph/layer/hinsage.py
@export("msg_MeanHinAggregator_call")
def msg_MeanHinAggregator_call(
    x_self: Int,
    x_neigh: Int,
    w_self: Int,
    w_neigh: Int,
    bias: Int,
    result: Int,
    work: Int,
    scratch: Int,
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
) abi("C"):
    MeanHinAggregator_call(
        f(x_self),
        f(x_neigh),
        f(w_self),
        f(w_neigh),
        f(bias),
        f(result),
        f(work),
        f(scratch),
        b,
        h,
        nr,
        s,
        d_self,
        d,
        half_output_dim,
        has_bias,
        act,
        alpha,
    )


# --------------------------------- stellargraph/layer/ppnp.py, appnp.py
@export("msg_PPNPPropagationLayer_call")
def msg_PPNPPropagationLayer_call(
    features: Int,
    a: Int,
    result: Int,
    gathered: Int,
    out_indices: Int,
    n: Int,
    m: Int,
    fdim: Int,
    final_layer: Int,
) abi("C"):
    PPNPPropagationLayer_call(
        f(features), f(a), f(result), f(gathered), i(out_indices), n, m, fdim,
        final_layer,
    )


@export("msg_APPNPPropagationLayer_call")
def msg_APPNPPropagationLayer_call(
    propagated: Int,
    features: Int,
    a: Int,
    result: Int,
    gathered: Int,
    out_indices: Int,
    n: Int,
    m: Int,
    fdim: Int,
    tp: Float64,
    final_layer: Int,
) abi("C"):
    APPNPPropagationLayer_call(
        f(propagated), f(features), f(a), f(result), f(gathered), i(out_indices),
        n, m, fdim, tp, final_layer,
    )


@export("msg_APPNP_propagate")
def msg_APPNP_propagate(
    x: Int, a: Int, dst: Int, work: Int, n: Int, fdim: Int, tp: Float64, k: Int
) abi("C"):
    APPNP_propagate(f(x), f(a), f(dst), f(work), n, fdim, tp, k)


# ----------------------------- stellargraph/layer/link_inference.py
@export("msg_LeakyClippedLinear_call")
def msg_LeakyClippedLinear_call(
    x: Int, dst: Int, n: Int, low: Float64, high: Float64, alpha: Float64
) abi("C"):
    LeakyClippedLinear_call(f(x), f(dst), n, low, high, alpha)


@export("msg_link_inference_edge_function")
def msg_link_inference_edge_function(
    x0: Int,
    x1: Int,
    le: Int,
    kernel: Int,
    bias: Int,
    result: Int,
    n: Int,
    d: Int,
    output_dim: Int,
    method: Int,
    act: Int,
    alpha: Float64,
    has_bias: Int,
    clip: Int,
    clip_low: Float64,
    clip_high: Float64,
) abi("C") -> Int:
    return link_inference_edge_function(
        f(x0),
        f(x1),
        f(le),
        f(kernel),
        f(bias),
        f(result),
        n,
        d,
        output_dim,
        method,
        act,
        alpha,
        has_bias,
        clip,
        clip_low,
        clip_high,
    )


# ------------------------------------- stellargraph/data/explorer.py
@export("msg_naive_weighted_choices")
def msg_naive_weighted_choices(
    indptr: Int, colind: Int, weights: Int, work: Int, node: Int, state: Int
) abi("C") -> Int:
    return naive_weighted_choices(
        i(indptr), i(colind), f(weights), f(work), node, UInt64(state)
    )


@export("msg_uniform_random_walk")
def msg_uniform_random_walk(
    indptr: Int,
    colind: Int,
    roots: Int,
    walks_out: Int,
    lens_out: Int,
    work: Int,
    n: Int,
    n_walks: Int,
    length: Int,
    seed: Int,
) abi("C"):
    uniform_random_walk(
        i(indptr),
        i(colind),
        i(roots),
        i(walks_out),
        i(lens_out),
        i(work),
        n,
        n_walks,
        length,
        seed,
    )


@export("msg_biased_random_walk")
def msg_biased_random_walk(
    indptr: Int,
    colind: Int,
    roots: Int,
    walks_out: Int,
    lens_out: Int,
    work: Int,
    work2: Int,
    indices: Int,
    n: Int,
    n_walks: Int,
    length: Int,
    p: Float64,
    q: Float64,
    seed: Int,
) abi("C"):
    biased_random_walk(
        i(indptr),
        i(colind),
        i(roots),
        i(walks_out),
        i(lens_out),
        f(work),
        f(work2),
        i(indices),
        n,
        n_walks,
        length,
        p,
        q,
        seed,
    )


# ------------------------------------------------- mojostellargraph.sparse
@export("msg_coo_to_csr")
def msg_coo_to_csr(
    rows: Int,
    cols: Int,
    values: Int,
    indptr: Int,
    colind: Int,
    result: Int,
    cursor: Int,
    e: Int,
    n: Int,
) abi("C"):
    coo_to_csr(i(rows), i(cols), f(values), i(indptr), i(colind), f(result), i(cursor), e, n)


@export("msg_sparse_dense_matmul")
def msg_sparse_dense_matmul(
    indptr: Int, colind: Int, values: Int, dense: Int, dst: Int, n: Int, d: Int
) abi("C"):
    sparse_dense_matmul(i(indptr), i(colind), f(values), f(dense), f(dst), n, d)
