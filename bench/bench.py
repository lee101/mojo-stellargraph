"""mojo-stellargraph against the NumPy reference, on the same arrays.

    pixi run bench

The reference is `tests/upstream_reference.py`, the same NumPy transliteration
of upstream `stellargraph` the parity tests use, because the real package needs
Python < 3.9 and TensorFlow 2.1 and cannot be installed here. That makes this
a Mojo-versus-NumPy comparison, not Mojo-versus-TensorFlow, and the numbers
say so: a single-threaded scalar-plus-SIMD kernel is not going to beat NumPy's
BLAS on a dense matmul, and where it does not, the table says so.

The pixi task holds a machine-wide flock so a concurrent factory job cannot
distort the timings. Always go through `pixi run bench`.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mojostellargraph as sg  # noqa: E402
import upstream_reference as ref  # noqa: E402


def timeit(fn, repeat: int = 3, budget: float = 1.0) -> float:
    """Best of `repeat` runs, or a single run when one is already expensive.

    The first measurement is the estimate: a case that takes longer than
    `budget` is timed once, so a slow reference cannot dominate the run."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > budget:
        return first
    best = first
    for _ in range(repeat - 1):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def graph(n: int, d: int, seed: int = 0, density: float = 0.02, loops: bool = True):
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < density).astype(np.float64)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    if loops:
        a = a + np.eye(n)
    x = np.ascontiguousarray(rng.normal(size=(n, d)))
    return a, x


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


# ------------------------------------------------ stellargraph/core/utils.py
@case("normalize_adj symmetric (1500 x 1500)")
def _():
    a, _x = graph(1500, 1)
    return (
        lambda: sg.normalize_adj(a, True),
        lambda: ref._normalize_adj(a, True),
    )


@case("normalize_adj left-only (1500 x 1500)")
def _():
    a, _x = graph(1500, 1)
    return (lambda: sg.normalize_adj(a, False), lambda: ref._normalize_adj(a, False))


@case("normalized_laplacian (1500 x 1500)")
def _():
    a, _x = graph(1500, 1)
    return (
        lambda: sg.normalized_laplacian(a, True),
        lambda: ref._normalized_laplacian(a, True),
    )


@case("GCN_Aadj_feats_op gcn (1200 x 1200)")
def _():
    a, _x = graph(1200, 1)
    return (
        lambda: sg.GCN_Aadj_feats_op(None, a, 1, "gcn"),
        lambda: (None, ref.gcn_aadj_feats_op(a, 1, "gcn")),
    )


@case("GCN_Aadj_feats_op sgc k=3 (700 x 700)")
def _():
    a, _x = graph(700, 1)
    return (
        lambda: sg.GCN_Aadj_feats_op(None, a, 3, "sgc"),
        lambda: (None, ref.gcn_aadj_feats_op(a, 3, "sgc")),
    )


@case("chebyshev_polynomial k=4 (400 x 400)")
def _():
    a, _x = graph(400, 1)
    return (
        lambda: sg.chebyshev_polynomial(a, 4),
        lambda: ref._chebyshev_polynomial(a, 4),
    )


@case("invert (700 x 700)")
def _():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(700, 700))
    m = m @ m.T + 700 * np.eye(700)
    return (lambda: sg.invert(m), lambda: np.linalg.inv(m))


@case("GraphPreProcessingLayer (1200 x 1200)")
def _():
    a, _x = graph(1200, 1)
    layer = sg.GraphPreProcessingLayer(1200)
    return (lambda: layer(a), lambda: ref.graph_pre_processing_layer(a))


# ------------------------------------- stellargraph/layer/gcn.py
@case("GraphConvolution 64 features (1500 x 1500, 64 -> 32)")
def _():
    a, x = graph(1500, 64)
    rng = np.random.default_rng(2)
    layer = sg.GraphConvolution(32, activation="relu").build(64)
    layer.kernel[:] = rng.normal(size=(64, 32))
    layer.bias[:] = rng.normal(size=32)
    return (
        lambda: layer(x, a),
        lambda: ref.graph_convolution_call(x, a, layer.kernel, layer.bias, "relu"),
    )


@case("GCN stack 32-32-8 (1000 x 1000, 64 -> 32 -> 32 -> 8)")
def _():
    a, x = graph(1000, 64)
    gen = type("G", (), {"method": "gcn", "node_list": range(1000),
                         "features": np.zeros((1000, 64))})()
    model = sg.GCN([32, 32, 8], gen, activations=["relu", "relu", "softmax"])
    model.build(64, seed=3)

    def ours():
        return model(x, a)

    def theirs():
        h = x
        for layer in model._layers:
            h = ref.graph_convolution_call(
                h, a, layer.kernel, layer.bias, layer.activation
            )
        return h

    return ours, theirs


# ------------------------------ stellargraph/layer/graph_attention.py
@case("GraphAttention 8 heads x 32 (1000 nodes, 64 -> 32)")
def _():
    a, x = graph(1000, 64)
    rng = np.random.default_rng(4)
    heads, units = 8, 32
    layer = sg.GraphAttention(units, attn_heads=heads, activation="softmax").build(64)
    w = rng.normal(size=(heads, 64, units))
    ak = rng.normal(size=(heads, 2 * units))
    b = rng.normal(size=(heads, units))
    layer.kernels[:] = w
    layer.attn_kernels[:] = ak
    layer.biases[:] = b
    return (
        lambda: layer(x, a),
        lambda: ref.graph_attention_call(
            x, a, w, ak.reshape(heads, 2, units), b, heads, "concat", "softmax"
        ),
    )


@case("GraphAttentionSparse 8 heads x 32 (1000 nodes, 64 -> 32)")
def _():
    a, x = graph(1000, 64)
    rng = np.random.default_rng(4)
    heads, units = 8, 32
    sp = sg.SparseTensor.from_dense(a)
    layer = sg.GraphAttentionSparse(
        units, attn_heads=heads, activation="softmax"
    ).build(64)
    w = rng.normal(size=(heads, 64, units))
    ak = rng.normal(size=(heads, 2 * units))
    b = rng.normal(size=(heads, units))
    layer.kernels[:] = w
    layer.attn_kernels[:] = ak
    layer.biases[:] = b
    idx = sp.indices
    return (
        lambda: layer(x, sp),
        lambda: ref.graph_attention_sparse_call(
            x, idx, w, ak.reshape(heads, 2, units), b, heads, "concat", "softmax"
        ),
    )


# ------------------------------- stellargraph/layer/graphsage.py
def sampled(n, heads, samples, d, seed=5):
    """`samples` neighbour counts, one per hop, plus the head-node features."""
    rng = np.random.default_rng(seed)
    return (
        [np.ascontiguousarray(rng.normal(size=(n, heads, s, d))) for s in samples],
        np.ascontiguousarray(rng.normal(size=(n, heads, d))),
    )


@case("MeanAggregator 2 hops (20000 x 10 heads x 25 x 64 -> 128)")
def _():
    (x1, x2), head = sampled(20000, 10, [25, 25], 64)
    agg = sg.MeanAggregator(128, bias=True, act="relu")
    agg._build_group_weights(64, 0, 128)
    agg._build_group_weights(64, 1, 128)

    def ours():
        agg.group_aggregate(head, 0)
        agg.group_aggregate(x1, 1)
        agg.group_aggregate(x2, 1)

    def theirs():
        ref.mean_aggregator_group_aggregate(head, agg.w_group[0], 0)
        ref.mean_aggregator_group_aggregate(x1, agg.w_group[1], 1)
        ref.mean_aggregator_group_aggregate(x2, agg.w_group[1], 1)

    return ours, theirs


@case("MaxPoolingAggregator 2 hops (20000 x 10 heads x 25 x 64 -> 128)")
def _():
    (x1, x2), head = sampled(20000, 10, [25, 25], 64)
    agg = sg.MaxPoolingAggregator(128, bias=True, act="relu")
    for g in (0, 1):
        agg._build_group_weights(64, g, 128)

    def ours():
        agg.group_aggregate(x1, 1)
        agg.group_aggregate(x2, 1)

    def theirs():
        ref.max_pooling_aggregator_group_aggregate(
            x1, agg.w_group[1], agg.w_pool[1], agg.b_pool[1], 1
        )
        ref.max_pooling_aggregator_group_aggregate(
            x2, agg.w_group[1], agg.w_pool[1], agg.b_pool[1], 1
        )

    return ours, theirs


@case("AttentionalAggregator (20000 x 10 heads x 25 x 64 -> 128)")
def _():
    (x1,), head = sampled(20000, 10, [25], 64)
    agg = sg.AttentionalAggregator(128, bias=True, act="relu")
    agg._build_group_weights(64, 1, 128)
    return (
        lambda: agg.group_aggregate(head, x1, 1),
        lambda: ref.attentional_aggregator_group_aggregate(
            head, x1, agg.w_group[1], agg.w_attn_s, agg.w_attn_g
        ),
    )


def _graphsage_model(model, xin):
    """The `GraphSAGE.__call__` layer loop, in NumPy, over the weights the
    model just built. This is the same walk upstream's `apply_layer` makes."""
    h_layer = [np.ascontiguousarray(x) for x in xin]
    for layer in range(model.max_hops):
        layer_out = []
        for i in range(model.max_hops - layer):
            head_shape = h_layer[i].shape[1]
            aggs = model._aggs[layer][i]
            x_self = h_layer[i]
            neigh = h_layer[i + 1].reshape(
                h_layer[i].shape[0], head_shape, -1, h_layer[i + 1].shape[-1]
            )
            parts = [
                ref.mean_aggregator_group_aggregate(x_self, aggs[0].w_group[0], 0)
            ]
            parts += [
                ref.mean_aggregator_group_aggregate(neigh, a.w_group[g], g)
                for g, a in enumerate(aggs[1:], 1)
            ]
            layer_out.append(
                ref.graphsage_aggregator_call(
                    np.concatenate(parts, axis=2), aggs[0].bias, aggs[0].act
                )
            )
        h_layer = layer_out
    out = h_layer[0]
    return ref.graphsage_normalization(out, model.normalize)


@case("GraphSAGE 3 layers, mean (20000 x 10 heads, 64 -> 128 -> 64 -> 32)")
def _():
    (x1, x2, x3), head = sampled(20000, 10, [25, 25, 25], 64)
    model = sg.GraphSAGE([128, 64, 32], aggregator=sg.MeanAggregator, normalize="l2")
    xin = [head, x1, x2, x3]
    model(xin)  # builds the weights from the input widths
    return (lambda: model(xin), lambda: _graphsage_model(model, xin))


# ------------------------------------- stellargraph/layer/hinsage.py
@case("MeanHinAggregator 3 relations (20000 x 10 heads x 25 x 64 -> 128)")
def _():
    rng = np.random.default_rng(6)
    b, hn, s, d, nr = 20000, 10, 25, 64, 3
    head = rng.normal(size=(b, hn, d))
    rel = rng.normal(size=(nr, b, hn, s, d))
    agg = sg.MeanHinAggregator(128, nr, "relu", True).build(d, d)
    return (
        lambda: agg(head, rel),
        lambda: ref.mean_hin_aggregator_call(
            head, rel, agg.w_self, agg.w_neigh, agg.bias, "relu"
        ),
    )


# ------------------------------- stellargraph/layer/{ppnp,appnp}.py
@case("PPNPPropagationLayer (1500 x 1500, 64 -> 128)")
def _():
    a, x = graph(1500, 64)
    layer = sg.PPNPPropagationLayer(64)
    return (
        lambda: layer(x, a),
        lambda: ref.ppnp_propagation_layer_call(x, a),
    )


@case("APPNP_propagate k=10 (1500 x 1500, 64 -> 128)")
def _():
    a, x = graph(1500, 64)
    model = sg.APPNP([64], None, teleport_probability=0.15)
    return (
        lambda: model.propagate(x, a, 10),
        lambda: ref.appnp_propagate(x, a, 10, 0.15),
    )


# --------------------------- stellargraph/layer/link_inference.py
@case("link_inference hadamard 200k edges (64 -> 8)")
def _():
    rng = np.random.default_rng(7)
    u = rng.normal(size=(200000, 64))
    v = rng.normal(size=(200000, 64))
    k = rng.normal(size=(64, 8))
    b = rng.normal(size=8)
    f = sg.link_inference(
        output_dim=8, output_act="sigmoid", edge_embedding_method="hadamard",
        kernel=k, bias=b,
    )
    return (
        lambda: f(u, v),
        lambda: ref.link_inference(u, v, k, b, 8, "sigmoid", "hadamard"),
    )


@case("link_inference concat 200k edges (64 -> 8)")
def _():
    rng = np.random.default_rng(7)
    u = rng.normal(size=(200000, 64))
    v = rng.normal(size=(200000, 64))
    k = rng.normal(size=(128, 8))
    f = sg.link_inference(
        output_dim=8, output_act="sigmoid", edge_embedding_method="concat", kernel=k
    )
    return (
        lambda: f(u, v),
        lambda: ref.link_inference(u, v, k, np.zeros(8), 8, "sigmoid", "concat"),
    )


# ------------------------------------- stellargraph/data/explorer.py
@case("UniformRandomWalk 200k walks of length 10 (100k nodes)")
def _():
    rng = np.random.default_rng(8)
    n = 100000
    deg = rng.integers(1, 5, size=n)
    rows = np.repeat(np.arange(n), deg)
    cols = rng.integers(0, n, size=rows.size)
    edges = np.stack([rows, cols], axis=1)
    walker = sg.UniformRandomWalk(edges, n)
    indptr, colind = sg.csr_from_edges(edges, n)
    roots = np.arange(0, n, 2)
    return (
        lambda: walker.run(roots, 4, 10, 42),
        lambda: ref.uniform_random_walk(indptr, colind, roots, 4, 10, 42),
    )


@case("BiasedRandomWalk 50k walks of length 10 (50k nodes, p=0.5 q=2)")
def _():
    rng = np.random.default_rng(9)
    n = 50000
    deg = rng.integers(1, 5, size=n)
    rows = np.repeat(np.arange(n), deg)
    cols = rng.integers(0, n, size=rows.size)
    edges = np.stack([rows, cols], axis=1)
    walker = sg.BiasedRandomWalk(edges, n)
    indptr, colind = sg.csr_from_edges(edges, n)
    roots = np.arange(0, n, 2)
    return (
        lambda: walker.run(roots, 2, 10, 0.5, 2.0, 42),
        lambda: ref.biased_random_walk(indptr, colind, roots, 2, 0.5, 2.0, 10, 42),
    )


def machine() -> str:
    model = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return "{} / {} cores / {}".format(
        model or platform.processor(), os.cpu_count(), platform.platform()
    )


def main() -> int:
    print("mojo-stellargraph benchmark")
    print("baseline: tests/upstream_reference.py (NumPy transliteration of")
    print("          stellargraph v0.8.1; the real package needs Python < 3.9)")
    print("machine:  {}".format(machine()))
    print()
    rows = []
    for name, build in CASES:
        ours, theirs = build()
        t_ours = timeit(ours)
        t_theirs = timeit(theirs)
        rows.append((name, t_ours, t_theirs))

    print()
    print("| case | mojo-stellargraph | numpy reference | |")
    print("| --- | ---: | ---: | --- |")
    faster = 0
    for name, a, b in rows:
        ratio = b / a if a > 0 else float("inf")
        if ratio >= 1.0:
            faster += 1
        if ratio >= 1.0:
            verdict = "**{:.1f}x faster**".format(ratio)
        else:
            verdict = "{:.1f}x slower".format(1.0 / ratio)
        print(
            "| `{}` | {:.2f} ms | {:.2f} ms | {} |".format(
                name, a * 1e3, b * 1e3, verdict
            )
        )
    print()
    print(
        "{} of {} cases faster; {} slower.".format(
            faster, len(rows), len(rows) - faster
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
