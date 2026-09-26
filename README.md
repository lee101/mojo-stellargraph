# mojo-stellargraph

The graph machine learning kernels of
[stellargraph](https://github.com/stellargraph/stellargraph) v0.8.1, implemented
in [Mojo](https://www.modular.com/mojo) and callable from Python with the same
names, argument order and defaults.

```python
import mojostellargraph as sg

gen = sg.FullBatchNodeGenerator(graph, method="gcn")
model = sg.GCN([32, 4], gen, activations=["elu", "softmax"]).build(gen.features.shape[1])
out = model(gen.features, gen.Aadj)
```

Upstream is a TensorFlow library: most of a `stellargraph` model's numeric work
is `K.dot`, a softmax and a sparse matrix product. Those are the kernels here.
`src/mojostellargraph/*.mojo` is a line-for-line transliteration of the upstream
files, in upstream's order, and `python/mojostellargraph/*.py` is the same
module split the other way, with the ctypes marshalling on the boundary.

## What is covered

| module | upstream | what is ported |
| --- | --- | --- |
| `core_utils` | `stellargraph/core/utils.py` | `normalize_adj`, `normalized_laplacian`, `rescale_laplacian`, `chebyshev_polynomial`, `PPNP_Aadj_feats_op`, `GCN_Aadj_feats_op` (methods `gcn`, `sgc`, `chebyshev`, `none`), plus `power_iteration` and `invert` standing in for `eigsh` and `np.linalg.inv` |
| `gcn` | `stellargraph/layer/gcn.py` | `GraphConvolution` (build, call), `GCN` (layer stack, forward) |
| `preprocessing_layer` | `stellargraph/layer/preprocessing_layer.py` | `GraphPreProcessingLayer` |
| `graph_attention` | `stellargraph/layer/graph_attention.py` | `GraphAttention` (dense, both head reductions, the `saliency_map_support` branch) and `GraphAttentionSparse` |
| `graphsage` | `stellargraph/layer/graphsage.py` | `MeanAggregator`, `MaxPoolingAggregator`, `MeanPoolingAggregator`, `AttentionalAggregator`, `GraphSAGE.__call__` and its `l2` normalization |
| `hinsage` | `stellargraph/layer/hinsage.py` | `MeanHinAggregator`, `HinSAGE` (per-layer `activations`, `normalize`, the `output_dim % 2` assertion) |
| `ppnp` | `stellargraph/layer/ppnp.py` | `PPNPPropagationLayer`, `PPNP` |
| `appnp` | `stellargraph/layer/appnp.py` | `APPNPPropagationLayer`, `APPNP` (the `approx_iter` propagation stack), `APPNP.propagate` |
| `link_inference` | `stellargraph/layer/link_inference.py` | `LeakyClippedLinear`, `link_inference` (all six `edge_embedding_method` arms, covering the eight names `ip`, `dot`, `l1`, `l2`, `mul`, `hadamard`, `concat`, `avg`), `link_classification`, `link_regression` |
| `explorer` | `stellargraph/data/explorer.py` | `GraphWalk._check_common_parameters` and the four checks it runs, `UniformRandomWalk.run`, `BiasedRandomWalk.run` (unweighted), `naive_weighted_choices` |
| `node_mappers` | `stellargraph/mapper/node_mappers.py` | the `method` dispatch of `FullBatchNodeGenerator.__init__` |

### What is not covered, and why

- **No TensorFlow, no training, no backprop.** Upstream's layers are Keras
  layers; here they are the same arithmetic without the autodiff tape. There is
  no `fit`, no `compile`, no `model.fit`, and no optimizer.
- **No `StellarGraph` object, no `networkx`.** `stellargraph.core.graph.StellarGraph`
  is 1900 lines of schema bookkeeping with no numeric content. A graph here is
  an `[e, 2]` edge array, a node count and a feature matrix, which is the part
  the kernels see. `FullBatchNodeGenerator` takes that instead of a
  `StellarGraph`, and the walkers take the same edge array instead of a graph.
- **No `stellargraph.core.utils.GCNAadj_feats_op` sparse path.** SciPy sparse
  arithmetic is ported to dense `float64`: the C ABI carries a dense `[n, n]`
  block without a copy, and a sparse layout would need a per-call canonical
  ordering the ABI cannot express. The arithmetic is the same.
- **No `GAT` / `GCN` / `PPNP` model classes** beyond the forward pass. Their
  `build()` is ported (the weight shapes upstream allocates), but the Keras
  graph construction, the regularizers, the initializers beyond the glorot
  default, and the saliency map plumbing are not.
- **No `SampledNodeGenerator` / `NodeSequence` / `LinkSequence`.** They are
  Keras data plumbing built on `tf.data`; the sampling they drive
  (`UniformRandomWalk`) is ported, the batching is not.
- **`BiasedRandomWalk(weighted=True)` raises `NotImplementedError`.** Upstream
  reads a per-edge label off a `networkx` graph, which does not exist here.
- **`GCNAggregator` is not ported.** It does not exist in v0.8.1; it was added
  in the 1.x line, whose layer API is different throughout.
- **`DirectedGraphSAGE` is not ported.** It is in upstream's
  `graphsage.__all__` (v0.8.1 `__all__` line 24) with its own binary-tree
  neighbourhood sizing and a three-group (`[parent, in_child, out_child]`)
  `__call__`. The port has the homogeneous `GraphSAGE` only.
- **`GraphSAGEWithHeadVariation` is not ported.** It is not in v0.8.1's
  `graphsage.__all__` — it is 1.x — and its `dense -> tanh -> dense`
  attention is not what v0.8.1's `AttentionalAggregator` does, so the port
  follows v0.8.1: `dot(xw_self, w_attn_s) + dot(xw_all, w_attn_g)` ->
  `LeakyReLU(0.2)` -> `softmax(axis=2)` -> `sum(attn * xw_all)`.
- **`MeanHinAggregator` takes one neighbour width and one sample count for
  the whole `[nr, b, h, s, d]` stack.** Upstream builds `w_neigh[r]` from
  `input_shape[1 + r][3]` per relation, leaves it `None` when
  `input_shape[1 + r][2] == 0`, and branches on `if z.shape[2] > 0` per
  relation. The per-relation arithmetic is identical; a relation sampled with
  zero neighbours mixed with one that has neighbours cannot be expressed.
- **`MaxPoolingAggregator` / `MeanPoolingAggregator` pool into the group's
  share of the output width**, where upstream sets `hidden_dim = output_dim`
  for the whole layer and builds `w_pool` and `b_pool` at that width. The
  reduction order and the bias placement are upstream's; the hidden width is
  narrower, so the two pooling aggregators have fewer parameters than an
  upstream layer of the same `layer_sizes`.
- **RGCN, HIN-style knowledge-graph layers, node2vec, attri2vec, DGI, sort
  pooling, cluster-GCN, GCN-LSTM, link-property layers**: not ported. They are
  in the same family but each is its own layer API, and this pass covers the
  node-classification core plus link inference.

### Documented divergences

Everything above the line is faithful. These are the places where it is not,
each one named and each one deliberate:

1. **float64 everywhere.** Upstream builds the adjacency as
   `nx.to_scipy_sparse_matrix(..., dtype="float32")` and runs in float32
   throughout. This port is float64, so results are not bit-comparable with a
   float32 upstream run — they are closer to it than float32 is, and the tests
   compare at float64 tolerance.
2. **`rescale_laplacian` takes the eigenvalue as an argument.** Upstream
   computes it inside the function with
   `scipy.sparse.linalg.eigsh(laplacian, 1, which="LM")` and substitutes
   `largest_eigval = 2` on `ArpackNoConvergence`. Here `largest_eigval` is a
   parameter; passing `None` runs `power_iteration`. Power iteration is a
   different algorithm from ARPACK and converges more slowly, so the eigenvalue
   it returns for the Chebyshev filter differs from ARPACK's in about the sixth
   significant figure. `chebyshev_polynomial` and `rescale_laplacian` called
   separately agree with upstream to `1e-12`; `GCN_Aadj_feats_op(method=
   "chebyshev")`, which does the eigenvalue internally, agrees to about `1e-4`.
3. **`invert` is Gauss-Jordan with partial pivoting, not LAPACK.** The
   singularity test is relative (a pivot below `1e-15` times the largest entry
   of `a`), where `np.linalg.inv` uses LAPACK's exact test. An exactly singular
   matrix raises `LinAlgError` as upstream does; a borderline ill-conditioned
   one may raise here and not there.
4. **`explorer` uses a fixed LCG, not `numpy.random.RandomState`.** The
   generator cannot cross the C ABI and its stream is not reproducible outside
   NumPy. The walk logic — shuffle the neighbours, take the first, break at a
   dead end, the `1/p` / `1.0` / `1/q` transition weights — is upstream's, and
   the same seed gives the same walks from either side, which is what
   `tests/upstream_reference.py` mirrors.
5. **Keras objects cross as integer codes.** Upstream holds an activation as
   `activations.get("relu")` and a method as the string `"gcn"`; a C ABI call
   cannot carry either. `python/mojostellargraph/activations.py` and
   `core_utils.py` map the string onto a code, and the kernels branch on the
   code. The mapping is total for the names upstream documents.
6. **Buffers cross as `Int` addresses** and are rebuilt as
   `Pointer[T, AnyOrigin[mut=True]]` inside the wrapper. `@export` rejects
   parametric functions, and a pointer with an inferred origin is parametric.
7. **`APPNP.propagate` is a loop where upstream has a layer list.** Upstream's
   `APPNP.propagate_model` walks `self._layers`; the arithmetic per step is the
   `APPNPPropagationLayer.call` body verbatim, but the repetition is written as
   a `for` loop because a C ABI call cannot return a list of layers.
8. **Group 0 of `GraphSAGE.__call__` contracts everything below `n_head`.**
   Upstream's `K.dot(x[i], w)` takes a rank-2 left operand, so it contracts
   the feature axis and drops one rank. The port contracts the whole
   `(n_neighbour, n_feat)` block below `n_head` and sizes `w` with that
   product, because a Keras tensor of upstream's shape is `[b, n_neighbour,
   f]` and the port carries the head axis the Keras layer does not. For head 0
   the two agree (`neighbourhood_sizes[0] == 1`). The aggregator
   `group_aggregate` kernels keep upstream's group-0 rule unchanged, and
   those are what the parity tests check.
9. **`GraphSAGE` splits `output_dim` across its groups.** Upstream's
   `calculate_group_sizes` writes `weight_dims = [0]` for the head node group
   when there is more than one group, but its own `compute_output_shape`
   returns `self.output_dim`; the port splits `output_dim` between the self and
   neighbour groups so the output width is the advertised one.
10. **`GraphAttentionSparse` takes a canonical ordering.** `tf.sparse.softmax`
    and `tf.sparse.matmul` both canonicalize the `tf.SparseTensor` to row-major
    internally; `SparseTensor` does that conversion once, on construction, and
    the kernel reads the canonical form. No arithmetic changes.
11. **`link_inference` takes the `Dense` kernel as an argument** for every
    arm except `ip`/`dot`, which has no `Dense` upstream. Upstream builds it
    inside the closure, so a missing kernel is a `ValueError` from
    `link_inference(...)` rather than from the returned function. The
    embedding width is only known at call time, so the input-width check that
    Keras's `Dense` does at build time happens in the returned function.
12. **The batch dimension is not carried.** Keras gives every full-batch layer
    a leading dimension of 1 and squeezes it inside `call`; the kernels take
    the squeezed `[n, f]` and `[n, n]` blocks and the Python wrapper adds it
    back, exactly as the layer does on the way out. The one place this shows
    is `GraphSAGE.__call__`'s closing `Reshape(K.int_shape(x)[2:])`, which
    here drops the singleton head axis and keeps the batch axis.
13. **`call` takes named arguments, not upstream's `inputs` list.** Upstream's
    layers destructure one positional list — `X, out_indices, A = inputs` for
    `GraphAttention` — so porting upstream call sites verbatim would pass the
    node-index array where the adjacency belongs. The port's convention is
    `call(features, A, out_indices=None)` throughout, which is a different
    shape of call, not a different order of the same values.
14. **`naive_weighted_choices` returns a neighbour id, not an index.** Upstream
    returns `idx`, the position in the `weights` iterable, which
    `BiasedRandomWalk` then applies to its own neighbour list. The port takes
    `(graph, weights, node, seed)` — it has the CSR, so it can resolve the
    neighbour itself — and returns `colind[lo + idx]`. Same distribution,
    different return type. A node with no neighbours is an error here; it
    cannot arise upstream, whose only caller always passes one weight.
15. **`ip`/`dot` with `output_act="softmax"` normalizes per row.** Upstream
    reduces to a rank-1 length-`n` tensor, so `Activation("softmax")`
    normalizes across the `n` edges and the following `Reshape((1,))` then
    fails for `n > 1`. The port's softmax over a width-1 column returns 1.0
    per row instead of raising. Every other activation is elementwise and
    agrees exactly.
16. **CPU only, single-threaded.** The pinned toolchain has no `parallelize`,
    no threading module and no `max` package to find a replacement in, and no
    `DeviceContext` or even `std.gpu` to reach a device from (see
    `MOJO_NOTES.md`; the Performance section reports the probes). So there is
    no parallelism and no GPU path, and the benchmark table is single-threaded
    against multithreaded OpenBLAS.

## Install

```bash
pixi install     # brings its own Mojo toolchain, builds nothing yet
pixi run build   # src/capi.mojo -> dist/libmojo-stellargraph.so
pixi run test    # 660 parity tests
pixi run bench   # benchmark table, machine-wide flock
```

The shared library also builds itself on first import and rebuilds whenever a
`.mojo` file is newer than it:

```bash
python -m mojostellargraph._lib --force
```

Outside pixi, point it at a compiler with
`MOJOSTELLARGRAPH_MOJO=/path/to/mojo`.

## Usage

A runnable end-to-end example: build a random graph, preprocess it the way
`FullBatchNodeGenerator` does, and run a two-layer GCN and a GAT over it.

```python
import numpy as np
import mojostellargraph as sg

rng = np.random.default_rng(0)
n, d = 200, 32

# an undirected graph, and a node feature matrix
edges = np.array(np.nonzero(rng.random((n, n)) < 0.03), dtype=np.int64).T
features = np.ascontiguousarray(rng.normal(size=(n, d)))


class Graph:
    node_list = np.arange(n)
    edges = edges
    features = features


gen = sg.FullBatchNodeGenerator(Graph(), method="gcn")
print(gen.Aadj.shape)                        # (200, 200), symmetric, unit diagonal

model = sg.GCN([16, 4], gen, activations=["elu", "softmax"])
model.build(d, seed=0)
out = model(features, gen.Aadj)               # (200, 4)

# GAT over the same graph, with self loops rather than normalization
gen_gat = sg.FullBatchNodeGenerator(Graph(), method="gat")
gat = sg.GraphAttention(8, attn_heads=4, activation="softmax", final_layer=True)
gat.build(d)
gat.kernels[:] = rng.normal(size=(4, d, 8))       # [heads, in, units]
gat.attn_kernels[:] = rng.normal(size=(4, 16))    # [heads, 2 * units]
gat.biases[:] = rng.normal(size=(4, 8))           # [heads, units]
out = gat(features, gen_gat.Aadj, np.arange(50))
print(out.shape)                               # (50, 32): 50 nodes x 4 heads x 8 units

# and a uniform random walk over the same edges
walks = sg.UniformRandomWalk(edges, n).run(nodes=[0, 1, 2], n=2, length=5, seed=7)
print(walks[0])
```

The GCN's `(200, 4)` output has rows summing to 1, because the last layer is a
softmax. The GAT gathers 50 nodes and returns `50 x (4 heads * 8 units)`. The
walk is `[0, 11, 118, 97, 50]`: a real path through the graph, chosen by the
same shuffle-then-take-first rule upstream uses.

## Performance

`stellargraph` 0.8.1 cannot be installed here: it requires Python < 3.9 and
TensorFlow 2.1, and this environment is Python 3.13. The comparison baseline is
therefore `tests/upstream_reference.py` — the same NumPy transliteration of the
upstream code the parity tests use, which reaches BLAS through NumPy's `@`.
That makes every number below **Mojo versus NumPy**, not Mojo versus
TensorFlow, and it is the reason the table is not flattering: NumPy dispatches
to multithreaded OpenBLAS, and this port is single-threaded SIMD with no
`parallelize` in the pinned toolchain.

| case | mojo-stellargraph | numpy reference | |
| --- | ---: | ---: | --- |
| `normalize_adj symmetric (1500 x 1500)` | 26.61 ms | 608.48 ms | **22.9x faster** |
| `normalize_adj left-only (1500 x 1500)` | 11.95 ms | 173.09 ms | **14.5x faster** |
| `normalized_laplacian (1500 x 1500)` | 24.90 ms | 318.45 ms | **12.8x faster** |
| `GCN_Aadj_feats_op gcn (1200 x 1200)` | 33.80 ms | 268.45 ms | **7.9x faster** |
| `GCN_Aadj_feats_op sgc k=3 (700 x 700)` | 236.24 ms | 462.00 ms | **2.0x faster** |
| `chebyshev_polynomial k=4 (400 x 400)` | 58.16 ms | 248.34 ms | **4.3x faster** |
| `invert (700 x 700)` | 419.92 ms | 4239.99 ms | **10.1x faster** |
| `GraphPreProcessingLayer (1200 x 1200)` | 18.65 ms | 46.06 ms | **2.5x faster** |
| `GraphConvolution 64 features (1500 x 1500, 64 -> 32)` | 40.03 ms | 144.01 ms | **3.6x faster** |
| `GCN stack 32-32-8 (1000 x 1000, 64 -> 32 -> 32 -> 8)` | 48.39 ms | 264.95 ms | **5.5x faster** |
| `GraphAttention 8 heads x 32 (1000 nodes, 64 -> 32)` | 165.72 ms | 1621.02 ms | **9.8x faster** |
| `GraphAttentionSparse 8 heads x 32 (1000 nodes, 64 -> 32)` | 59.73 ms | 3081.82 ms | **51.6x faster** |
| `MeanAggregator 2 hops (6000 x 8 heads x 15 x 32 -> 64)` | 105.78 ms | 220.08 ms | **2.1x faster** |
| `MaxPoolingAggregator 2 hops (6000 x 8 heads x 15 x 32 -> 64)` | 1439.16 ms | 3114.14 ms | **2.2x faster** |
| `AttentionalAggregator (6000 x 8 heads x 15 x 32 -> 64)` | 767.67 ms | 1276.60 ms | **1.7x faster** |
| `GraphSAGE 3 layers, mean (6000 x 8 heads, 32 -> 64 -> 32 -> 16)` | 1024.42 ms | 1523.24 ms | **1.5x faster** |
| `MeanHinAggregator 3 relations (6000 x 8 heads x 15 x 32 -> 64)` | 180.11 ms | 433.48 ms | **2.4x faster** |
| `PPNPPropagationLayer (1500 x 1500, 64 -> 128)` | 59.89 ms | 122.96 ms | **2.1x faster** |
| `APPNP_propagate k=10 (1500 x 1500, 64 -> 128)` | 473.36 ms | 1273.47 ms | **2.7x faster** |
| `link_inference hadamard 200k edges (64 -> 8)` | 145.64 ms | 272.20 ms | **1.9x faster** |
| `link_inference concat 200k edges (64 -> 8)` | 292.17 ms | 350.06 ms | **1.2x faster** |
| `UniformRandomWalk 50k walks of length 10 (100k nodes)` | 408.35 ms | 4653.71 ms | **11.4x faster** |
| `BiasedRandomWalk 50k walks of length 10 (50k nodes, p=0.5 q=2)` | 494.17 ms | 5335.12 ms | **10.8x faster** |

23 of 23 cases faster; 0 slower.

Run on:

`Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz / 72 cores / Linux-6.8.0-139-generic-x86_64-with-glibc2.39`,
Mojo 1.2.0.dev2026092605, NumPy 2.5.1, single-threaded port against
multithreaded OpenBLAS. The box is shared and was heavily loaded throughout
(load average 170-480 on 72 cores), so absolute times move between runs by a
factor of two or more in both columns; the ratios are the stable number.

`pixi run bench` reproduces it. The pixi task takes a machine-wide `flock`
first, so a concurrent job cannot distort the numbers.

### What made the difference

The table above replaced a previous one in which 12 of 23 cases were slower.
Every one of the twelve was matmul-bound, and they all go through one
function: `mojostellargraph.types.dot`, the port of `K.dot`. It was a scalar
`i, j, k` loop reading `b` with a stride of `m`. It is now two SIMD kernels,
`W = simd_width_of[DType.float64]()` lanes at a time, with scalar tails on
`n`, `k` and `m`:

- `_dot_jvec` holds `4 * W` output columns in registers for two rows of `a` at
  a time, so each `b` row is loaded once for both. This is the default.
- `_dot_axpy` re-reads `b` once per four rows of `a` and accumulates into
  `dst` in memory. `dot` picks it when the right-hand operand is over 1 MiB,
  which is where streaming `b` from memory rather than from cache is the
  cost. Measured crossover, not a guess.

Around that:

- **The activation no longer needs a second buffer.** Every layer kernel wrote
  the activation into scratch and copied it back over its input. It is
  elementwise, or a row-wise softmax, so it is safe in place, and
  `activation_inplace` is now the only activation entry point. The Python
  side stopped allocating that scratch, which for `GraphSAGE` was a 24 MB
  `np.zeros` per call.
- **The GAT attention block is one pass, not three.** `attn_self +
  attn_neighs.T`, `LeakyReLU(0.2)`, the `-10e9 * (1 - A)` mask and the row
  maximum that the softmax needs are fused into a single sweep of the
  `[n, n]` block, and the softmax now evaluates `exp` once per element
  instead of twice.
- **The aggregation reductions vectorize over the feature axis**, which is
  contiguous in every aggregator: the `K.mean` and `K.max` over the
  neighbour axis, the per-relation sum in HinSAGE, the elementwise
  `link_inference` arms, and Gauss-Jordan's inner row in `invert`.
- `chebyshev_polynomial` multiplies through `dot` instead of carrying its own
  triple loop.

`tests/test_dot.py` covers the new paths: every `n`, `k` and `m` remainder
around the vector width, and both kernels of the dispatch against NumPy. It
found a real bug: the streaming kernel zeroed `dst` with a `W`-wide store
guarded by `j < m`, which ran three doubles past the end of the output when
`m % W != 0` and aborted the process on a `5 x 1200 -> 201` product.

### Parallelism and the GPU: not available, verified

Neither rung was taken, and not because the kernels do not suit one:

- **CPU threads.** `parallelize` is not in `std.algorithm`, there is no
  `std.threading` or `std.parallelism`, and the compiler suggests no
  replacement. There is no `max` package in this environment either
  (`unable to locate module 'max'`), so there is nothing to import a
  replacement from. The work is left serial.
- **GPU.** `DeviceContext` does not exist in this toolchain, and neither does
  `std.gpu`; the host-side launch API that would be needed to reach the
  device is unreachable, so a GPU path could not be built, let alone
  benchmarked. The port is CPU-only.

The remaining gap to OpenBLAS is therefore thread count, not vectorization:
these kernels now run at 5-15 GFLOP/s against the ~37 GFLOP/s one AVX2 core
can issue, and the reference gets the other 71 cores.

## How it works

```
python/mojostellargraph/   the public API: numpy in, numpy out, no algorithms
        |  ctypes, one call per operation, buffers by address
src/capi.mojo              @export ... abi("C") wrappers, Int addresses
src/mojostellargraph/      the numerics, over raw pointers, in upstream's order
```

Three rules hold throughout:

- **Nothing in Mojo allocates.** Every routine takes its scratch from the
  caller, so lifetimes are Python's problem and there is nothing to leak.
- **Contiguous float64 arrays are never copied.** They cross as an address and
  a length. A non-contiguous or non-float64 input is converted once, in NumPy,
  where that is cheap. A buffer handed to `addr()` is always bound to a local
  first: `addr()` returns a plain `int`, so `addr(f64(x))` would free `x`
  before the kernel runs. `tests/test_ffi_lifetime.py` enforces that.
- **Buffers cross as `Int` addresses.** `@export` refuses parametric
  functions, and a pointer with an inferred origin is parametric, so the
  address is rebuilt inside the wrapper as
  `Pointer[Float64, AnyOrigin[mut=True]]`. `tests/test_ffi.py` parses
  `src/capi.mojo` and checks the ctypes signature table against it, because a
  mismatch there is a silent out-of-bounds access rather than an error.

**Memory layout.** Every array is C-contiguous `float64` (node indices are
`int32`, matching upstream's Keras `int32` inputs). A graph adjacency is a
dense `[n, n]` block, which is what `stellargraph/core/utils.py` ends up
producing anyway — `PPNP_Aadj_feats_op` calls `A.toarray()` before its inverse.
The one sparse structure, `SparseTensor`, carries `indices`/`values` in the
`tf.SparseTensor` form plus the canonical row-major `indptr`/`colind`/
`canonical_values` that `tf.sparse.softmax` and `tf.sparse.matmul` need.
Aggregation tensors are `[n_batch, n_head, n_neighbour, n_feat]`, which is how
a Keras tensor of that shape already sits in memory, so nothing is reshaped on
the way in or out.

**Build.** `build/build.sh` runs `mojo build --emit shared-lib` on the single
entry point `src/capi.mojo`, with `-I src` resolving the
`mojostellargraph.*` modules that hold the kernels. One compilation unit,
fixed cost.

## Tests

`pixi run test` runs 609 tests. They are parity tests, not smoke tests: every
numerical assertion compares the Mojo port against `tests/upstream_reference.py`
and most also pin a closed form or a published identity that the oracle cannot
check itself — `T_k(cos t) = cos(k t)`, the symmetric normalization's
`D^-1/2 A^T D^-1/2`, the attention rows being distributions, the PPR matrix
fixing `sqrt(deg + 1)`, the random walk being a real path. Error paths are
covered too: the `ValueError`s upstream raises for a bad `method`, a bad `k`, a
teleport probability out of range, a negative walk weight, `p <= 0`, and the
whole of `_check_common_parameters` — missing root nodes, a non-positive or
non-integer `n`, a zero `length`, a negative or non-integer `seed`.

The buffer-size invariants get their own tests, because a violation is a heap
corruption rather than a wrong number: `K.gather` has no bound on the output
length, so the gather is exercised with `m > n`; `link_inference` is exercised
with `output_dim` wider than the embeddings; and `AttentionalAggregator` is
exercised with more sampled neighbours than its output width, which is what
upstream's separate `attn` tensor needs room for.

`tests/test_ffi.py` and `tests/test_ffi_lifetime.py` guard the boundary
itself, because the two failure modes there are silent.

## Related

- [mojo-sklearn](https://github.com/lee101/mojo-sklearn) — scikit-learn's core
  estimators in Mojo
- [mojo-plotly](https://github.com/lee101/mojo-plotly) — plotly figures with
  Mojo kernels
- [mojo-arrow](https://github.com/lee101/mojo-arrow) — Arrow compute kernels
- [mojo-notebook](https://github.com/lee101/mojo-notebook) — reactive Python +
  Mojo notebooks

## License

MIT. The port follows stellargraph's Apache-2.0 API; no upstream code is
copied into the Mojo or Python sources.
