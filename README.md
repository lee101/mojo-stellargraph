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
| `hinsage` | `stellargraph/layer/hinsage.py` | `MeanHinAggregator`, `HinSAGE` |
| `ppnp` | `stellargraph/layer/ppnp.py` | `PPNPPropagationLayer`, `PPNP` |
| `appnp` | `stellargraph/layer/appnp.py` | `APPNPPropagationLayer`, `APPNP.propagate` |
| `link_inference` | `stellargraph/layer/link_inference.py` | `LeakyClippedLinear`, `link_inference` (all seven `edge_embedding_method` branches), `link_classification`, `link_regression` |
| `explorer` | `stellargraph/data/explorer.py` | `UniformRandomWalk.run`, `BiasedRandomWalk.run` (unweighted), `naive_weighted_choices` |
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
   Upstream's `K.dot(x[i], w)` contracts the feature axis and drops one rank.
   For head 0 the two agree (`neighbourhood_sizes[0] == 1`); for the later
   heads the port flattens the neighbour axis into the head axis and sizes the
   weight accordingly. The aggregator `group_aggregate` kernels keep upstream's
   group-0 rule unchanged, and those are what the parity tests check.
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
    branch except `ip`/`dot`, which has no `Dense`. Upstream builds it inside
    the closure. The embedding width is only known at call time, so the
    input-width check that Keras's `Dense` does at build time happens in the
    returned function.
12. **The batch dimension is not carried.** Keras gives every full-batch layer
    a leading dimension of 1 and squeezes it inside `call`; the kernels take
    the squeezed `[n, f]` and `[n, n]` blocks and the Python wrapper adds it
    back, exactly as the layer does on the way out.
13. **CPU only, single-threaded.** The pinned toolchain has no `parallelize`
    and no `DeviceContext` (see `MOJO_NOTES.md`), so there is no parallelism
    and no GPU path. The benchmark table says where that costs.

## Install

```bash
pixi install     # brings its own Mojo toolchain, builds nothing yet
pixi run build   # src/capi.mojo -> dist/libmojo-stellargraph.so
pixi run test    # 561 parity tests
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
to multithreaded OpenBLAS, and this port is single-threaded scalar-plus-SIMD
with no `parallelize` in the pinned toolchain.

BENCH_TABLE_PLACEHOLDER

Run on:

BENCH_MACHINE_PLACEHOLDER

`pixi run bench` reproduces it. The pixi task takes a machine-wide `flock`
first, so a concurrent job cannot distort the numbers.

Reading the table: the cases where this port wins are the ones whose inner
loop is not a matmul — the adjacency transforms, the walks, the aggregations —
and the losses are the dense `n x n x f` products that OpenBLAS blocks,
multithreads and vectorizes. The dense products are the part upstream hands to
`K.dot`, which is a single fused BLAS call; here they are three nested Mojo
loops. The structure is left intact on purpose so a later pass can block it.

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

`pixi run test` runs 561 tests. They are parity tests, not smoke tests: every
numerical assertion compares the Mojo port against `tests/upstream_reference.py`
and most also pin a closed form or a published identity that the oracle cannot
check itself — `T_k(cos t) = cos(k t)`, the symmetric normalization's
`D^-1/2 A^T D^-1/2`, the attention rows being distributions, the PPR matrix
fixing `sqrt(deg + 1)`, the random walk being a real path. Error paths are
covered too: the `ValueError`s upstream raises for a bad `method`, a bad `k`, a
teleport probability out of range, a negative walk weight, `p <= 0`.

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
