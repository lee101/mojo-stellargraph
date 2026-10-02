# mojo-stellargraph

The graph machine learning kernels of
[stellargraph](https://github.com/stellargraph/stellargraph), implemented in
[Mojo](https://www.modular.com/mojo) and callable from Python with upstream's
names, argument order and defaults.

The target is upstream **1.2.1**, the last release. It is installed for real
(see [Parity against real upstream](#parity-against-real-upstream)), so every
claim below is checked against the actual package, not against a
reimplementation of it.

```python
import numpy as np
import mojostellargraph as sg

rng = np.random.default_rng(0)
n, d = 200, 32
edges = np.array(np.nonzero(rng.random((n, n)) < 0.03), dtype=np.int64).T
features = np.ascontiguousarray(rng.normal(size=(n, d)))

class Graph:
    node_list = np.arange(n)
    edges = edges
    features = features

gen = sg.FullBatchNodeGenerator(Graph(), method="gcn")
model = sg.GCN([16, 4], gen, activations=["elu", "softmax"]).build(d, seed=0)
out = model(features, gen.Aadj)          # (200, 4), rows sum to 1

walks = sg.UniformRandomWalk(edges, n=2, length=5, seed=7).run([0, 1, 2])
biased = sg.BiasedRandomWalk(edges, n=2, p=0.5, q=2.0, length=5).run([0, 1, 2], seed=3)
```

Upstream is a TensorFlow library, so most of a model's numeric work is
`K.dot`, a softmax and a sparse matrix product. Those are the kernels here.
`src/mojostellargraph/*.mojo` is a line-for-line transliteration of the
upstream files, in upstream's order, and `python/mojostellargraph/*.py` is the
same module split the other way, with the ctypes marshalling on the boundary.

## What is covered

| module | upstream | what is ported |
| --- | --- | --- |
| `core_utils` | `stellargraph/core/utils.py` | `normalize_adj` (both branches, `add_self_loops`), `normalized_laplacian`, `calculate_laplacian`, `rescale_laplacian`, `PPNP_Aadj_feats_op`, `GCN_Aadj_feats_op` (methods `gcn`, `sgc`, `none`), plus `power_iteration` and `invert` standing in for `eigsh` and `np.linalg.inv` |
| `node_mappers` | `stellargraph/mapper/full_batch_generators.py` | the `method` dispatch of `FullBatchGenerator.__init__`, `flow`, `weighted` |
| `gcn` | `stellargraph/layer/gcn.py` | `GraphConvolution` (build, call), `GCN` (layer stack, forward) |
| `preprocessing_layer` | `stellargraph/layer/preprocessing_layer.py` | `GraphPreProcessingLayer` |
| `graph_attention` | `stellargraph/layer/graph_attention.py` | `GraphAttention` (dense, both head reductions, the `saliency_map_support` branch) and `GraphAttentionSparse` |
| `graphsage` | `stellargraph/layer/graphsage.py` | `MeanAggregator`, `MaxPoolingAggregator`, `MeanPoolingAggregator`, `AttentionalAggregator`, `GraphSAGE.__call__` and its `l2` normalization |
| `hinsage` | `stellargraph/layer/hinsage.py` | `MeanHinAggregator`, `HinSAGE` (per-layer `activations`, `normalize`, the `output_dim % 2` assertion) |
| `ppnp` | `stellargraph/layer/ppnp.py` | `PPNPPropagationLayer`, `PPNP` |
| `appnp` | `stellargraph/layer/appnp.py` | `APPNPPropagationLayer`, `APPNP` (the `approx_iter` propagation stack), `APPNP.propagate` |
| `link_inference` | `stellargraph/layer/link_inference.py` | `LeakyClippedLinear`, `LinkEmbedding`'s eight operator names, `link_inference`, `link_classification`, `link_regression` |
| `explorer` | `stellargraph/data/explorer.py` | `UniformRandomWalk`, `BiasedRandomWalk` (including `weighted=True`), `naive_weighted_choices`, and the 1.2.1 parameter validation |

### Not covered, and why

- **`chebyshev_polynomial` and `GCN_Aadj_feats_op(method="chebyshev")` are
  gone**, because upstream 1.2.1 removed them: `GCN_Aadj_feats_op` raises
  `ValueError("method 'chebyshev' did not behave correctly and has been
  removed")`. The port raises the same error rather than computing a
  polynomial nothing calls.
- **No TensorFlow, no training, no backprop.** Upstream's layers are Keras
  layers; here they are the same arithmetic without the autodiff tape. There is
  no `fit`, no `compile`, no optimizer.
- **No `StellarGraph` object, no `networkx`.** `StellarGraph` is ~1900 lines
  of schema bookkeeping with no numeric content. A graph here is an `[e, 2]`
  edge array, a node count and a feature matrix, which is the part the kernels
  see. Any object exposing `node_list`, `edges` and `features` works.
- **Sparse arithmetic is ported to dense `float64`.** The C ABI carries a dense
  `[n, n]` block without a copy, and a sparse layout would need a per-call
  canonical ordering the ABI cannot express. Upstream's SciPy matrices are
  densified on the way in, so an upstream-shaped call still works.
- **`flow` returns a tuple, not a Keras `NodeSequence`.** The sequence is
  `tf.data` plumbing; the triple a model call consumes is what crosses here.
- **No `SampledNodeGenerator` / `NodeSequence` / `LinkSequence`.** They are
  Keras data plumbing; the sampling they drive (`UniformRandomWalk`) is
  ported, the batching is not.
- **`GCNAggregator`, `DirectedGraphSAGE`, `GraphSAGEWithHeadVariation`, RGCN,
  node2vec, attri2vec, DGI, sort pooling, cluster-GCN, GCN-LSTM, the
  link-property layers**: not ported. Each is its own layer API; this pass
  covers the node-classification core plus link inference.
- **CPU only, single-threaded.** See [Documented divergences](#documented-divergences).

### Documented divergences

Each of these is a place where the port is not a like-for-like translation, named
and deliberate.

1. **float64 everywhere.** Upstream builds the adjacency as
   `to_adjacency_matrix(dtype="float32")` and runs Keras in float32. This port
   is float64. The SciPy functions therefore agree to ~1e-16 and the Keras
   layers to float32 epsilon; the parity tests assert each at its own real
   precision rather than one loose tolerance.
2. **`rescale_laplacian` takes the eigenvalue as an argument.** Upstream
   computes it with `scipy.sparse.linalg.eigsh(laplacian, 1, which="LM")` and
   substitutes `largest_eigval = 2` on `ArpackNoConvergence`. Here
   `largest_eigval` is a parameter; passing `None` runs `power_iteration`,
   which is a different algorithm from ARPACK. The `(2/lambda) L - I`
   arithmetic is upstream's and is parity-checked with the eigenvalue supplied
   explicitly, which isolates the part that is actually shared.
3. **`invert` is Gauss-Jordan with partial pivoting, not LAPACK.** The
   singularity test is relative (a pivot below `1e-15` times the largest entry
   of `a`) where `np.linalg.inv` uses LAPACK's exact test. An exactly singular
   matrix raises `LinAlgError` as upstream does; a borderline ill-conditioned
   one may raise here and not there.
4. **`explorer` uses a fixed LCG, not `numpy.random.RandomState`.** A
   `RandomState` cannot cross the C ABI and its stream is not reproducible
   outside NumPy. The walk logic is upstream's — shuffle the neighbours, take
   the first, break at a dead end, the `1/p` / `1.0` / `1/q` transition weights
   — and the neighbour relation is asserted equal to upstream's
   `neighbor_arrays`. The node *sequences* for a given seed are not expected to
   match, and the tests do not claim they do.
5. **`leaky_relu` uses alpha=0.2.** Upstream calls
   `tf.keras.activations.get("leaky_relu")`, which in the Keras 2 that
   `stellargraph` 1.2.1 requires resolves to `tf.nn.leaky_relu(x, alpha=0.2)`.
   That was measured against the installed package, not assumed.
6. **Keras objects cross as integer codes.** Upstream holds an activation as
   `activations.get("relu")` and a method as the string `"gcn"`; a C ABI call
   cannot carry either. `activations.py` and `core_utils.py` map the string
   onto a code and the kernels branch on it.
7. **Buffers cross as `Int` addresses** and are rebuilt as
   `Pointer[T, AnyOrigin[mut=True]]` inside the wrapper, because `@export`
   rejects parametric functions and a pointer with an inferred origin is
   parametric. A pointer is non-nullable, so an argument that is genuinely
   absent crosses as the address `0` and the pointer is only constructed inside
   the branch that reads it (`biased_random_walk`'s `edge_weights_addr`).
8. **`APPNP.propagate` is a loop where upstream has a layer list.** The
   arithmetic per step is the `APPNPPropagationLayer.call` body verbatim, but
   the repetition is a `for` loop because a C ABI call cannot return a list of
   layers.
9. **Group 0 of `GraphSAGE.__call__` contracts everything below `n_head`.**
   Upstream's `K.dot(x[i], w)` takes a rank-2 left operand, so it contracts
   the feature axis and drops one rank. The port contracts the whole
   `(n_neighbour, n_feat)` block below `n_head`, because a Keras tensor of
   upstream's shape is `[b, n_neighbour, f]` and the port carries the head axis
   the Keras layer does not. For head 0 the two agree
   (`neighbourhood_sizes[0] == 1`). The aggregator `group_aggregate` kernels
   keep upstream's group-0 rule unchanged.
10. **`GraphSAGE` splits `output_dim` across its groups.** Upstream's
    `calculate_group_sizes` writes `weight_dims = [0]` for the head-node group
    when there is more than one group, but its own `compute_output_shape`
    returns `self.output_dim`; the port splits `output_dim` so the output
    width is the advertised one.
11. **`MaxPoolingAggregator` / `MeanPoolingAggregator` pool into the group's
    share of the output width**, where upstream sets
    `hidden_dim = output_dim` for the whole layer. The reduction order and the
    bias placement are upstream's; the hidden width is narrower, so these two
    aggregators have fewer parameters than an upstream layer of the same
    `layer_sizes`.
12. **`MeanHinAggregator` takes one neighbour width and one sample count for
    the whole `[nr, b, h, s, d]` stack.** Upstream builds `w_neigh[r]` per
    relation from `input_shape[1 + r][3]`, leaves it `None` when
    `input_shape[1 + r][2] == 0`, and branches per relation. The per-relation
    arithmetic is identical; a relation sampled with zero neighbours mixed with
    one that has neighbours cannot be expressed.
13. **`GraphAttentionSparse` takes a canonical ordering.** `tf.sparse.softmax`
    and `tf.sparse.matmul` both canonicalize the `tf.SparseTensor` to row-major
    internally; `SparseTensor` does that once, on construction. No arithmetic
    changes.
14. **`link_inference` takes the `Dense` kernel as an argument** for every arm
    except `ip`/`dot`, which has no `Dense` upstream. The embedding width is
    only known at call time, so the input-width check Keras's `Dense` does at
    build time happens in the returned function.
15. **The batch dimension is not carried.** Keras gives every full-batch layer
    a leading dimension of 1 and squeezes it inside `call`; the kernels take
    the squeezed `[n, f]` and `[n, n]` blocks.
16. **`call` takes named arguments, not upstream's `inputs` list.** Upstream
    destructures one positional list — `X, out_indices, A = inputs` — so
    porting the call sites verbatim would pass the node-index array where the
    adjacency belongs. The convention here is
    `call(features, A, out_indices=None)` throughout.
17. **`ip`/`dot` with `output_act="softmax"` normalizes per row.** Upstream
    reduces to a rank-1 tensor, so `Activation("softmax")` normalizes across
    the `n` edges and the following `Reshape((1,))` then fails for `n > 1`.
    The port's softmax over a width-1 column returns 1.0 per row instead of
    raising. Every other activation is elementwise and agrees exactly.
18. **`seed=None` walks are reproducible rather than drawn from global state.**
    `seed=None` is upstream's default and means "take a seed from NumPy's global
    random state". A C ABI call cannot carry that, so `None` crosses as the seed
    `0`: the walk is the same algorithm over the same neighbours as the
    `seed=<int>` case above, and the same `None` always replays. Passing a seed
    explicitly is the way to vary a walk.

## Parity against real upstream

`stellargraph` 1.2.1 declares `python_requires < 3.9` and depends on
TensorFlow 2.x, so it cannot live in this repo's Python 3.13 pixi environment.
`tools/setup_upstream_env.sh` creates a separate `uv`-managed Python 3.8
environment under `.upstream/` (gitignored), and `tools/dump_upstream.py` runs
under it to dump reference outputs into `tests/upstream_refs/`:

```bash
pixi run setup-upstream   # create .upstream/.venv (stellargraph 1.2.1)
pixi run dump-upstream   # regenerate tests/upstream_refs/*.npz
```

Those arrays are produced by calling the installed package, not by
transliterating it: `normalize_adj`, `GCN_Aadj_feats_op`, `PPNP_Aadj_feats_op`,
`rescale_laplacian` (including its `eigsh` call), `FullBatchNodeGenerator`,
`GraphConvolution`, `LinkEmbedding`, `LeakyClippedLinear`, `link_inference`,
`keras.activations.get` and `StellarGraph.neighbor_arrays` are all exercised.
`tests/test_upstream_parity.py` asserts the Mojo port reproduces them; the
reference arrays are committed, so the test suite runs without the Python 3.8
environment present.

The benchmark, by contrast, compares against `tests/upstream_reference.py`, a
NumPy transliteration run under the same interpreter and thread settings as the
port. See [Benchmarks](#benchmarks) for why.

## Install

```bash
pixi install     # brings its own Mojo toolchain, builds nothing yet
pixi run build   # src/capi.mojo -> dist/libmojo-stellargraph.so
pixi run test    # the parity suite
pixi run bench   # benchmark table, under a machine-wide flock
```

`pixi.toml` pins `mojo` and `max` to matching dates
(`1.2.0.dev2026092905` / `26.7.0.dev2026092905`). They are released in
lockstep and a mismatched pair fails to solve, so they must move together.

The shared library also builds itself on first import and rebuilds whenever a
`.mojo` file is newer than it:

```bash
python -m mojostellargraph._lib          # force a rebuild
MOJOSTELLARGRAPH_LIB=/path/to/lib.so     # use a prebuilt library, never compile
```

## How it works

**FFI.** `src/capi.mojo` is the only compilation unit: one `@export("name")`
function per kernel, each a flat list of `Int` addresses, sizes and scalars,
with an explicit `abi("C")` effect. `@export` rejects parametric functions, so
every buffer crosses as an address and is rebuilt inside the wrapper as
`Pointer[T, AnyOrigin[mut=True]]`. The body of each wrapper is the upstream
call and nothing else; the branch structure, argument order and arithmetic live
in `src/mojostellargraph/`, where they can be read next to the originals.

**Memory layout.** Matrices are dense row-major `float64` with the C ABI
carrying no shape, so the Python side allocates the output and all scratch and
owns every allocation; nothing in Mojo allocates, so nothing can leak. A graph
is CSR `(indptr, colind)` in `int32`, and walks come back as an
`[n_roots * n, length]` block plus a length vector, because a list of
variable-length lists does not survive the ABI.

**Why a NumPy baseline in the benchmark.** Upstream's numeric work runs inside
TensorFlow, on a multithreaded BLAS, in float32. Timing the Mojo kernels
against that would compare a single-threaded float64 kernel loop to a tuned
vendored BLAS and say almost nothing about the port. The benchmark therefore
runs both sides in the same process, on the same interpreter, with the same
thread count, against `tests/upstream_reference.py` — the same NumPy
transliteration of upstream the unit tests use. Where that reference hits
`numpy.linalg` the comparison is genuinely against BLAS, and the table says so.

**The benchmark checks its own results.** A timing table says nothing about
whether a kernel was right, so `bench.py` compares each case against the
reference before it times it and refuses to print a table if any case
disagrees. A wrong kernel fails the benchmark rather than being timed
alongside a correct one.

## Benchmarks

```bash
pixi run bench
```

The `pixi` task holds a machine-wide `flock` on `/tmp/mojo-bench.lock` so a
concurrent job cannot distort the numbers; always go through it.

Measured on the machine below. **Provenance:** the `pixi run bench` invocation
could not be used for this run, because another job on this host had held the
machine-wide lock for 23 hours of CPU with a queue of waiting jobs ahead of it.
The numbers below come from running `bench/bench.py` directly — the same script
the task runs — on an otherwise idle machine. The `flock` exists to exclude
contention, and there was none to exclude. These are real measurements.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores,
Linux 6.8.0-142-generic, glibc 2.39, Python 3.13, Mojo 1.2.0.dev2026092905.

| case | mojo-stellargraph | numpy reference | |
| --- | ---: | ---: | --- |
| `normalize_adj symmetric (1500 x 1500)` | 15.26 ms | 295.62 ms | **19.4x faster** |
| `normalize_adj left-only (1500 x 1500)` | 6.29 ms | 175.05 ms | **27.8x faster** |
| `normalized_laplacian (1500 x 1500)` | 18.50 ms | 359.19 ms | **19.4x faster** |
| `GCN_Aadj_feats_op gcn (1200 x 1200)` | 21.28 ms | 300.92 ms | **14.1x faster** |
| `GCN_Aadj_feats_op sgc k=3 (700 x 700)` | 161.08 ms | 542.93 ms | **3.4x faster** |
| `invert (700 x 700)` | 327.94 ms | 6722.33 ms | **20.5x faster** |
| `calculate_laplacian (1500 x 1500)` | 7.38 ms | 396.32 ms | **53.7x faster** |
| `GraphPreProcessingLayer (1200 x 1200)` | 19.76 ms | 42.95 ms | **2.2x faster** |
| `GraphConvolution 64 features (1500 x 1500, 64 -> 32)` | 43.51 ms | 31.93 ms | 1.4x slower |
| `GCN stack 32-32-8 (1000 x 1000, 64 -> 32 -> 32 -> 8)` | 45.73 ms | 173.93 ms | **3.8x faster** |
| `GraphAttention 8 heads x 32 (1000 nodes, 64 -> 32)` | 100.72 ms | 1333.41 ms | **13.2x faster** |
| `GraphAttentionSparse 8 heads x 32 (1000 nodes, 64 -> 32)` | 38.92 ms | 1831.79 ms | **47.1x faster** |
| `MeanAggregator 2 hops (6000 x 8 heads x 15 x 32 -> 64)` | 73.43 ms | 243.78 ms | **3.3x faster** |
| `MaxPoolingAggregator 2 hops (6000 x 8 heads x 15 x 32 -> 64)` | 654.66 ms | 1668.59 ms | **2.5x faster** |
| `AttentionalAggregator (6000 x 8 heads x 15 x 32 -> 64)` | 558.04 ms | 1043.89 ms | **1.9x faster** |
| `GraphSAGE 3 layers, mean (6000 x 8 heads, 32 -> 64 -> 32 -> 16)` | 490.98 ms | 1059.01 ms | **2.2x faster** |
| `MeanHinAggregator 3 relations (6000 x 8 heads x 15 x 32 -> 64)` | 159.94 ms | 244.81 ms | **1.5x faster** |
| `PPNPPropagationLayer (1500 x 1500, 64 -> 128)` | 39.46 ms | 15.21 ms | 2.6x slower |
| `APPNP_propagate k=10 (1500 x 1500, 64 -> 128)` | 402.36 ms | 1059.51 ms | **2.6x faster** |
| `link_inference hadamard 200k edges (64 -> 8)` | 96.65 ms | 199.09 ms | **2.1x faster** |
| `link_inference concat 200k edges (64 -> 8)` | 167.47 ms | 257.28 ms | **1.5x faster** |
| `UniformRandomWalk 50k walks of length 10 (100k nodes)` | 209.33 ms | 1592.60 ms | **7.6x faster** |
| `BiasedRandomWalk 50k walks of length 10 (50k nodes, p=0.5 q=2)` | 198.34 ms | 1680.73 ms | **8.5x faster** |

21 of 23 cases faster; 2 slower.

Reading the table:

- The wins are where upstream's work is elementwise or a sparse gather — the
  normalization family, the aggregators, the walks, GAT — because the reference
  there is a NumPy expression allocating temporaries while the port is a
  single pass over the caller's buffer.
- The two losses are honest. `GraphConvolution` and `PPNPPropagationLayer` are
  essentially `A @ features @ kernel`, a dense matmul, and NumPy hands that to
  a multithreaded BLAS. The port is single-threaded and sweeps the whole
  `n x n` block even where the matrix is sparse, so it loses by 1.4x and 2.6x.
  These are exactly the cases where a BLAS call in the Python wrapper would
  win, and that is the obvious next step; this pass prioritised fidelity over
  it.
- `invert` at 20.5x is the port's Gauss-Jordan against `numpy.linalg.inv`: a
  single-threaded kernel loop beating a multithreaded LAPACK on a 700x700
  solve. That says how little work this matrix holds, not that the algorithm
  is better; at larger sizes LAPACK wins.