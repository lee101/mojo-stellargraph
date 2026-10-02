"""mojo-stellargraph

The graph machine learning kernels of
[stellargraph](https://github.com/stellargraph/stellargraph), implemented in
[Mojo](https://www.modular.com/mojo) and callable from Python with the same
names and signatures.

    from mojostellargraph import FullBatchNodeGenerator, GCN

    generator = FullBatchNodeGenerator(graph, method="gcn")
    model = GCN(layer_sizes=[32, 4], generator=generator,
                activations=["elu", "softmax"])
    out = model.build(generator.features.shape[1])(generator.features,
                                                  generator.Aadj)

The modules mirror the upstream ones they port:

| module | upstream |
| --- | --- |
| `mojostellargraph.core_utils` | `stellargraph/core/utils.py` |
| `mojostellargraph.gcn` | `stellargraph/layer/gcn.py` |
| `mojostellargraph.preprocessing_layer` | `stellargraph/layer/preprocessing_layer.py` |
| `mojostellargraph.graph_attention` | `stellargraph/layer/graph_attention.py` |
| `mojostellargraph.graphsage` | `stellargraph/layer/graphsage.py` |
| `mojostellargraph.hinsage` | `stellargraph/layer/hinsage.py` |
| `mojostellargraph.ppnp` | `stellargraph/layer/ppnp.py` |
| `mojostellargraph.appnp` | `stellargraph/layer/appnp.py` |
| `mojostellargraph.link_inference` | `stellargraph/layer/link_inference.py` |
| `mojostellargraph.explorer` | `stellargraph/data/explorer.py` |
| `mojostellargraph.node_mappers` | `stellargraph/mapper/node_mappers.py` |
"""

from . import activations  # noqa: F401
from ._lib import build, lib
from .appnp import APPNP, APPNPPropagationLayer
from .core_utils import (
    GCN_Aadj_feats_op,
    PPNP_Aadj_feats_op,
    add_self_loops,
    calculate_laplacian,
    invert,
    normalize_adj,
    normalized_laplacian,
    power_iteration,
    rescale_laplacian,
)
from .explorer import (
    BiasedRandomWalk,
    GraphWalk,
    UniformRandomWalk,
    csr_from_edges,
    naive_weighted_choices,
)
from .gcn import GCN, GraphConvolution
from .graph_attention import GraphAttention, GraphAttentionSparse
from .graphsage import (
    AttentionalAggregator,
    GraphSAGE,
    MaxPoolingAggregator,
    MeanAggregator,
    MeanPoolingAggregator,
)
from .hinsage import HinSAGE, MeanHinAggregator
from .link_inference import (
    LeakyClippedLinear,
    link_classification,
    link_inference,
    link_regression,
)
from .node_mappers import FullBatchNodeGenerator
from .ppnp import PPNP, PPNPPropagationLayer
from .preprocessing_layer import GraphPreProcessingLayer
from .sparse import SparseTensor, sparse_matmul_dense

__version__ = "0.1.0"

__all__ = [
    "APPNP",
    "APPNPPropagationLayer",
    "AttentionalAggregator",
    "BiasedRandomWalk",
    "FullBatchNodeGenerator",
    "GCN",
    "GCN_Aadj_feats_op",
    "GraphAttention",
    "GraphAttentionSparse",
    "GraphConvolution",
    "GraphPreProcessingLayer",
    "GraphSAGE",
    "GraphWalk",
    "HinSAGE",
    "LeakyClippedLinear",
    "MaxPoolingAggregator",
    "MeanAggregator",
    "MeanHinAggregator",
    "MeanPoolingAggregator",
    "PPNP",
    "PPNPPropagationLayer",
    "PPNP_Aadj_feats_op",
    "SparseTensor",
    "UniformRandomWalk",
    "activations",
    "add_self_loops",
    "build",
    "calculate_laplacian",
    "csr_from_edges",
    "lib",
    "link_classification",
    "link_inference",
    "link_regression",
    "invert",
    "naive_weighted_choices",
    "normalize_adj",
    "normalized_laplacian",
    "power_iteration",
    "rescale_laplacian",
    "sparse_matmul_dense",
]
