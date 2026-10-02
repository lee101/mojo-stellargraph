"""Port of the adjacency branch of `stellargraph/mapper/full_batch_generators.py`
(upstream 1.2.1), `FullBatchNodeGenerator.__init__`.

Upstream turns a `StellarGraph` into a sparse adjacency and then, depending on
`method`, transforms it. The transforms are ported in `core_utils`; what is
here is the dispatch, which upstream writes in Python and so is Python here.

Two differences from upstream are forced by the boundary and are documented in
the README:

* upstream takes a `StellarGraph` and calls `G.to_adjacency_matrix(weighted=)`
  and `G.node_features()`; here the graph is anything exposing `edges`,
  `node_list` and `features`, which is the pair the walkers already take.
* upstream's `Aadj` is a SciPy sparse matrix by default (`sparse=True`); the
  C ABI carries a dense row-major block, so `Aadj` is dense here and `sparse`
  only records the flag, exactly as upstream's `use_sparse`.
"""

from __future__ import annotations

import numpy as np

from . import core_utils
from ._lib import addr, f64, lib


class FullBatchNodeGenerator:
    """
    A data generator for use with full-batch models on homogeneous graphs.

    ```
    FullBatchNodeGenerator(G, name=None, method="gcn", k=1, sparse=True,
                           transform=None, teleport_probability=0.1,
                           weighted=False)
    ```

    Args:
        G: anything with `node_list`, `features` and `edges`; upstream takes a
           `StellarGraph`.
        name (str): an optional name of the generator.
        method (str): one of ``gcn`` (default), ``sgc``, ``self_loops``, ``gat``,
           ``ppnp`` or ``none``. Upstream 1.2.1 removed ``chebyshev``.
        k (int): the smoothing order for ``sgc``.
        sparse (bool): recorded as ``use_sparse``; upstream's default adjacency
           is sparse, here it is always dense (see the module docstring).
        transform (callable): an optional function taking
           ``(features=..., A=...)``.
        teleport_probability (float): alpha for ``ppnp``.
        weighted (bool): if True, use edge weights from ``G``'s `edge_weights`.
    """

    def __init__(self, G, name=None, method: str = "gcn", k: int = 1,
                 sparse: bool = True, transform=None, teleport_probability: float = 0.1,
                 weighted: bool = False):
        self.graph = G
        self.name = name
        self.k = int(k)
        self.teleport_probability = teleport_probability
        self.method = method
        self.use_sparse = bool(sparse)
        self.weighted = bool(weighted)

        if not hasattr(G, "edges"):
            raise TypeError("Graph must expose `edges` and `node_list`.")
        self.node_list = np.asarray(G.node_list)
        n = len(self.node_list)
        edges = np.asarray(G.edges, dtype=np.int64).reshape(-1, 2)

        # upstream: `G.to_adjacency_matrix(weighted=weighted)`
        self.Aadj = np.zeros((n, n), dtype=np.float64)
        if edges.size:
            self.Aadj[edges[:, 0], edges[:, 1]] = 1.0
        if self.weighted:
            weights = getattr(G, "edge_weights", None)
            if weights is None:
                raise ValueError(
                    "weighted=True needs `edge_weights` on the graph"
                )
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size != edges.shape[0]:
                raise ValueError(
                    "edge_weights has {} entries for {} edges".format(
                        w.size, edges.shape[0]
                    )
                )
            # upstream's `to_adjacency_matrix(weighted=True)` puts the weight
            # on each stored entry, so a zero weight is an absent edge
            self.Aadj[edges[:, 0], edges[:, 1]] = w

        self.features = f64(G.features)

        if transform is not None:
            if not callable(transform):
                raise ValueError("argument 'transform' must be a callable.")
            self.features, self.Aadj = transform(
                features=self.features, A=self.Aadj
            )
        elif self.method in ["gcn", "sgc"]:
            self.features, self.Aadj = core_utils.GCN_Aadj_feats_op(
                features=self.features, A=self.Aadj, k=self.k, method=self.method
            )
        elif self.method in ["gat", "self_loops"]:
            # upstream: `A + sp.diags(np.ones(n) - A.diagonal())`
            self.Aadj = core_utils.add_self_loops(self.Aadj)
        elif self.method in ["ppnp"]:
            if self.use_sparse:
                raise ValueError(
                    "sparse: method='ppnp' requires 'sparse=False', found "
                    "'sparse=True' (consider using the APPNP model for sparse "
                    "support)"
                )
            self.features, self.Aadj = core_utils.PPNP_Aadj_feats_op(
                features=self.features,
                A=self.Aadj,
                teleport_probability=self.teleport_probability,
            )
        elif self.method in [None, "none"]:
            pass
        else:
            raise ValueError(
                "Undefined method for adjacency matrix transformation. "
                "Accepted: 'gcn' (default), 'sgc', and 'self_loops'."
            )

    def flow(self, node_ids, targets=None, use_ilocs: bool = False):
        """Upstream `flow(node_ids, targets=None, use_ilocs=False)` returns a
        Keras sequence; here it returns the `(node_ids, features, A)` triple a
        model call takes. The Keras sequence cannot cross the boundary."""
        node_ids = np.asarray(node_ids)
        return (node_ids, self.features, self.Aadj)

    def num_batch_dims(self) -> int:
        """Upstream `FullBatchGenerator.num_batch_dims`."""
        return 1

    def default_corrupt_input_index_groups(self):
        """Upstream `FullBatchNodeGenerator.default_corrupt_input_index_groups`."""
        return [[0]]

    def __len__(self) -> int:
        return len(self.node_list)