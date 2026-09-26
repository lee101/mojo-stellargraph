"""Port of the adjacency branch of `stellargraph/mapper/node_mappers.py`
(v0.8.1), `FullBatchNodeGenerator.__init__`.

Upstream turns a `StellarGraph` into a SciPy adjacency and then, depending on
`method`, transforms it. The transforms are ported in `core_utils`; what is
here is the dispatch, which upstream writes in Python and so is Python here.
"""

from __future__ import annotations

import numpy as np

from . import core_utils
from ._lib import addr, f64, lib


class FullBatchNodeGenerator:
    """
    A generator class for operating over the full graph.

    ```
    FullBatchNodeGenerator(G, name=None, method="gcn", k=1, sparse=True,
                           transform=None, teleport_probability=0.1)
    ```

    Args:
        G: anything with `nodes()`, `node_list`, `features` and an adjacency;
           upstream takes a `StellarGraph`, this takes the pair the walkers
           take — an `[e, 2]` edge array and the number of nodes — plus the
           node feature matrix.
        method: 'gcn' (default), 'chebyshev', 'sgc', 'self_loops', 'gat',
           'ppnp' or 'none'.
        k: smoothing order for 'sgc', Chebyshev series order for 'chebyshev'.
        sparse: if True a sparse adjacency is produced, as upstream's default.
        transform: optional callable taking `(features=..., A=...)`.
        teleport_probability: alpha for 'ppnp'.
    """

    def __init__(self, G, name=None, method: str = "gcn", k: int = 1,
                 sparse: bool = True, transform=None, teleport_probability: float = 0.1):
        self.graph = G
        self.name = name
        self.k = int(k)
        self.teleport_probability = teleport_probability
        self.method = method
        self.use_sparse = bool(sparse)

        if not hasattr(G, "edges"):
            raise TypeError("Graph must expose `edges` and `node_list`.")
        self.node_list = np.asarray(G.node_list)
        n = len(self.node_list)
        edges = np.asarray(G.edges, dtype=np.int64).reshape(-1, 2)
        # upstream: nx.to_scipy_sparse_matrix(..., dtype="float32")
        self.Aadj = np.zeros((n, n), dtype=np.float64)
        if edges.size:
            self.Aadj[edges[:, 0], edges[:, 1]] = 1.0
        self.features = f64(G.features)

        if transform is not None:
            if not callable(transform):
                raise ValueError("argument 'transform' must be a callable.")
            self.features, self.Aadj = transform(
                features=self.features, A=self.Aadj
            )
        elif self.method in ["gcn", "chebyshev", "sgc"]:
            self.features, self.Aadj = core_utils.GCN_Aadj_feats_op(
                features=self.features, A=self.Aadj, k=self.k, method=self.method
            )
        elif self.method in ["gat", "self_loops"]:
            self.Aadj = _self_loops(self.Aadj, n)
        elif self.method in ["ppnp"]:
            if self.use_sparse:
                raise ValueError(
                    "use_sparse=true' is incompatible with 'ppnp'."
                    "Set 'use_sparse=True' or consider using the APPNP model instead."
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
                "Accepted: 'gcn' (default), 'chebyshev','sgc', and 'self_loops'."
            )

    def flow(self, node_ids, targets=None):
        """Upstream `flow(node_ids, targets=None)` returns a Keras sequence; here
        it returns the `(node_ids, features, A)` triple a model call takes."""
        node_ids = np.asarray(node_ids)
        return (node_ids, self.features, self.Aadj)

    def __len__(self) -> int:
        return len(self.node_list)


def _self_loops(adj, n):
    out = np.zeros((n, n), dtype=np.float64)
    a = f64(adj)
    lib().msg_self_loops(addr(a), addr(out), n)
    return out
