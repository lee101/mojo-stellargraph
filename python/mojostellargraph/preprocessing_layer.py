"""Port of `stellargraph/layer/preprocessing_layer.py` (v0.8.1):
`GraphPreProcessingLayer`.

GCN takes an already-normalized Laplacian, but `GCN(..., generator)` with
`method="none"` inserts this layer so the normalization stays inside the model.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class GraphPreProcessingLayer:
    """
    Args:
        num_of_nodes (int): The number of nodes in the graph.
    """

    def __init__(self, num_of_nodes, **kwargs):
        self.num_of_nodes = int(num_of_nodes)
        self.output_dims = (self.num_of_nodes, self.num_of_nodes)

    def call(self, adj: np.ndarray) -> np.ndarray:
        """The adjacency pre-processing GCN requires: symmetric, with self
        loops, and normalized."""
        adj = f64(adj)
        # upstream's `call` takes every size from `adj.shape[0]`;
        # `num_of_nodes` only fixes `output_dims`
        n = adj.shape[0]
        dst = np.zeros((n, n), dtype=np.float64)
        work = np.zeros((n, n), dtype=np.float64)
        rowsum = np.zeros(n, dtype=np.float64)
        lib().msg_GraphPreProcessingLayer_call(
            addr(adj), addr(dst), addr(work), addr(rowsum), n
        )
        return dst

    def __call__(self, adj: np.ndarray) -> np.ndarray:
        return self.call(adj)
