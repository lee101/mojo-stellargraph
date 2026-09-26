"""Port of the adjacency branch of `stellargraph/mapper/node_mappers.py`
(v0.8.1), `FullBatchNodeGenerator.__init__`.

Upstream builds a SciPy COO adjacency from the graph and then, depending on
`method`, hands it to one of the transforms in `stellargraph/core/utils.py` or
just adds self loops. The transforms themselves are ported in
`core_utils.mojo`; what lives here is the `method="gat" / "self_loops"` arm,
which is the only one that is a single expression upstream.

```
elif self.method in ["gat", "self_loops"]:
    self.Aadj = self.Aadj + sps.diags(np.ones(n) - self.Aadj.diagonal())
```
"""

from mojostellargraph.core_utils import add_self_loops
from mojostellargraph.types import FPtr


def self_loops(adj: FPtr, dst: FPtr, n: Int):
    """Upstream `self.Aadj + sps.diags(np.ones(n) - self.Aadj.diagonal())`."""
    add_self_loops(adj, dst, n)
