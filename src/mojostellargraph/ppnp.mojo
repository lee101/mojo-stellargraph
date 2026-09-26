"""Port of `stellargraph/layer/ppnp.py` (v0.8.1): `PPNPPropagationLayer.call`.

PPNP propagates the node features once through the personalized-PageRank
matrix the generator precomputes (`method="ppnp"`), then the `PPNP` model puts
a stack of fully connected layers in front of it.
"""

from mojostellargraph.types import FPtr, IPtr, dot, iget


def PPNPPropagationLayer_call(
    features: FPtr,
    a: FPtr,
    result: FPtr,
    gathered: FPtr,
    out_indices: IPtr,
    n: Int,
    m: Int,
    f: Int,
    final_layer: Int,
):
    """Upstream `PPNPPropagationLayer.call`.

    ```
    features, out_indices, *As = inputs          # batch dim squeezed off
    A = As[0]
    output = K.dot(A, features)
    if self.final_layer:
        output = K.gather(output, out_indices)
    ```
    """
    # output = K.dot(A, features)
    dot(a, features, result, n, n, f)

    # if self.final_layer: output = K.gather(output, out_indices)
    # `out_indices` need not be sorted, so the gather writes a separate buffer
    # rather than reading rows it has already overwritten.
    if final_layer:
        for s in range(m):
            var r = Int(iget(out_indices, s))
            for j in range(f):
                gathered.unsafe_store(s * f + j, result.unsafe_load(r * f + j))
