"""Port of `stellargraph/layer/gcn.py` (v0.8.1): `GraphConvolution` and `GCN`.

Upstream is a Keras layer, so `GraphConvolution.call` is the entire numeric
body and `GCN` is a stack of `[Dropout, GraphConvolution]` pairs evaluated in
order. `GraphConvolution_call` keeps the upstream statement order verbatim;
`GCN` is the Python class in python/mojostellargraph/gcn.py, which walks the
same list.
"""

from mojostellargraph.types import (
    FPtr,
    IPtr,
    activation_inplace,
    dot,
    iget,
)


def GraphConvolution_call(
    features: FPtr,
    adj: FPtr,
    kernel: FPtr,
    bias: FPtr,
    result: FPtr,
    work: FPtr,
    out_indices: IPtr,
    n: Int,
    m: Int,
    f: Int,
    units: Int,
    use_bias: Int,
    act: Int,
    alpha: Float64,
    final_layer: Int,
):
    """Upstream `GraphConvolution.call`.

    ```
    features, out_indices, *As = inputs
    batch_dim, n_nodes, _ = K.int_shape(features)     # batch dim squeezed off
    A = As[0]
    h_graph = K.dot(A, features)
    output = K.dot(h_graph, self.kernel)
    if self.bias is not None:
        output += self.bias
    output = self.activation(output)
    if self.final_layer:
        output = K.gather(output, out_indices)
    ```

    `m` is the number of output indices and is only read when `final_layer` is
    set. The batch dimension Keras squeezes and re-expands is not carried here;
    the Python wrapper adds it, exactly as the layer does on the way out.
    """
    # h_graph = K.dot(A, features)
    dot(adj, features, work, n, n, f)
    # output = K.dot(h_graph, self.kernel)
    dot(work, kernel, result, n, f, units)

    # if self.bias is not None: output += self.bias
    if use_bias:
        for i in range(n):
            for j in range(units):
                result.unsafe_store(
                    i * units + j,
                    result.unsafe_load(i * units + j) + bias.unsafe_load(j),
                )

    # output = self.activation(output)
    activation_inplace(result, n, units, act, alpha)

    # if self.final_layer: output = K.gather(output, out_indices)
    # The gather is staged in `work` first: `out_indices` need not be sorted,
    # and writing straight into `result` would read a row already overwritten.
    if final_layer:
        for s in range(m):
            var r = Int(iget(out_indices, s))
            for j in range(units):
                work.unsafe_store(s * units + j, result.unsafe_load(r * units + j))
        var g = 0
        while g < m * units:
            result.unsafe_store(g, work.unsafe_load(g))
            g += 1
