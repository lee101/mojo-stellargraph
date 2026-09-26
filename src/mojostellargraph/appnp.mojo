"""Port of `stellargraph/layer/appnp.py` (v0.8.1):
`APPNPPropagationLayer.call` and the power iteration APPNP stacks around a
trained base model.

APPNP is the approximate half of Klicpera et al. (2018): instead of inverting
`(I - (1 - alpha) A)` up front as PPNP does, it repeatedly applies
`(1 - alpha) A Z + alpha Z` to the base model's output. `APPNP_propagate` is
that repetition, run at inference time to turn a trained base model into a
node model.
"""

from mojostellargraph.types import FPtr, IPtr, dot, iget


def _propagate_step(
    propagated_features: FPtr,
    features: FPtr,
    a: FPtr,
    result: FPtr,
    n: Int,
    f: Int,
    teleport_probability: Float64,
):
    """The arithmetic of one `APPNPPropagationLayer.call`, without the gather."""
    # output = (1 - teleport_probability) * K.dot(A, propagated_features)
    dot(a, propagated_features, result, n, n, f)
    # + teleport_probability * features
    for i in range(n * f):
        result.unsafe_store(
            i,
            (1.0 - teleport_probability) * result.unsafe_load(i)
            + teleport_probability * features.unsafe_load(i),
        )


def APPNPPropagationLayer_call(
    propagated_features: FPtr,
    features: FPtr,
    a: FPtr,
    result: FPtr,
    gathered: FPtr,
    out_indices: IPtr,
    n: Int,
    m: Int,
    f: Int,
    teleport_probability: Float64,
    final_layer: Int,
):
    """Upstream `APPNPPropagationLayer.call`.

    ```
    propagated_features, features, out_indices, *As = inputs
    A = As[0]
    output = (1 - self.teleport_probability) * K.dot(A, propagated_features) \\
             + self.teleport_probability * features
    if self.final_layer:
        output = K.gather(output, out_indices)
    ```
    """
    _propagate_step(
        propagated_features, features, a, result, n, f, teleport_probability
    )

    # `out_indices` need not be sorted, so the gather writes a separate buffer
    # rather than reading rows it has already overwritten.
    if final_layer:
        for s in range(m):
            var r = Int(iget(out_indices, s))
            for j in range(f):
                gathered.unsafe_store(s * f + j, result.unsafe_load(r * f + j))


def APPNP_propagate(
    x: FPtr,
    a: FPtr,
    dst: FPtr,
    work: FPtr,
    n: Int,
    f: Int,
    teleport_probability: Float64,
    k: Int,
):
    """The repeated propagation the APPNP layer stack performs, run at
    inference.

    Upstream expresses this as a chain of `APPNPPropagationLayer`s walked in
    `APPNP.propagate_model`; `k` is the number of propagation layers in
    `self._layers`. There is no upstream function with this signature, so this
    is the one place in the port where a loop stands in for a layer list. The
    arithmetic per step is the `APPNPPropagationLayer.call` body verbatim.
    """
    var i = 0
    while i < n * f:
        dst.unsafe_store(i, x.unsafe_load(i))
        i += 1
    for _ in range(k):
        _propagate_step(dst, x, a, work, n, f, teleport_probability)
        var j = 0
        while j < n * f:
            dst.unsafe_store(j, work.unsafe_load(j))
            j += 1
