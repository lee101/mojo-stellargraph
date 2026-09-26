"""Port of `stellargraph/layer/link_inference.py` (v0.8.1):
`LeakyClippedLinear.call` and the `edge_function` inside `link_inference`.

`link_inference(output_dim=1, output_act="linear", edge_embedding_method="ip",
clip_limits=None, name="link_inference")` returns a closure that maps a pair
of node embeddings to an edge prediction. The closure is a seven-way branch on
`edge_embedding_method`, each arm followed by a
`Dense(output_dim, activation=output_act)`; all seven are ported below, in
upstream's order.
"""

from mojostellargraph.types import (
    FPtr,
    Vec,
    W,
    activation_inplace,
    dot,
    relu,
)

# Upstream compares `edge_embedding_method` against strings; a C ABI call
# cannot carry one, so these codes stand in and the Python wrapper keeps the
# string dispatch visible.
comptime IP = 0
comptime DOT = 1
comptime L1 = 2
comptime L2 = 3
comptime MUL = 4
comptime CONCAT = 5
comptime AVG = 6


def LeakyClippedLinear_call(
    x: FPtr, dst: FPtr, n: Int, low: Float64, high: Float64, alpha: Float64
):
    """Upstream `LeakyClippedLinear.call`.

    ```
    self.gamma = K.cast_to_floatx(1 - alpha)
    self.lo    = K.cast_to_floatx(low)
    self.hi    = K.cast_to_floatx(high)

    x_lo = K.relu(self.lo - x)
    x_hi = K.relu(x - self.hi)
    return x + self.gamma * x_lo - self.gamma * x_hi
    ```
    """
    var gamma = 1.0 - alpha
    var i = 0
    while i + W <= n:
        var t = x.unsafe_load[width=W](i)
        var x_lo = max(Vec(low) - t, Vec(0.0))
        var x_hi = max(t - Vec(high), Vec(0.0))
        dst.unsafe_store(i, t + Vec(gamma) * x_lo - Vec(gamma) * x_hi)
        i += W
    while i < n:
        var t = x.unsafe_load(i)
        var x_lo = relu(low - t)
        var x_hi = relu(t - high)
        dst.unsafe_store(i, t + gamma * x_lo - gamma * x_hi)
        i += 1


def link_inference_edge_function(
    x0: FPtr,
    x1: FPtr,
    le: FPtr,
    kernel: FPtr,
    bias: FPtr,
    result: FPtr,
    n: Int,
    d: Int,
    output_dim: Int,
    method: Int,
    act: Int,
    alpha: Float64,
    has_bias: Int,
    clip: Int,
    clip_low: Float64,
    clip_high: Float64,
) -> Int:
    """Upstream `link_inference(...)([x0, x1])`.

    ```
    if method in ("ip", "dot"):
        out = K.sum(x[0] * x[1], axis=-1, keepdims=False)
        out = Activation(output_act)(out)
        out = Reshape((1,))(out)
    elif method == "l1":     le = K.abs(x[0] - x[1]);     out = Dense(output_dim, act)(le)
    elif method == "l2":     le = K.square(x[0] - x[1]);  out = Dense(output_dim, act)(le)
    elif method in ("mul", "hadamard"): le = Multiply()([x0, x1]);    out = Dense(...)
    elif method == "concat": le = Concatenate()([x0, x1]); out = Dense(...)
    elif method == "avg":    le = Average()([x0, x1]);    out = Dense(...)
    if clip_limits:
        out = LeakyClippedLinear(low=clip_limits[0], high=clip_limits[1], alpha=0.1)(out)
    ```

    `le` is the `2 * d` buffer for `concat` and the `d` buffer otherwise.
    Returns the width of `out`.
    """
    if method == IP or method == DOT:
        # out = K.sum(x[0] * x[1], axis=-1, keepdims=False)
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var c = SIMD[DType.float64, W](0.0)
            var j = 0
            while j + W <= d:
                c = c + r0.unsafe_load[width=W](j) * r1.unsafe_load[width=W](j)
                j += W
            var acc = c.reduce_add()
            while j < d:
                acc += r0.unsafe_load(j) * r1.unsafe_load(j)
                j += 1
            result.unsafe_store(i, acc)
        # result = Activation(output_act)(result)
        activation_inplace(result, n, 1, act, alpha)
        if clip:
            LeakyClippedLinear_call(result, result, n, clip_low, clip_high, 0.1)
        return 1

    if method == L1:
        # le = K.abs(x[0] - x[1])
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var drow = le.unsafe_offset(i * d)
            var j = 0
            while j + W <= d:
                drow.unsafe_store(j, abs(r0.unsafe_load[width=W](j) - r1.unsafe_load[width=W](j)))
                j += W
            while j < d:
                drow.unsafe_store(j, abs(r0.unsafe_load(j) - r1.unsafe_load(j)))
                j += 1
    elif method == L2:
        # le = K.square(x[0] - x[1])
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var drow = le.unsafe_offset(i * d)
            var j = 0
            while j + W <= d:
                var t = r0.unsafe_load[width=W](j) - r1.unsafe_load[width=W](j)
                drow.unsafe_store(j, t * t)
                j += W
            while j < d:
                var t = r0.unsafe_load(j) - r1.unsafe_load(j)
                drow.unsafe_store(j, t * t)
                j += 1
    elif method == MUL:
        # le = Multiply()([x0, x1])
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var drow = le.unsafe_offset(i * d)
            var j = 0
            while j + W <= d:
                drow.unsafe_store(j, r0.unsafe_load[width=W](j) * r1.unsafe_load[width=W](j))
                j += W
            while j < d:
                drow.unsafe_store(j, r0.unsafe_load(j) * r1.unsafe_load(j))
                j += 1
    elif method == CONCAT:
        # le = Concatenate()([x0, x1])
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var drow = le.unsafe_offset(i * 2 * d)
            var j = 0
            while j + W <= d:
                drow.unsafe_store(j, r0.unsafe_load[width=W](j))
                drow.unsafe_store(j + d, r1.unsafe_load[width=W](j))
                j += W
            while j < d:
                drow.unsafe_store(j, r0.unsafe_load(j))
                drow.unsafe_store(j + d, r1.unsafe_load(j))
                j += 1
    elif method == AVG:
        # le = Average()([x0, x1])
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var drow = le.unsafe_offset(i * d)
            var j = 0
            while j + W <= d:
                drow.unsafe_store(
                    j, (r0.unsafe_load[width=W](j) + r1.unsafe_load[width=W](j)) / 2.0
                )
                j += W
            while j < d:
                drow.unsafe_store(j, (r0.unsafe_load(j) + r1.unsafe_load(j)) / 2.0)
                j += 1

    # Dense(output_dim, activation=output_act): `kernel` is the Keras
    # `(in_dim, output_dim)` weight, row-major.
    var in_dim = 2 * d if method == CONCAT else d
    dot(le, kernel, result, n, in_dim, output_dim)
    if has_bias:
        for i in range(n):
            var drow = result.unsafe_offset(i * output_dim)
            var j = 0
            while j + W <= output_dim:
                drow.unsafe_store(
                    j, drow.unsafe_load[width=W](j) + bias.unsafe_load[width=W](j)
                )
                j += W
            while j < output_dim:
                drow.unsafe_store(j, drow.unsafe_load(j) + bias.unsafe_load(j))
                j += 1
    activation_inplace(result, n, output_dim, act, alpha)
    if clip:
        LeakyClippedLinear_call(result, result, n * output_dim, clip_low, clip_high, 0.1)
    return output_dim
