"""Port of `stellargraph/layer/link_inference.py` (v0.8.1):
`LeakyClippedLinear.call` and the `edge_function` inside `link_inference`.

`link_inference(output_dim=1, output_act="linear", edge_embedding_method="ip",
clip_limits=None, name="link_inference")` returns a closure that maps a pair
of node embeddings to an edge prediction. The closure is a seven-way branch on
`edge_embedding_method`, each arm followed by a
`Dense(output_dim, activation=output_act)`; all seven are ported below, in
upstream's order.
"""

from mojostellargraph.types import FPtr, activation, relu

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
    """
    var gamma = 1.0 - alpha
    for i in range(n):
        var t = x.unsafe_load(i)
        var x_lo = relu(low - t)
        var x_hi = relu(t - high)
        dst.unsafe_store(i, t + gamma * x_lo - gamma * x_hi)


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
            var acc = 0.0
            for j in range(d):
                acc += x0.unsafe_load(i * d + j) * x1.unsafe_load(i * d + j)
            result.unsafe_store(i, acc)
        # result = Activation(output_act)(result)
        activation(result, le, n, 1, act, alpha)
        if clip:
            LeakyClippedLinear_call(result, le, n, clip_low, clip_high, 0.1)
        var k = 0
        while k < n:
            result.unsafe_store(k, le.unsafe_load(k))
            k += 1
        return 1

    if method == L1:
        # le = K.abs(x[0] - x[1])
        for i in range(n):
            for j in range(d):
                le.unsafe_store(
                    i * d + j,
                    abs(x0.unsafe_load(i * d + j) - x1.unsafe_load(i * d + j)),
                )
    elif method == L2:
        # le = K.square(x[0] - x[1])
        for i in range(n):
            for j in range(d):
                var t = x0.unsafe_load(i * d + j) - x1.unsafe_load(i * d + j)
                le.unsafe_store(i * d + j, t * t)
    elif method == MUL:
        # le = Multiply()([x0, x1])
        for i in range(n):
            for j in range(d):
                le.unsafe_store(
                    i * d + j, x0.unsafe_load(i * d + j) * x1.unsafe_load(i * d + j)
                )
    elif method == CONCAT:
        # le = Concatenate()([x0, x1])
        for i in range(n):
            for j in range(d):
                le.unsafe_store(i * 2 * d + j, x0.unsafe_load(i * d + j))
                le.unsafe_store(i * 2 * d + d + j, x1.unsafe_load(i * d + j))
    elif method == AVG:
        # le = Average()([x0, x1])
        for i in range(n):
            for j in range(d):
                le.unsafe_store(
                    i * d + j,
                    (x0.unsafe_load(i * d + j) + x1.unsafe_load(i * d + j)) / 2.0,
                )

    # Dense(output_dim, activation=output_act): `kernel` is the Keras
    # `(in_dim, output_dim)` weight, row-major.
    var in_dim = 2 * d if method == CONCAT else d
    for i in range(n):
        for j in range(output_dim):
            var acc = 0.0
            for c in range(in_dim):
                acc += le.unsafe_load(i * in_dim + c) * kernel.unsafe_load(c * output_dim + j)
            if has_bias:
                acc += bias.unsafe_load(j)
            result.unsafe_store(i * output_dim + j, acc)
    activation(result, le, n, output_dim, act, alpha)
    if clip:
        LeakyClippedLinear_call(result, le, n * output_dim, clip_low, clip_high, 0.1)
    var k = 0
    while k < n * output_dim:
        result.unsafe_store(k, le.unsafe_load(k))
        k += 1
    return output_dim
