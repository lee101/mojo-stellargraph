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


def _scale_half(x: FPtr, count: Int):
    """`x[:count] *= 0.5`, in place: the `Average()` of the concat operators."""
    var half = Vec(0.5)
    var i = 0
    while i + W <= count:
        x.unsafe_store(i, x.unsafe_load[width=W](i) * half)
        i += W
    while i < count:
        x.unsafe_store(i, x.unsafe_load(i) * 0.5)
        i += 1


def _dot_concat(
    x0: FPtr, x1: FPtr, kernel: FPtr, dst: FPtr, n: Int, d: Int, out_dim: Int
):
    """`dst[i] = concat([x0[i], x1[i]]) @ kernel`, without the concatenation.

    Two rows of the virtual `[n, 2 * d]` left operand are contracted at once, so
    each `kernel` column is read once for both rows and the concatenation itself
    is never stored. `a0`/`b0` are the two `d`-runs of row `i` and `a1`/`b1`
    the same of row `i + 1`.
    """
    var i = 0
    while i + 2 <= n:
        # row i:     x0[i] supplies kernel[0:d]   and x1[i] supplies kernel[d:2d]
        # row i + 1: x0[i+1] supplies kernel[0:d] and x1[i+1] supplies kernel[d:2d]
        var a0 = x0.unsafe_offset(i * d)
        var b0 = x1.unsafe_offset(i * d)
        var a1 = x0.unsafe_offset((i + 1) * d)
        var b1 = x1.unsafe_offset((i + 1) * d)
        var d0 = dst.unsafe_offset(i * out_dim)
        var d1 = d0.unsafe_offset(out_dim)
        var j = 0
        while j + 2 * W <= out_dim:
            # two output column blocks at a time, and both rows: `k*` is row `i`
            # and `c*` row `i + 1`, each accumulator folding that row's
            # `kernel[0:d]` and `kernel[d:2d]` halves together
            var k0 = Vec(0.0)
            var k1 = Vec(0.0)
            var c0 = Vec(0.0)
            var c1 = Vec(0.0)
            var t = 0
            while t < d:
                var lo = kernel.unsafe_load[width=W](t * out_dim + j)
                var hi = kernel.unsafe_load[width=W](t * out_dim + j + W)
                var lo2 = kernel.unsafe_load[width=W]((t + d) * out_dim + j)
                var hi2 = kernel.unsafe_load[width=W]((t + d) * out_dim + j + W)
                k0 = k0 + lo * a0.unsafe_load(t) + lo2 * b0.unsafe_load(t)
                k1 = k1 + hi * a0.unsafe_load(t) + hi2 * b0.unsafe_load(t)
                c0 = c0 + lo * a1.unsafe_load(t) + lo2 * b1.unsafe_load(t)
                c1 = c1 + hi * a1.unsafe_load(t) + hi2 * b1.unsafe_load(t)
                t += 1
            d0.unsafe_store(j, k0)
            d0.unsafe_store(j + W, k1)
            d1.unsafe_store(j, c0)
            d1.unsafe_store(j + W, c1)
            j += 2 * W
        while j < out_dim:
            # the columns past the vector lanes: one at a time, both rows
            var acc0 = 0.0
            var acc1 = 0.0
            var t = 0
            while t < d:
                acc0 += a0.unsafe_load(t) * kernel.unsafe_load(t * out_dim + j)
                acc0 += b0.unsafe_load(t) * kernel.unsafe_load((t + d) * out_dim + j)
                acc1 += a1.unsafe_load(t) * kernel.unsafe_load(t * out_dim + j)
                acc1 += b1.unsafe_load(t) * kernel.unsafe_load((t + d) * out_dim + j)
                t += 1
            d0.unsafe_store(j, acc0)
            d1.unsafe_store(j, acc1)
            j += 1
        i += 2
    if i < n:
        _dot_concat_row(x0.unsafe_offset(i * d), x1.unsafe_offset(i * d), kernel, dst.unsafe_offset(i * out_dim), d, out_dim)


def _dot_concat_row(
    arow: FPtr, brow: FPtr, kernel: FPtr, drow: FPtr, d: Int, out_dim: Int
):
    """`_dot_concat` for the odd row left over by its two-row blocking."""
    var j = 0
    while j + W <= out_dim:
        var c0 = Vec(0.0)
        var c1 = Vec(0.0)
        var t = 0
        while t < d:
            var u = arow.unsafe_load(t)
            var p = brow.unsafe_load(t)
            c0 = c0 + kernel.unsafe_load[width=W](t * out_dim + j) * u
            c1 = c1 + kernel.unsafe_load[width=W]((t + d) * out_dim + j) * p
            t += 1
        drow.unsafe_store(j, c0 + c1)
        j += W
    while j < out_dim:
        var acc = 0.0
        var t = 0
        while t < d:
            acc += arow.unsafe_load(t) * kernel.unsafe_load(t * out_dim + j)
            acc += brow.unsafe_load(t) * kernel.unsafe_load((t + d) * out_dim + j)
            t += 1
        drow.unsafe_store(j, acc)
        j += 1


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
    elif method == AVG:
        # le = Average()([x0, x1])
        for i in range(n):
            var r0 = x0.unsafe_offset(i * d)
            var r1 = x1.unsafe_offset(i * d)
            var drow = le.unsafe_offset(i * d)
            var j = 0
            while j + W <= d:
                drow.unsafe_store(j, r0.unsafe_load[width=W](j) + r1.unsafe_load[width=W](j))
                j += W
            while j < d:
                drow.unsafe_store(j, r0.unsafe_load(j) + r1.unsafe_load(j))
                j += 1
        # `Average()` is a divide by two; folding it in here as a second pass
        # over the same block is cheaper than a divide in the loop above
        _scale_half(le, n * d)

    # Dense(output_dim, activation=output_act): `kernel` is the Keras
    # `(in_dim, output_dim)` weight, row-major.
    if method == CONCAT:
        # `le = Concatenate()([x0, x1])` is upstream's statement, but the
        # concatenation is never read: the `Dense` below consumes it once and
        # nothing else looks at it. Materialising it is `2 * n * d` doubles
        # stored and then re-read -- 400 MB of traffic at the benchmark's
        # 200k x 64 -- to compute `x0 @ kernel[:d] + x1 @ kernel[d:]`. Doing
        # that product directly reads both halves where they already are and
        # writes neither. `le` stays allocated: it is the destination of the
        # activation and the LeakyClippedLinear below.
        _dot_concat(x0, x1, kernel, result, n, d, output_dim)
    else:
        dot(le, kernel, result, n, d, output_dim)
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
