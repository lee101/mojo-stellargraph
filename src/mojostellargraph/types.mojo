"""Shared aliases and the dense primitives the layer kernels are built from.

`stellargraph` is a TensorFlow library, so almost every numeric step in it is a
call to `K.dot` / `K.softmax` / a Keras activation. Those are the primitives
ported here, in the same order and with the same argument shapes, so the layer
files below read like the Keras ones they came from.

Everything is float64. Upstream builds its adjacency as `float32`
(`nx.to_scipy_sparse_matrix(dtype="float32")`); float64 is a documented
divergence, and it is the only dtype the tests compare exactly.
"""

from max.algorithm import parallelize
from std.math import exp, log, sqrt, tanh
from std.sys import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime Vec = SIMD[DType.float64, W]

# Row blocks per work item, and the product size at which `_dot_axpy` spreads
# its blocks over workers. 8 * 8 * 1e6 = 6.4e7: below that the measured
# `parallelize` launch (about 4 ms here) is longer than the serial loop it
# replaces, so the threshold sits above every shape a small caller passes.
comptime DOT_BLOCK = 8
comptime DOT_PAR_THRESHOLD = 16000000
comptime DOT_WORKERS = 16


def _dot_axpy_parallel(a: FPtr, b: FPtr, dst: FPtr, n: Int, k: Int, m: Int):
    """`_dot_axpy` with its row blocks handed to `parallelize`.

    Each work item owns a disjoint run of complete four-row blocks, so no two
    workers write the same `dst` row and no reduction is needed. The odd
    trailing rows and the four-row tail of each item go through the same
    block routine the serial path uses, which keeps the arithmetic identical.
    """
    var blocks = n // 4
    var items = (blocks + DOT_BLOCK - 1) // DOT_BLOCK

    def work(item: Int) {imm}:
        var lo = item * DOT_BLOCK
        var hi = lo + DOT_BLOCK
        if hi > blocks:
            hi = blocks
        var b4 = lo * 4
        while b4 < hi * 4:
            _dot_axpy_block(a, b, dst, b4, k, m)
            b4 += 4

    parallelize(work, items, DOT_WORKERS)

    var i = blocks * 4
    while i < n:
        _dot_axpy_row(a, b, dst, i, k, m)
        i += 1


def _dot_axpy_block(a: FPtr, b: FPtr, dst: FPtr, i: Int, k: Int, m: Int):
    """The four-row body of `_dot_axpy`, at rows `i .. i + 3`."""
    var a0 = a.unsafe_offset(i * k)
    var a1 = a0.unsafe_offset(k)
    var a2 = a1.unsafe_offset(k)
    var a3 = a2.unsafe_offset(k)
    var d0 = dst.unsafe_offset(i * m)
    var d1 = d0.unsafe_offset(m)
    var d2 = d1.unsafe_offset(m)
    var d3 = d2.unsafe_offset(m)
    var zero = Vec(0.0)
    var j = 0
    while j + W <= m:
        d0.unsafe_store(j, zero)
        d1.unsafe_store(j, zero)
        d2.unsafe_store(j, zero)
        d3.unsafe_store(j, zero)
        j += W
    while j < m:
        d0.unsafe_store(j, 0.0)
        d1.unsafe_store(j, 0.0)
        d2.unsafe_store(j, 0.0)
        d3.unsafe_store(j, 0.0)
        j += 1
    var t = 0
    while t < k:
        var x0 = a0.unsafe_load(t)
        var x1 = a1.unsafe_load(t)
        var x2 = a2.unsafe_load(t)
        var x3 = a3.unsafe_load(t)
        j = 0
        while j + W <= m:
            var bv = b.unsafe_load[width=W](t * m + j)
            d0.unsafe_store(j, d0.unsafe_load[width=W](j) + bv * x0)
            d1.unsafe_store(j, d1.unsafe_load[width=W](j) + bv * x1)
            d2.unsafe_store(j, d2.unsafe_load[width=W](j) + bv * x2)
            d3.unsafe_store(j, d3.unsafe_load[width=W](j) + bv * x3)
            j += W
        while j < m:
            var bv = b.unsafe_load(t * m + j)
            d0.unsafe_store(j, d0.unsafe_load(j) + bv * x0)
            d1.unsafe_store(j, d1.unsafe_load(j) + bv * x1)
            d2.unsafe_store(j, d2.unsafe_load(j) + bv * x2)
            d3.unsafe_store(j, d3.unsafe_load(j) + bv * x3)
            j += 1
        t += 1


def _dot_axpy_row(a: FPtr, b: FPtr, dst: FPtr, i: Int, k: Int, m: Int):
    """`_dot_axpy` for the single row left over by its four-row blocking."""
    var arow = a.unsafe_offset(i * k)
    var drow = dst.unsafe_offset(i * m)
    var zero = Vec(0.0)
    var j = 0
    while j + W <= m:
        drow.unsafe_store(j, zero)
        j += W
    while j < m:
        drow.unsafe_store(j, 0.0)
        j += 1
    var t = 0
    while t < k:
        var av = arow.unsafe_load(t)
        j = 0
        while j + W <= m:
            drow.unsafe_store(j, drow.unsafe_load[width=W](j) + av * b.unsafe_load[width=W](t * m + j))
            j += W
        while j < m:
            drow.unsafe_store(j, drow.unsafe_load(j) + av * b.unsafe_load(t * m + j))
            j += 1
        t += 1

# Upstream holds activations as Keras objects (`activations.get("relu")`).
# A C ABI call cannot take one, so the Python side maps the upstream string to
# one of these codes and the kernel branches on the code. See
# python/mojostellargraph/activations.py for the mapping.
comptime ACT_LINEAR = 0
comptime ACT_RELU = 1
comptime ACT_ELU = 2
comptime ACT_SOFTMAX = 3
comptime ACT_SIGMOID = 4
comptime ACT_TANH = 5
comptime ACT_SOFTPLUS = 6
comptime ACT_LEAKY_RELU = 7
comptime ACT_HARD_SIGMOID = 8
comptime ACT_EXP = 9
comptime ACT_SOFT_SIGN = 10


def dot(a: FPtr, b: FPtr, dst: FPtr, n: Int, k: Int, m: Int):
    """`dst[n, m] = a[n, k] @ b[k, m]`, row-major. Upstream: `K.dot`.

    Two SIMD kernels, chosen on the size of `b`. Both multiply-accumulate
    `W` output columns at a time; which is faster depends only on whether
    `b` is still in cache by the time the next block of rows of `a` needs
    it again.

    `_dot_jvec` holds the output in registers and re-reads `b` once per two
    rows of `a`, so it wins whenever `b` fits in the last-level cache, and
    also on the short-`k` products the pooling aggregators run. `_dot_axpy`
    re-reads `b` once per four rows of `a` and accumulates in `dst` in
    memory, so it wins on the products where streaming `b` dominates. The
    threshold is 1 MiB, measured; past it the two kernels cross over.
    """
    if k * m * 8 <= 1048576:
        _dot_jvec(a, b, dst, n, k, m)
    else:
        _dot_axpy(a, b, dst, n, k, m)


def _dot_jvec(a: FPtr, b: FPtr, dst: FPtr, n: Int, k: Int, m: Int):
    """`dot` two rows of `a` at a time, `4 * W` output columns in registers.

    Each `b` row is loaded once and used for both rows of `a`, which halves
    the traffic in `b` against a one-row-at-a-time kernel and leaves four
    independent accumulator chains per FMA port.
    """
    var i = 0
    while i + 2 <= n:
        var a0 = a.unsafe_offset(i * k)
        var a1 = a0.unsafe_offset(k)
        var d0 = dst.unsafe_offset(i * m)
        var d1 = d0.unsafe_offset(m)
        var j = 0
        while j + 4 * W <= m:
            var c00 = Vec(0.0)
            var c01 = Vec(0.0)
            var c02 = Vec(0.0)
            var c03 = Vec(0.0)
            var c10 = Vec(0.0)
            var c11 = Vec(0.0)
            var c12 = Vec(0.0)
            var c13 = Vec(0.0)
            var t = 0
            while t < k:
                var b0 = b.unsafe_load[width=W](t * m + j)
                var b1 = b.unsafe_load[width=W](t * m + j + W)
                var b2 = b.unsafe_load[width=W](t * m + j + 2 * W)
                var b3 = b.unsafe_load[width=W](t * m + j + 3 * W)
                c00 = c00 + b0 * a0.unsafe_load(t)
                c01 = c01 + b1 * a0.unsafe_load(t)
                c02 = c02 + b2 * a0.unsafe_load(t)
                c03 = c03 + b3 * a0.unsafe_load(t)
                c10 = c10 + b0 * a1.unsafe_load(t)
                c11 = c11 + b1 * a1.unsafe_load(t)
                c12 = c12 + b2 * a1.unsafe_load(t)
                c13 = c13 + b3 * a1.unsafe_load(t)
                t += 1
            d0.unsafe_store(j, c00)
            d0.unsafe_store(j + W, c01)
            d0.unsafe_store(j + 2 * W, c02)
            d0.unsafe_store(j + 3 * W, c03)
            d1.unsafe_store(j, c10)
            d1.unsafe_store(j + W, c11)
            d1.unsafe_store(j + 2 * W, c12)
            d1.unsafe_store(j + 3 * W, c13)
            j += 4 * W
        while j + W <= m:
            var c0 = Vec(0.0)
            var c1 = Vec(0.0)
            var t = 0
            while t < k:
                var bv = b.unsafe_load[width=W](t * m + j)
                c0 = c0 + bv * a0.unsafe_load(t)
                c1 = c1 + bv * a1.unsafe_load(t)
                t += 1
            d0.unsafe_store(j, c0)
            d1.unsafe_store(j, c1)
            j += W
        while j < m:
            var s0 = 0.0
            var s1 = 0.0
            var t = 0
            while t < k:
                var bv = b.unsafe_load(t * m + j)
                s0 += a0.unsafe_load(t) * bv
                s1 += a1.unsafe_load(t) * bv
                t += 1
            d0.unsafe_store(j, s0)
            d1.unsafe_store(j, s1)
            j += 1
        i += 2
    if i < n:
        _dot_row(a.unsafe_offset(i * k), b, dst.unsafe_offset(i * m), k, m)


def _dot_row(arow: FPtr, b: FPtr, drow: FPtr, k: Int, m: Int):
    """`_dot_jvec` for the odd row left over by its two-row blocking."""
    var j = 0
    while j + 4 * W <= m:
        var c0 = Vec(0.0)
        var c1 = Vec(0.0)
        var c2 = Vec(0.0)
        var c3 = Vec(0.0)
        var t = 0
        while t < k:
            var av = arow.unsafe_load(t)
            c0 = c0 + b.unsafe_load[width=W](t * m + j) * av
            c1 = c1 + b.unsafe_load[width=W](t * m + j + W) * av
            c2 = c2 + b.unsafe_load[width=W](t * m + j + 2 * W) * av
            c3 = c3 + b.unsafe_load[width=W](t * m + j + 3 * W) * av
            t += 1
        drow.unsafe_store(j, c0)
        drow.unsafe_store(j + W, c1)
        drow.unsafe_store(j + 2 * W, c2)
        drow.unsafe_store(j + 3 * W, c3)
        j += 4 * W
    while j + W <= m:
        var c0 = Vec(0.0)
        var t = 0
        while t < k:
            c0 = c0 + b.unsafe_load[width=W](t * m + j) * arow.unsafe_load(t)
            t += 1
        drow.unsafe_store(j, c0)
        j += W
    while j < m:
        var acc = 0.0
        var t = 0
        while t < k:
            acc += arow.unsafe_load(t) * b.unsafe_load(t * m + j)
            t += 1
        drow.unsafe_store(j, acc)
        j += 1


def _dot_axpy(a: FPtr, b: FPtr, dst: FPtr, n: Int, k: Int, m: Int):
    """`dot` as four rank-1 updates at a time: one streaming pass over `b`
    per block of four rows of `a`, accumulating in `dst`.

    Above `DOT_PAR_THRESHOLD` flops the blocks of rows are spread over
    workers instead of walked one after another. `b` is streamed once per
    block of four rows in both forms, so each worker streams its own copy of
    it; the product is large enough by then for the shared read to be the
    smaller cost. Below the threshold the launch costs more than the loop it
    replaces -- measured at roughly 4 ms per launch on this host -- so the
    serial path stays and every small `dot` a caller makes pays nothing.
    """
    if n * k * m >= DOT_PAR_THRESHOLD:
        _dot_axpy_parallel(a, b, dst, n, k, m)
        return

    var i = 0
    while i + 4 <= n:
        _dot_axpy_block(a, b, dst, i, k, m)
        i += 4
    while i < n:
        _dot_axpy_row(a, b, dst, i, k, m)
        i += 1


def dot_vec(a: FPtr, x: FPtr, dst: FPtr, n: Int, k: Int):
    """`dst[n] = a[n, k] @ x[k]`. Upstream: `K.dot(A, v)` on a rank-1 tensor."""
    for i in range(n):
        var arow = a.unsafe_offset(i * k)
        var c = SIMD[DType.float64, W](0.0)
        var t = 0
        while t + W <= k:
            c = c + arow.unsafe_load[width=W](t) * x.unsafe_load[width=W](t)
            t += W
        var acc = c.reduce_add()
        while t < k:
            acc += arow.unsafe_load(t) * x.unsafe_load(t)
            t += 1
        dst.unsafe_store(i, acc)


def relu(x: Float64) -> Float64:
    return max(x, 0.0)


def leaky_relu(x: Float64, alpha: Float64) -> Float64:
    return x if x >= 0.0 else alpha * x


def sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


def softplus(x: Float64) -> Float64:
    return log(1.0 + exp(x))


def activate(x: Float64, act: Int, alpha: Float64) -> Float64:
    """One element of the activation upstream holds as a Keras object."""
    if act == ACT_RELU:
        return relu(x)
    if act == ACT_ELU:
        return x if x >= 0.0 else exp(x) - 1.0
    if act == ACT_SIGMOID:
        return sigmoid(x)
    if act == ACT_TANH:
        return tanh(x)
    if act == ACT_SOFTPLUS:
        return softplus(x)
    if act == ACT_LEAKY_RELU:
        return leaky_relu(x, alpha)
    if act == ACT_HARD_SIGMOID:
        return max(0.0, min(1.0, 0.2 * x + 0.5))
    if act == ACT_SOFT_SIGN:
        return x / (1.0 + abs(x))
    if act == ACT_EXP:
        return exp(x)
    return x


def activate_vec(x: Vec, act: Int, alpha: Float64) -> Vec:
    """`activate` on a vector of `W` elements.

    The stdlib's `exp`, `log` and `tanh` are defined for `SIMD` as well as
    for `Float64`, but a Mojo function cannot be generic over the two, so
    the chain is written out a second time rather than once per element.
    """
    if act == ACT_RELU:
        return max(x, Vec(0.0))
    if act == ACT_ELU:
        return max(x, Vec(0.0)) + min(exp(x) - Vec(1.0), Vec(0.0))
    if act == ACT_SIGMOID:
        return Vec(1.0) / (Vec(1.0) + exp(-x))
    if act == ACT_TANH:
        return tanh(x)
    if act == ACT_SOFTPLUS:
        return log(Vec(1.0) + exp(x))
    if act == ACT_LEAKY_RELU:
        return max(x, Vec(0.0)) + min(Vec(alpha) * x, Vec(0.0))
    if act == ACT_HARD_SIGMOID:
        return max(Vec(0.0), min(Vec(1.0), Vec(0.2) * x + Vec(0.5)))
    if act == ACT_SOFT_SIGN:
        return x / (Vec(1.0) + abs(x))
    if act == ACT_EXP:
        return exp(x)
    return x


def softmax_row(x: FPtr, r: Int, d: Int, m: Float64):
    """In-place softmax of one `[d]` row, given that row's maximum.

    `exp` is evaluated once per element, not once per element per stage:
    the shifted exponential overwrites the row, and the last pass divides
    by the sum. A caller that has just written the row and computed its
    maximum on the way past can skip the reduction and call this.
    """
    var s = 0.0
    var j = 0
    while j + W <= d:
        var e = exp(x.unsafe_load[width=W](r * d + j) - m)
        x.unsafe_store(r * d + j, e)
        s += e.reduce_add()
        j += W
    while j < d:
        var e = exp(x.unsafe_load(r * d + j) - m)
        x.unsafe_store(r * d + j, e)
        s += e
        j += 1
    var inv = 1.0 / s
    j = 0
    while j + W <= d:
        x.unsafe_store(r * d + j, x.unsafe_load[width=W](r * d + j) * inv)
        j += W
    while j < d:
        x.unsafe_store(r * d + j, x.unsafe_load(r * d + j) * inv)
        j += 1


def softmax_rows(x: FPtr, n: Int, d: Int):
    """In-place row-wise softmax. Upstream: `K.softmax(x, axis=1)`.

    Upstream subtracts the row max inside the kernel; that is a numerical
    stability detail rather than part of the definition, and it cancels.
    """
    for r in range(n):
        var m = -1.7976931348623157e308
        var j = 0
        while j + W <= d:
            m = max(Vec(m), x.unsafe_load[width=W](r * d + j)).reduce_max()
            j += W
        while j < d:
            m = max(m, x.unsafe_load(r * d + j))
            j += 1
        softmax_row(x, r, d, m)


def softmax_dim2(x: FPtr, dst: FPtr, b: Int, h: Int, k: Int):
    """Softmax over the third axis of a `[b, h, k]` tensor.

    Upstream: `K.softmax(attn_u, axis=2)` in `AttentionalAggregator.call`.
    """
    for r in range(b * h):
        var m = -1.7976931348623157e308
        for j in range(k):
            m = max(m, x.unsafe_load(r * k + j))
        var s = 0.0
        for j in range(k):
            s += exp(x.unsafe_load(r * k + j) - m)
        for j in range(k):
            dst.unsafe_store(r * k + j, exp(x.unsafe_load(r * k + j) - m) / s)


def l2_normalize(x: FPtr, dst: FPtr, n: Int, d: Int):
    """Row-wise L2 normalization. Upstream: `K.l2_normalize(x, axis=-1)`.

    Keras clamps the norm away from zero with its own epsilon; the same clamp
    is applied here so the two agree on all-zero rows.
    """
    var eps = 1e-7
    for r in range(n):
        var s = 0.0
        var j = 0
        while j + W <= d:
            var v = x.unsafe_load[width=W](r * d + j)
            s += (v * v).reduce_add()
            j += W
        while j < d:
            var t = x.unsafe_load(r * d + j)
            s += t * t
            j += 1
        s = sqrt(s)
        var scale = 1.0 / max(s, eps)
        j = 0
        while j + W <= d:
            dst.unsafe_store(r * d + j, x.unsafe_load[width=W](r * d + j) * scale)
            j += W
        while j < d:
            dst.unsafe_store(r * d + j, x.unsafe_load(r * d + j) * scale)
            j += 1


def activation_inplace(x: FPtr, n: Int, d: Int, act: Int, alpha: Float64):
    """Elementwise activation over an `[n, d]` block, in place.

    Upstream: `self.activation(output)`. Every layer kernel here used to
    write into a second scratch buffer and copy the result back over the
    input; the copy was one extra pass over `n * d` doubles per call and
    nothing else. Both the elementwise branch and `softmax_rows` are safe
    on aliased input and output. `ACT_SOFTMAX` is row-wise, matching how
    Keras applies a softmax activation.
    """
    var i = 0
    while i + W <= n * d:
        x.unsafe_store(i, activate_vec(x.unsafe_load[width=W](i), act, alpha))
        i += W
    while i < n * d:
        x.unsafe_store(i, activate(x.unsafe_load(i), act, alpha))
        i += 1
    if act == ACT_SOFTMAX:
        softmax_rows(x, n, d)


def iput(p: IPtr, idx: Int, v: Int):
    """Store an `Int` into an `Int32` buffer. Node indices cross the FFI as
    `Int` and are stored narrow, which is what upstream's `int32` Keras
    inputs use."""
    p.unsafe_store(idx, Int32(v))


def iget(p: IPtr, idx: Int) -> Int:
    return Int(p.unsafe_load(idx))
