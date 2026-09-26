"""Shared aliases and the dense primitives the layer kernels are built from.

`stellargraph` is a TensorFlow library, so almost every numeric step in it is a
call to `K.dot` / `K.softmax` / a Keras activation. Those are the primitives
ported here, in the same order and with the same argument shapes, so the layer
files below read like the Keras ones they came from.

Everything is float64. Upstream builds its adjacency as `float32`
(`nx.to_scipy_sparse_matrix(dtype="float32")`); float64 is a documented
divergence, and it is the only dtype the tests compare exactly.
"""

from std.math import exp, log, sqrt, tanh
from std.sys import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()

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
    """`dst[n, m] = a[n, k] @ b[k, m]`, row-major. Upstream: `K.dot`."""
    for i in range(n):
        for j in range(m):
            var acc = 0.0
            for t in range(k):
                acc += a.unsafe_load(i * k + t) * b.unsafe_load(t * m + j)
            dst.unsafe_store(i * m + j, acc)


def dot_vec(a: FPtr, x: FPtr, dst: FPtr, n: Int, k: Int):
    """`dst[n] = a[n, k] @ x[k]`. Upstream: `K.dot(A, v)` on a rank-1 tensor."""
    for i in range(n):
        var acc = 0.0
        for t in range(k):
            acc += a.unsafe_load(i * k + t) * x.unsafe_load(t)
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


def activation(x: FPtr, dst: FPtr, n: Int, d: Int, act: Int, alpha: Float64):
    """Elementwise over an `[n, d]` block. Upstream: `self.activation(output)`.

    `ACT_SOFTMAX` is row-wise, matching how Keras applies a softmax activation.
    """
    for i in range(n * d):
        dst.unsafe_store(
            i, activate(x.unsafe_load(i), act, alpha)
        )
    if act == ACT_SOFTMAX:
        softmax_rows(dst, n, d)


def softmax_rows(x: FPtr, n: Int, d: Int):
    """In-place row-wise softmax. Upstream: `K.softmax(x, axis=1)`.

    Upstream subtracts the row max inside the kernel; that is a numerical
    stability detail rather than part of the definition, and it cancels.
    """
    for r in range(n):
        var m = -1.7976931348623157e308
        for j in range(d):
            m = max(m, x.unsafe_load(r * d + j))
        var s = 0.0
        for j in range(d):
            s += exp(x.unsafe_load(r * d + j) - m)
        for j in range(d):
            x.unsafe_store(r * d + j, exp(x.unsafe_load(r * d + j) - m) / s)


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
        for j in range(d):
            var t = x.unsafe_load(r * d + j)
            s += t * t
        s = sqrt(s)
        var scale = 1.0 / max(s, eps)
        for j in range(d):
            dst.unsafe_store(r * d + j, x.unsafe_load(r * d + j) * scale)


def iput(p: IPtr, idx: Int, v: Int):
    """Store an `Int` into an `Int32` buffer. Node indices cross the FFI as
    `Int` and are stored narrow, which is what upstream's `int32` Keras
    inputs use."""
    p.unsafe_store(idx, Int32(v))


def iget(p: IPtr, idx: Int) -> Int:
    return Int(p.unsafe_load(idx))
