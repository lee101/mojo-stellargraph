"""The activation names upstream passes to `activations.get(...)`, mapped to
the integer codes the Mojo kernels branch on.

`stellargraph` holds an activation as a Keras object; the C ABI cannot carry
one, so the string upstream spells stays the public API and the code is an
implementation detail of the boundary.
"""

from __future__ import annotations

import numpy as np

ACT_LINEAR = 0
ACT_RELU = 1
ACT_ELU = 2
ACT_SOFTMAX = 3
ACT_SIGMOID = 4
ACT_TANH = 5
ACT_SOFTPLUS = 6
ACT_LEAKY_RELU = 7
ACT_HARD_SIGMOID = 8
ACT_EXP = 9
ACT_SOFT_SIGN = 10

# Every name `keras.activations.get` accepts, as documented upstream. `None`
# is the identity and is what `GraphConvolution(activation=None)` means.
CODES = {
    "linear": ACT_LINEAR,
    None: ACT_LINEAR,
    "relu": ACT_RELU,
    "elu": ACT_ELU,
    "softmax": ACT_SOFTMAX,
    "sigmoid": ACT_SIGMOID,
    "tanh": ACT_TANH,
    "softplus": ACT_SOFTPLUS,
    "softsign": ACT_SOFT_SIGN,
    "leaky_relu": ACT_LEAKY_RELU,
    "hard_sigmoid": ACT_HARD_SIGMOID,
    "exponential": ACT_EXP,
}


def code(activation) -> int:
    """The kernel code for an upstream activation name."""
    try:
        return CODES[activation]
    except KeyError:
        raise ValueError(
            "unsupported activation {!r}; stellargraph passes this to "
            "keras.activations.get".format(activation)
        ) from None


#: `keras.layers.LeakyReLU`'s default negative slope, which is the `alpha`
#: every layer passes when upstream writes `LeakyReLU(alpha=0.2)` nowhere.
LEAKY_RELU_ALPHA = 0.01


def alpha(activation) -> float:
    """The `alpha` a Keras activation is built with; 0.0 for the rest."""
    return LEAKY_RELU_ALPHA if activation == "leaky_relu" else 0.0


def numpy_activation(activation):
    """A NumPy version of the same activation, for the reference path and for
    places where a scalar loop would be wasteful. `None` is the identity."""
    if activation in (None, "linear"):
        return lambda x: x
    if activation == "relu":
        return lambda x: np.maximum(x, 0.0)
    if activation == "elu":
        return lambda x: np.where(x >= 0, x, np.expm1(x))
    if activation == "softmax":
        return lambda x: _softmax_last_axis(np.atleast_2d(x))
    if activation == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if activation == "tanh":
        return np.tanh
    if activation == "softplus":
        return lambda x: np.log1p(np.exp(x))
    if activation == "leaky_relu":
        return lambda x: np.where(x >= 0, x, 0.01 * x)
    if activation == "hard_sigmoid":
        return lambda x: np.clip(0.2 * x + 0.5, 0.0, 1.0)
    if activation == "softsign":
        return lambda x: x / (1.0 + np.abs(x))
    if activation == "exponential":
        return np.exp
    raise ValueError("unsupported activation {!r}".format(activation))


def _softmax_last_axis(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)
