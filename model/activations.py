"""
Activation functions (forward + derivative) for the neural network.

All functions operate element-wise on plain Python lists (1-D vectors).
No external dependencies beyond the standard `math` module.
"""

import math


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _ensure_vector(x, name: str = "x") -> None:
    """Raise if `x` is not a non-empty list of numbers."""
    if not isinstance(x, list):
        raise TypeError(
            f"{name} must be a list, got {type(x).__name__}."
        )
    if len(x) == 0:
        raise ValueError(
            f"{name} is an empty list; activation is undefined."
        )


# ─────────────────────────────────────────────
# ReLU
# ─────────────────────────────────────────────

def relu(x: list) -> list:
    """
    Rectified Linear Unit — element-wise max(0, x).

    Parameters
    ----------
    x : list
        Pre-activation values (1-D vector).

    Returns
    -------
    list
        Element-wise ReLU output, same length as `x`.
    """
    _ensure_vector(x, "x")
    return [xi if xi > 0.0 else 0.0 for xi in x]


def relu_derivative(x: list) -> list:
    """
    Derivative of ReLU with respect to its input.

    dReLU/dx = 1 if x > 0 else 0.
    (Convention: derivative at x == 0 is 0.)

    Parameters
    ----------
    x : list
        The *pre-activation* values that were fed into `relu` during
        the forward pass.  Using the pre-activation vector rather than
        the post-activation one is customary because ReLU(0) == 0 and
        we want the sub-gradient at the kink explicitly.

    Returns
    -------
    list
        Element-wise derivative (1.0 or 0.0), same length as `x`.
    """
    _ensure_vector(x, "x")
    return [1.0 if xi > 0.0 else 0.0 for xi in x]


# ─────────────────────────────────────────────
# Sigmoid
# ─────────────────────────────────────────────

def sigmoid(x: list) -> list:
    """
    Logistic sigmoid — 1 / (1 + exp(-x)), applied element-wise.

    A numerically stable branch is used for large negative inputs to
    avoid `OverflowError` from `math.exp`.

    Parameters
    ----------
    x : list
        Pre-activation values (1-D vector).

    Returns
    -------
    list
        Element-wise sigmoid output in the open interval (0, 1).
    """
    _ensure_vector(x, "x")
    out = []
    for xi in x:
        if xi >= 0.0:
            # Standard form — exp(-xi) is safe for xi >= 0
            out.append(1.0 / (1.0 + math.exp(-xi)))
        else:
            # Stable form for negative xi:
            #   sigmoid(x) = exp(x) / (1 + exp(x))
            e = math.exp(xi)  # xi < 0 → e in (0, 1], no overflow
            out.append(e / (1.0 + e))
    return out


def sigmoid_derivative(x: list) -> list:
    """
    Derivative of sigmoid with respect to its input.

    d(sigmoid)/dx = sigmoid(x) · (1 − sigmoid(x)).

    Parameters
    ----------
    x : list
        The *pre-activation* values that were fed into `sigmoid`
        during the forward pass.

    Returns
    -------
    list
        Element-wise derivative, same length as `x`.
    """
    _ensure_vector(x, "x")
    s = sigmoid(x)
    return [si * (1.0 - si) for si in s]


# ─────────────────────────────────────────────
# Softmax
# ─────────────────────────────────────────────

def softmax(x: list) -> list:
    """
    Softmax — exp(xi) / Σ exp(xj).

    Implements the standard numerical-stability trick of subtracting
    max(x) from every element before exponentiating.  This shifts the
    input so the largest value becomes 0, guaranteeing `math.exp`
    never overflows while leaving the result unchanged.

    Parameters
    ----------
    x : list
        Pre-activation logits (1-D vector).

    Returns
    -------
    list
        Probability distribution (sums to 1.0, each entry in [0, 1]).
    """
    _ensure_vector(x, "x")
    x_max = max(x)
    exps = [math.exp(xi - x_max) for xi in x]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_jacobian(p: list) -> list:
    """
    Full Jacobian of softmax with respect to its input.

    Given softmax output `p` of length n, the Jacobian J is an n×n
    matrix where:

        J[i][j] = p[i] · (δ_ij − p[j])
                = p[i] · (1 − p[i])   if i == j
                = −p[i] · p[j]        if i != j

    This is rarely needed in practice because the combined
    softmax + cross-entropy gradient collapses to (y_pred − y_true),
    but it is included for completeness / experimentation.

    Parameters
    ----------
    p : list
        The *softmax output* (probability vector), not the logits.

    Returns
    -------
    list
        n × n Jacobian as a list of lists.
    """
    _ensure_vector(p, "p")
    n = len(p)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = p[i] * (1.0 - p[i])
            else:
                J[i][j] = -p[i] * p[j]
    return J


# ─────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────

# Useful later when wiring layers dynamically from a config string.
ACTIVATIONS = {
    "relu":    {"forward": relu,    "derivative": relu_derivative},
    "sigmoid": {"forward": sigmoid, "derivative": sigmoid_derivative},
    "softmax": {"forward": softmax, "derivative": softmax_jacobian},
}