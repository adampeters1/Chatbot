"""
Loss functions for multi-class classification.

Categorical cross-entropy is the workhorse here.  The combined
softmax + cross-entropy gradient is provided as a dedicated helper
because that simplified form (y_pred − y_true) is what backprop
actually uses.
"""

import math


# Small constant to keep log() away from zero.
_EPSILON = 1e-15


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _ensure_vector(x, name: str = "x") -> None:
    """Raise if `x` is not a non-empty list."""
    if not isinstance(x, list):
        raise TypeError(
            f"{name} must be a list, got {type(x).__name__}."
        )
    if len(x) == 0:
        raise ValueError(
            f"{name} is an empty list."
        )


def _check_pair(y_true: list, y_pred: list) -> None:
    """Shared shape / type checks for a (target, prediction) pair."""
    _ensure_vector(y_true, "y_true")
    _ensure_vector(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Shape mismatch: y_true has length "
            f"{len(y_true)} but y_pred has length {len(y_pred)}."
        )


def _clip(p: float, lo: float = _EPSILON, hi: float = 1.0 - _EPSILON) -> float:
    """Clamp a probability into [lo, hi] to keep log well-defined."""
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p


# ─────────────────────────────────────────────
# Categorical cross-entropy — single sample
# ─────────────────────────────────────────────

def categorical_cross_entropy(y_true: list, y_pred: list) -> float:
    """
    Categorical cross-entropy for a single sample.

        L = − Σ_i  y_true[i] · log( y_pred[i] )

    `y_true` is expected to be a one-hot (or soft-label) vector and
    `y_pred` a probability distribution (softmax output).  Predictions
    are clipped into [ε, 1 − ε] so log(0) never occurs.

    Parameters
    ----------
    y_true : list
        Target vector of length C (one-hot or soft labels).
    y_pred : list
        Predicted probability vector of length C.

    Returns
    -------
    float
        Scalar loss for this sample.
    """
    _check_pair(y_true, y_pred)
    loss = 0.0
    for t, p in zip(y_true, y_pred):
        p_clipped = _clip(p)
        loss -= t * math.log(p_clipped)
    return loss


def categorical_cross_entropy_derivative(y_true: list, y_pred: list) -> list:
    """
    Gradient of categorical cross-entropy with respect to y_pred.

        dL/dp_i = − y_true[i] / y_pred[i]

    Predictions are clipped into [ε, 1 − ε] before division so we
    never divide by zero.

    This is the *raw* cross-entropy gradient.  When the preceding
    layer is softmax, prefer `softmax_cross_entropy_gradient` — the
    combined closed-form (y_pred − y_true) — because it is both
    simpler and numerically stabler than chaining this gradient
    through the softmax Jacobian.

    Parameters
    ----------
    y_true : list
        Target vector.
    y_pred : list
        Predicted probability vector.

    Returns
    -------
    list
        Gradient vector, same length as `y_pred`.
    """
    _check_pair(y_true, y_pred)
    grad = []
    for t, p in zip(y_true, y_pred):
        p_clipped = _clip(p)
        grad.append(-t / p_clipped)
    return grad


# ─────────────────────────────────────────────
# Combined softmax + cross-entropy gradient
# ─────────────────────────────────────────────

def softmax_cross_entropy_gradient(y_true: list, y_pred: list) -> list:
    """
    Combined gradient of (softmax → cross-entropy) w.r.t. the
    pre-softmax logits.

    When the output layer uses softmax and the loss is categorical
    cross-entropy, the Jacobian-vector product collapses to the
    elegant closed form:

        dL/dz_i  =  y_pred[i] − y_true[i]

    This is the expression used for backpropagation from the output
    layer.  It avoids ever materialising the full softmax Jacobian.

    Parameters
    ----------
    y_true : list
        One-hot (or soft) target vector, length C.
    y_pred : list
        Softmax output (probability vector), length C.

    Returns
    -------
    list
        Gradient w.r.t. the logits z, same length as inputs.
    """
    _check_pair(y_true, y_pred)
    return [p - t for t, p in zip(y_true, y_pred)]


# ─────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────

def mean_categorical_cross_entropy(
    y_true_batch: list,
    y_pred_batch: list,
) -> float:
    """
    Average categorical cross-entropy over a batch of samples.

    Parameters
    ----------
    y_true_batch : list
        List of B target vectors (each length C).
    y_pred_batch : list
        List of B prediction vectors (each length C).

    Returns
    -------
    float
        Mean loss across the batch.
    """
    if not isinstance(y_true_batch, list) or not isinstance(y_pred_batch, list):
        raise TypeError(
            "Batch arguments must be lists of vectors; got "
            f"{type(y_true_batch).__name__} and "
            f"{type(y_pred_batch).__name__}."
        )
    if len(y_true_batch) == 0:
        raise ValueError("mean_categorical_cross_entropy received an empty batch.")
    if len(y_true_batch) != len(y_pred_batch):
        raise ValueError(
            "Batch size mismatch: "
            f"y_true_batch has {len(y_true_batch)} samples "
            f"but y_pred_batch has {len(y_pred_batch)}."
        )

    total = 0.0
    for yt, yp in zip(y_true_batch, y_pred_batch):
        total += categorical_cross_entropy(yt, yp)
    return total / len(y_true_batch)