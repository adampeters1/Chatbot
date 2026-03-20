"""
Tests for model/losses.py

Run from the project root with:
    python -m model.losses_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from model.activations import softmax, softmax_jacobian
from model.losses import (
    categorical_cross_entropy,
    categorical_cross_entropy_derivative,
    softmax_cross_entropy_gradient,
    mean_categorical_cross_entropy,
    _EPSILON,
)


# ─────────────────────────────────────────────
# Tiny harness
# ─────────────────────────────────────────────

PASS = "  PASS"
FAIL = "  FAIL"
_failures = []


def _almost_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def _list_almost_equal(l1, l2, tol=1e-9):
    if len(l1) != len(l2):
        return False
    return all(_almost_equal(a, b, tol) for a, b in zip(l1, l2))


def _run(name, passed):
    status = PASS if passed else FAIL
    print(f"{status}  {name}")
    if not passed:
        _failures.append(name)


def _expect_error(func, args, exc_type, name):
    try:
        func(*args)
        _run(name, False)
    except exc_type:
        _run(name, True)
    except Exception as e:
        print(f"  FAIL  {name}  (unexpected exception: {e})")
        _failures.append(name)


# ─────────────────────────────────────────────
# categorical_cross_entropy (single sample)
# ─────────────────────────────────────────────

def test_cce_single():
    print("\n--- categorical_cross_entropy ---")

    # perfect confident correct prediction  →  loss = −log(1−ε) ≈ 0
    y_true = [0.0, 1.0, 0.0]
    y_pred = [0.0, 1.0, 0.0]
    loss = categorical_cross_entropy(y_true, y_pred)
    _run("cce perfect prediction ≈ 0",
         _almost_equal(loss, -math.log(1.0 - _EPSILON)))

    # confident wrong prediction  →  loss = −log(ε)
    y_pred_wrong = [1.0, 0.0, 0.0]
    loss_wrong = categorical_cross_entropy(y_true, y_pred_wrong)
    _run("cce confident wrong == -log(ε)",
         _almost_equal(loss_wrong, -math.log(_EPSILON)))

    # uniform prediction  →  −log(1/C)
    C = 4
    y_true_u = [0.0, 0.0, 1.0, 0.0]
    y_pred_u = [1.0 / C] * C
    loss_u = categorical_cross_entropy(y_true_u, y_pred_u)
    _run("cce uniform prediction == -log(1/C)",
         _almost_equal(loss_u, -math.log(1.0 / C)))

    # arbitrary distribution, cross-check with numpy
    y_true_a = [0.0, 1.0, 0.0, 0.0]
    y_pred_a = [0.1, 0.7, 0.15, 0.05]
    expected = -np.sum(np.array(y_true_a) * np.log(np.array(y_pred_a)))
    _run("cce arbitrary distribution matches numpy",
         _almost_equal(categorical_cross_entropy(y_true_a, y_pred_a),
                       float(expected)))

    # prediction containing a zero on the true class → clipping kicks in,
    # result must be finite
    y_true_z = [1.0, 0.0]
    y_pred_z = [0.0, 1.0]
    loss_z = categorical_cross_entropy(y_true_z, y_pred_z)
    _run("cce log(0) clipped → finite",
         math.isfinite(loss_z) and loss_z > 0.0)

    # non-negative for any inputs
    _run("cce perfect prediction non-negative", loss >= 0.0)

    # shape mismatch
    _expect_error(categorical_cross_entropy,
                  [[0.0, 1.0], [0.3, 0.3, 0.4]],
                  ValueError, "cce length mismatch")
    _expect_error(categorical_cross_entropy,
                  ["bad", [0.3, 0.7]],
                  TypeError, "cce wrong type y_true")
    _expect_error(categorical_cross_entropy,
                  [[0.0, 1.0], "bad"],
                  TypeError, "cce wrong type y_pred")
    _expect_error(categorical_cross_entropy,
                  [[], []],
                  ValueError, "cce empty vectors")


# ─────────────────────────────────────────────
# categorical_cross_entropy_derivative
# ─────────────────────────────────────────────

def test_cce_derivative():
    print("\n--- categorical_cross_entropy_derivative ---")

    y_true = [0.0, 1.0, 0.0]
    y_pred = [0.2, 0.5, 0.3]
    grad = categorical_cross_entropy_derivative(y_true, y_pred)
    expected = [0.0, -1.0 / 0.5, 0.0]
    _run("cce_deriv basic one-hot",
         _list_almost_equal(grad, expected))

    # gradient at true class is negative, others are zero for one-hot
    _run("cce_deriv sign at true class negative", grad[1] < 0.0)
    _run("cce_deriv zero at non-true classes",
         _almost_equal(grad[0], 0.0) and _almost_equal(grad[2], 0.0))

    # zero prediction on the true class → clipping prevents div-by-zero
    y_pred_z = [0.0, 0.5, 0.5]
    grad_z = categorical_cross_entropy_derivative([1.0, 0.0, 0.0], y_pred_z)
    _run("cce_deriv clipped when p==0",
         math.isfinite(grad_z[0]) and grad_z[0] < 0.0)

    _expect_error(categorical_cross_entropy_derivative,
                  [[0.0, 1.0], [0.3, 0.3, 0.4]],
                  ValueError, "cce_deriv length mismatch")
    _expect_error(categorical_cross_entropy_derivative,
                  ["bad", [0.3, 0.7]],
                  TypeError, "cce_deriv wrong type y_true")
    _expect_error(categorical_cross_entropy_derivative,
                  [[0.0, 1.0], "bad"],
                  TypeError, "cce_deriv wrong type y_pred")


# ─────────────────────────────────────────────
# softmax_cross_entropy_gradient
# ─────────────────────────────────────────────

def test_softmax_cce_gradient():
    print("\n--- softmax_cross_entropy_gradient ---")

    y_true = [0.0, 1.0, 0.0]
    y_pred = [0.2, 0.5, 0.3]
    grad = softmax_cross_entropy_gradient(y_true, y_pred)
    expected = [0.2, -0.5, 0.3]
    _run("softmax_cce_grad basic",
         _list_almost_equal(grad, expected))

    # perfect prediction → zero gradient (up to ε from clipping elsewhere)
    y_pred_perfect = [0.0, 1.0, 0.0]
    grad_perfect = softmax_cross_entropy_gradient(y_true, y_pred_perfect)
    _run("softmax_cce_grad perfect prediction == 0",
         _list_almost_equal(grad_perfect, [0.0, 0.0, 0.0]))

    # gradient sums to zero when y_pred sums to 1 and y_true is one-hot
    _run("softmax_cce_grad sums to zero",
         _almost_equal(sum(grad), 0.0))

    # ── cross-check: combined gradient == Jᵀ · dL/dp ──
    # Jᵀ · (−y_true / p)  should equal  y_pred − y_true
    logits = [0.3, -1.0, 2.5, 0.1]
    p = softmax(logits)
    y_t = [0.0, 0.0, 1.0, 0.0]

    dL_dp = categorical_cross_entropy_derivative(y_t, p)
    J = softmax_jacobian(p)

    # manual Jᵀ · v  (4×4 · 4)
    combined = [
        sum(J[k][i] * dL_dp[k] for k in range(len(p)))
        for i in range(len(p))
    ]
    closed_form = softmax_cross_entropy_gradient(y_t, p)
    _run("softmax_cce_grad matches Jᵀ·dL/dp chain",
         _list_almost_equal(combined, closed_form, tol=1e-8))

    # ── finite-difference gradient check on the logits ──
    def loss_at(z):
        return categorical_cross_entropy(y_t, softmax(z))

    h = 1e-6
    fd_grad = []
    for i in range(len(logits)):
        z_plus  = list(logits); z_plus[i]  += h
        z_minus = list(logits); z_minus[i] -= h
        fd_grad.append((loss_at(z_plus) - loss_at(z_minus)) / (2 * h))

    _run("softmax_cce_grad matches finite diff",
         _list_almost_equal(closed_form, fd_grad, tol=1e-5))

    _expect_error(softmax_cross_entropy_gradient,
                  [[0.0, 1.0], [0.3, 0.3, 0.4]],
                  ValueError, "softmax_cce_grad length mismatch")
    _expect_error(softmax_cross_entropy_gradient,
                  ["bad", [0.3, 0.7]],
                  TypeError, "softmax_cce_grad wrong type y_true")
    _expect_error(softmax_cross_entropy_gradient,
                  [[0.0, 1.0], "bad"],
                  TypeError, "softmax_cce_grad wrong type y_pred")


# ─────────────────────────────────────────────
# mean_categorical_cross_entropy
# ─────────────────────────────────────────────

def test_mean_cce():
    print("\n--- mean_categorical_cross_entropy ---")

    y_true_batch = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    y_pred_batch = [
        [0.1, 0.7, 0.2],
        [0.8, 0.1, 0.1],
        [0.05, 0.05, 0.9],
    ]
    individual = [
        categorical_cross_entropy(yt, yp)
        for yt, yp in zip(y_true_batch, y_pred_batch)
    ]
    expected = sum(individual) / len(individual)
    result = mean_categorical_cross_entropy(y_true_batch, y_pred_batch)
    _run("mean_cce equals average of per-sample losses",
         _almost_equal(result, expected))

    # batch of one == single-sample loss
    single = mean_categorical_cross_entropy(
        [y_true_batch[0]], [y_pred_batch[0]]
    )
    _run("mean_cce batch-of-one matches single",
         _almost_equal(single, individual[0]))

    # numpy cross-check
    yt_np = np.array(y_true_batch)
    yp_np = np.clip(np.array(y_pred_batch), _EPSILON, 1.0 - _EPSILON)
    expected_np = float((-(yt_np * np.log(yp_np)).sum(axis=1)).mean())
    _run("mean_cce vs numpy",
         _almost_equal(result, expected_np))

    # finite even with zeros in predictions
    y_pred_batch_z = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result_z = mean_categorical_cross_entropy(y_true_batch, y_pred_batch_z)
    _run("mean_cce finite when predictions contain zeros",
         math.isfinite(result_z))

    # errors
    _expect_error(mean_categorical_cross_entropy,
                  [[], []], ValueError, "mean_cce empty batch")
    _expect_error(mean_categorical_cross_entropy,
                  [y_true_batch, y_pred_batch[:2]],
                  ValueError, "mean_cce batch size mismatch")
    _expect_error(mean_categorical_cross_entropy,
                  ["bad", y_pred_batch], TypeError,
                  "mean_cce wrong type y_true_batch")
    _expect_error(mean_categorical_cross_entropy,
                  [y_true_batch, "bad"], TypeError,
                  "mean_cce wrong type y_pred_batch")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  losses.py – full test suite")
    print("=" * 55)

    test_cce_single()
    test_cce_derivative()
    test_softmax_cce_gradient()
    test_mean_cce()

    print("\n" + "=" * 55)
    if _failures:
        print(f"  {len(_failures)} test(s) FAILED:")
        for f in _failures:
            print(f"    ✗  {f}")
    else:
        print("  All tests passed.")
    print("=" * 55)


if __name__ == "__main__":
    run_all_tests()