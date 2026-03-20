"""
Tests for model/activations.py

Run from the project root with:
    python -m model.activations_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from model.activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    softmax_jacobian,
    ACTIVATIONS,
)


# ─────────────────────────────────────────────
# Tiny harness (same style as matrix_ops_test)
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


def _matrix_almost_equal(m1, m2, tol=1e-9):
    if len(m1) != len(m2):
        return False
    return all(_list_almost_equal(r1, r2, tol) for r1, r2 in zip(m1, m2))


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
# ReLU
# ─────────────────────────────────────────────

def test_relu():
    print("\n--- relu ---")

    _run("relu mixed signs",
         relu([-2.0, -0.5, 0.0, 0.5, 2.0]) == [0.0, 0.0, 0.0, 0.5, 2.0])
    _run("relu all positive",
         relu([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0])
    _run("relu all negative",
         relu([-1.0, -2.0, -3.0]) == [0.0, 0.0, 0.0])
    _run("relu single zero", relu([0.0]) == [0.0])
    _run("relu single positive", relu([42.0]) == [42.0])

    # numpy cross-check
    rng = np.random.default_rng(0)
    x = rng.standard_normal(25).tolist()
    expected = np.maximum(0.0, np.array(x)).tolist()
    _run("relu vs numpy random",
         _list_almost_equal(relu(x), expected))

    _expect_error(relu, ["bad"], TypeError, "relu wrong type")
    _expect_error(relu, [[]], ValueError, "relu empty")


def test_relu_derivative():
    print("\n--- relu_derivative ---")

    _run("relu_derivative mixed signs",
         relu_derivative([-2.0, -0.5, 0.0, 0.5, 2.0]) == [0.0, 0.0, 0.0, 1.0, 1.0])
    _run("relu_derivative at zero is zero",
         relu_derivative([0.0]) == [0.0])
    _run("relu_derivative all positive",
         relu_derivative([1.0, 2.0]) == [1.0, 1.0])
    _run("relu_derivative all negative",
         relu_derivative([-1.0, -2.0]) == [0.0, 0.0])

    # finite-difference sanity check at a point away from the kink
    h = 1e-6
    x = [0.7]
    numeric = (relu([x[0] + h])[0] - relu([x[0] - h])[0]) / (2 * h)
    analytic = relu_derivative(x)[0]
    _run("relu_derivative matches finite diff at x=0.7",
         _almost_equal(numeric, analytic, tol=1e-4))

    _expect_error(relu_derivative, ["bad"], TypeError, "relu_derivative wrong type")
    _expect_error(relu_derivative, [[]], ValueError, "relu_derivative empty")


# ─────────────────────────────────────────────
# Sigmoid
# ─────────────────────────────────────────────

def test_sigmoid():
    print("\n--- sigmoid ---")

    _run("sigmoid(0) == 0.5", _almost_equal(sigmoid([0.0])[0], 0.5))

    result = sigmoid([-2.0, 0.0, 2.0])
    expected = [1.0 / (1.0 + math.exp(2.0)),
                0.5,
                1.0 / (1.0 + math.exp(-2.0))]
    _run("sigmoid basic values",
         _list_almost_equal(result, expected))

    # symmetry: sigmoid(-x) == 1 - sigmoid(x)
    s = sigmoid([3.0])[0]
    sn = sigmoid([-3.0])[0]
    _run("sigmoid symmetry", _almost_equal(sn, 1.0 - s))

    # large-|x| saturation without overflow
    big = sigmoid([1000.0, -1000.0])
    _run("sigmoid(1000) ≈ 1",  _almost_equal(big[0], 1.0, tol=1e-12))
    _run("sigmoid(-1000) ≈ 0", _almost_equal(big[1], 0.0, tol=1e-12))

    # numpy cross-check on random vector
    rng = np.random.default_rng(1)
    x = rng.standard_normal(25).tolist()
    expected = (1.0 / (1.0 + np.exp(-np.array(x)))).tolist()
    _run("sigmoid vs numpy random",
         _list_almost_equal(sigmoid(x), expected))

    # output range
    out = sigmoid([-5.0, -1.0, 0.0, 1.0, 5.0])
    _run("sigmoid output in (0, 1)",
         all(0.0 < o < 1.0 for o in out))

    _expect_error(sigmoid, ["bad"], TypeError, "sigmoid wrong type")
    _expect_error(sigmoid, [[]], ValueError, "sigmoid empty")


def test_sigmoid_derivative():
    print("\n--- sigmoid_derivative ---")

    _run("sigmoid_derivative(0) == 0.25",
         _almost_equal(sigmoid_derivative([0.0])[0], 0.25))

    d = sigmoid_derivative([2.0])[0]
    s = sigmoid([2.0])[0]
    _run("sigmoid_derivative matches s·(1−s)",
         _almost_equal(d, s * (1.0 - s)))

    # finite-difference check on a batch of points
    h = 1e-6
    xs = [-1.5, -0.3, 0.2, 1.1, 2.4]
    analytic = sigmoid_derivative(xs)
    fd_ok = True
    for i, xi in enumerate(xs):
        num = (sigmoid([xi + h])[0] - sigmoid([xi - h])[0]) / (2 * h)
        if not _almost_equal(num, analytic[i], tol=1e-4):
            fd_ok = False
            break
    _run("sigmoid_derivative matches finite diff", fd_ok)

    # numpy cross-check
    rng = np.random.default_rng(2)
    x = rng.standard_normal(25).tolist()
    s_np = 1.0 / (1.0 + np.exp(-np.array(x)))
    expected = (s_np * (1.0 - s_np)).tolist()
    _run("sigmoid_derivative vs numpy random",
         _list_almost_equal(sigmoid_derivative(x), expected))

    _expect_error(sigmoid_derivative, ["bad"], TypeError,
                  "sigmoid_derivative wrong type")
    _expect_error(sigmoid_derivative, [[]], ValueError,
                  "sigmoid_derivative empty")


# ─────────────────────────────────────────────
# Softmax
# ─────────────────────────────────────────────

def test_softmax():
    print("\n--- softmax ---")

    out = softmax([1.0, 2.0, 3.0])
    _run("softmax sums to 1", _almost_equal(sum(out), 1.0))
    _run("softmax positive",  all(p > 0.0 for p in out))
    _run("softmax monotone",
         out[0] < out[1] < out[2])

    # uniform input → uniform output
    u = softmax([5.0, 5.0, 5.0, 5.0])
    _run("softmax uniform input → uniform output",
         _list_almost_equal(u, [0.25, 0.25, 0.25, 0.25]))

    # shift invariance: softmax(x + c) == softmax(x)
    a = softmax([1.0, 2.0, 3.0])
    b = softmax([101.0, 102.0, 103.0])
    _run("softmax shift invariance",
         _list_almost_equal(a, b))

    # numerical stability — large logits that would overflow raw exp()
    big = softmax([1000.0, 1001.0, 1002.0])
    _run("softmax large logits sums to 1",
         _almost_equal(sum(big), 1.0))
    _run("softmax large logits finite",
         all(math.isfinite(p) for p in big))

    # singleton → [1.0]
    _run("softmax single element", _list_almost_equal(softmax([42.0]), [1.0]))

    # numpy cross-check
    rng = np.random.default_rng(3)
    x = rng.standard_normal(12).tolist()
    x_np = np.array(x)
    e = np.exp(x_np - x_np.max())
    expected = (e / e.sum()).tolist()
    _run("softmax vs numpy random",
         _list_almost_equal(softmax(x), expected))

    _expect_error(softmax, ["bad"], TypeError, "softmax wrong type")
    _expect_error(softmax, [[]], ValueError, "softmax empty")


def test_softmax_jacobian():
    print("\n--- softmax_jacobian ---")

    p = softmax([1.0, 2.0, 3.0])
    J = softmax_jacobian(p)

    # shape check
    shape_ok = (len(J) == 3) and all(len(row) == 3 for row in J)
    _run("softmax_jacobian shape 3x3", shape_ok)

    # diagonal: p_i * (1 - p_i)
    diag_ok = all(
        _almost_equal(J[i][i], p[i] * (1.0 - p[i])) for i in range(3)
    )
    _run("softmax_jacobian diagonal correct", diag_ok)

    # off-diagonal: -p_i * p_j
    offd_ok = True
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if not _almost_equal(J[i][j], -p[i] * p[j]):
                offd_ok = False
    _run("softmax_jacobian off-diagonal correct", offd_ok)

    # each row sums to 0 because Σ_j d p_i / d z_j = 0
    rows_sum_zero = all(_almost_equal(sum(row), 0.0) for row in J)
    _run("softmax_jacobian rows sum to zero", rows_sum_zero)

    # numpy cross-check on random vector
    rng = np.random.default_rng(4)
    x = rng.standard_normal(6).tolist()
    p2 = softmax(x)
    p_np = np.array(p2)
    expected = (np.diag(p_np) - np.outer(p_np, p_np)).tolist()
    _run("softmax_jacobian vs numpy",
         _matrix_almost_equal(softmax_jacobian(p2), expected))

    _expect_error(softmax_jacobian, ["bad"], TypeError,
                  "softmax_jacobian wrong type")
    _expect_error(softmax_jacobian, [[]], ValueError,
                  "softmax_jacobian empty")


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

def test_registry():
    print("\n--- ACTIVATIONS registry ---")

    _run("registry has relu",    "relu" in ACTIVATIONS)
    _run("registry has sigmoid", "sigmoid" in ACTIVATIONS)
    _run("registry has softmax", "softmax" in ACTIVATIONS)
    _run("registry relu forward matches",
         ACTIVATIONS["relu"]["forward"]([-1.0, 1.0]) == [0.0, 1.0])
    _run("registry sigmoid deriv matches",
         _almost_equal(
             ACTIVATIONS["sigmoid"]["derivative"]([0.0])[0], 0.25
         ))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  activations.py – full test suite")
    print("=" * 55)

    test_relu()
    test_relu_derivative()
    test_sigmoid()
    test_sigmoid_derivative()
    test_softmax()
    test_softmax_jacobian()
    test_registry()

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