"""
Lightweight sanity tests for model/layers.py

Run from the project root with:
    python -m model.layers_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.layers import DenseLayer
from model.activations import softmax, relu
from model.losses import (
    categorical_cross_entropy,
    softmax_cross_entropy_gradient,
)
from utils.matrix_ops import (
    matrix_vector_multiply,
    vector_add,
)


PASS = "  PASS"
FAIL = "  FAIL"
_failures = []


def _almost_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def _run(name, passed):
    status = PASS if passed else FAIL
    print(f"{status}  {name}")
    if not passed:
        _failures.append(name)


# ─────────────────────────────────────────────
# Construction / shapes
# ─────────────────────────────────────────────

def test_construction():
    print("\n--- construction & shapes ---")

    layer = DenseLayer(input_size=5, output_size=3,
                       activation="relu", seed=42)

    _run("W rows == output_size", len(layer.W) == 3)
    _run("W cols == input_size",  all(len(row) == 5 for row in layer.W))
    _run("b length == output_size", len(layer.b) == 3)
    _run("b initialised to zeros", all(v == 0.0 for v in layer.b))
    _run("activation wired",       layer.activation_name == "relu")

    try:
        DenseLayer(5, 3, activation="nope")
        _run("unknown activation raises", False)
    except ValueError:
        _run("unknown activation raises", True)


# ─────────────────────────────────────────────
# Forward pass correctness (ReLU)
# ─────────────────────────────────────────────

def test_forward_relu():
    print("\n--- forward (ReLU) ---")

    layer = DenseLayer(input_size=4, output_size=3,
                       activation="relu", seed=0)
    x = [0.5, -1.0, 2.0, 0.1]
    a = layer.forward(x)

    # manual recompute
    z_manual = vector_add(matrix_vector_multiply(layer.W, x), layer.b)
    a_manual = relu(z_manual)

    _run("forward caches input", layer._input == x)
    _run("forward caches z",     all(
        _almost_equal(zi, mi) for zi, mi in zip(layer._z, z_manual)))
    _run("forward output == manual ReLU(W·x+b)", all(
        _almost_equal(ai, mi) for ai, mi in zip(a, a_manual)))
    _run("forward output length == output_size", len(a) == 3)

# ─────────────────────────────────────────────
# Forward pass correctness (softmax)
# ─────────────────────────────────────────────

def test_forward_softmax():
    print("\n--- forward (softmax) ---")

    layer = DenseLayer(input_size=4, output_size=3,
                       activation="softmax", seed=1)
    x = [0.2, 0.4, -0.3, 1.1]
    a = layer.forward(x)

    _run("softmax output sums to 1",
         _almost_equal(sum(a), 1.0))
    _run("softmax output all positive",
         all(p > 0.0 for p in a))


# ─────────────────────────────────────────────
# Backward updates parameters
# ─────────────────────────────────────────────

def test_backward_updates():
    print("\n--- backward updates W & b ---")

    layer = DenseLayer(input_size=4, output_size=3,
                       activation="relu", seed=123)
    x = [1.0, 0.5, -0.3, 0.2]
    layer.forward(x)

    W_before_flat = [w for row in layer.W for w in row]
    b_before      = list(layer.b)

    upstream = [0.3, -0.1, 0.2]
    dx = layer.backward(upstream, learning_rate=0.1)

    W_after_flat = [w for row in layer.W for w in row]

    _run("dx has input_size length", len(dx) == 4)
    _run("W changed after update",
         any(not _almost_equal(a, b)
             for a, b in zip(W_before_flat, W_after_flat)))
    _run("b changed after update",
         any(not _almost_equal(a, b)
             for a, b in zip(b_before, layer.b)))


# ─────────────────────────────────────────────
# Training a 2-layer net reduces loss
# ─────────────────────────────────────────────

def test_two_layer_learns():
    print("\n--- 2-layer net reduces loss ---")

    in_dim, hid_dim, out_dim = 4, 6, 3
    hidden = DenseLayer(in_dim,  hid_dim, activation="relu",    seed=10)
    output = DenseLayer(hid_dim, out_dim, activation="softmax", seed=11)

    # tiny fixed batch
    X = [
        [0.5, -0.2, 0.1, 0.3],
        [0.9,  0.4, -0.7, 0.2],
        [-0.3, 0.8, 0.5, -0.1],
    ]
    Y = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    lr = 0.5
    epochs = 80

    def epoch_loss():
        total = 0.0
        for x, y in zip(X, Y):
            h = hidden.forward(x)
            p = output.forward(h)
            total += categorical_cross_entropy(y, p)
        return total / len(X)

    initial_loss = epoch_loss()

    for _ in range(epochs):
        for x, y in zip(X, Y):
            h = hidden.forward(x)
            p = output.forward(h)
            # combined softmax+CCE gradient == dL/dz at output layer
            g_out = softmax_cross_entropy_gradient(y, p)
            g_hid = output.backward(g_out, lr)
            hidden.backward(g_hid, lr)

    final_loss = epoch_loss()

    _run("loss strictly decreased", final_loss < initial_loss)
    _run("loss decreased meaningfully (>50%)",
         final_loss < 0.5 * initial_loss)
    print(f"       initial loss = {initial_loss:.4f}, "
          f"final loss = {final_loss:.4f}")


# ─────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────

def test_errors():
    print("\n--- error handling ---")

    layer = DenseLayer(3, 2, activation="relu", seed=0)

    try:
        layer.backward([0.1, 0.2], 0.1)
        _run("backward before forward raises", False)
    except RuntimeError:
        _run("backward before forward raises", True)

    try:
        layer.forward([1.0, 2.0])  # wrong length
        _run("forward wrong input length raises", False)
    except ValueError:
        _run("forward wrong input length raises", True)

    layer.forward([1.0, 2.0, 3.0])
    try:
        layer.backward([0.1, 0.2, 0.3], 0.1)  # wrong length
        _run("backward wrong grad length raises", False)
    except ValueError:
        _run("backward wrong grad length raises", True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  layers.py – sanity test suite")
    print("=" * 55)

    test_construction()
    test_forward_relu()
    test_forward_softmax()
    test_backward_updates()
    test_two_layer_learns()
    test_errors()

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