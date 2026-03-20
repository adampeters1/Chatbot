"""
Lightweight sanity tests for model/network.py

Run from the project root with:
    python -m model.network_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.network import Network
from model.losses import categorical_cross_entropy


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
# Construction
# ─────────────────────────────────────────────

def test_construction():
    print("\n--- construction ---")

    net = Network([8, 16, 4], seed=0)

    _run("len(net) == n_layers", len(net) == 2)
    _run("layer[0] in/out dims",
         net.layers[0].input_size == 8 and net.layers[0].output_size == 16)
    _run("layer[1] in/out dims",
         net.layers[1].input_size == 16 and net.layers[1].output_size == 4)
    _run("hidden is relu",  net.layers[0].activation_name == "relu")
    _run("output is softmax", net.layers[1].activation_name == "softmax")

    # two-layer spec → single softmax layer
    logreg = Network([10, 3], seed=1)
    _run("single-layer net == softmax only",
         len(logreg) == 1 and logreg.layers[0].activation_name == "softmax")

    # repr contains architecture
    _run("repr is informative",
         "8" in repr(net) and "16" in repr(net) and "4" in repr(net))

    # validation
    try:
        Network([5])
        _run("too-short spec raises", False)
    except ValueError:
        _run("too-short spec raises", True)

    try:
        Network([5, -3])
        _run("non-positive size raises", False)
    except ValueError:
        _run("non-positive size raises", True)

    try:
        Network("bad")
        _run("non-list spec raises", False)
    except TypeError:
        _run("non-list spec raises", True)


# ─────────────────────────────────────────────
# Forward
# ─────────────────────────────────────────────

def test_forward():
    print("\n--- forward ---")

    net = Network([6, 10, 5], seed=42)
    x = [0.2, -0.5, 1.1, 0.0, 0.7, -0.3]
    p = net.forward(x)

    _run("output length == n_classes", len(p) == 5)
    _run("output sums to 1", _almost_equal(sum(p), 1.0))
    _run("output all positive", all(pi > 0.0 for pi in p))
    _run("output all finite",   all(math.isfinite(pi) for pi in p))

    try:
        net.forward([0.1, 0.2])  # wrong length
        _run("forward wrong length raises", False)
    except ValueError:
        _run("forward wrong length raises", True)


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

def test_predict():
    print("\n--- predict ---")

    net = Network([4, 8, 3], seed=5)
    x = [0.1, -0.2, 0.3, 0.4]
    pred = net.predict(x)

    _run("predict returns int", isinstance(pred, int))
    _run("predict in valid range", 0 <= pred < 3)

    # predict(x) == argmax(forward(x))
    probs = net.forward(x)
    manual = max(range(len(probs)), key=lambda i: probs[i])
    _run("predict matches argmax(forward)", pred == manual)


# ─────────────────────────────────────────────
# Training reduces loss
# ─────────────────────────────────────────────

def test_learning():
    print("\n--- learning (loss decreases) ---")

    # tiny 3-class problem on 4-D inputs
    X = [
        [1.0,  0.2, -0.1,  0.3],
        [-0.4, 1.2,  0.5, -0.2],
        [0.3, -0.5,  0.9,  1.1],
        [0.8,  0.1, -0.3,  0.2],
        [-0.2, 0.9,  0.4, -0.1],
        [0.1, -0.6,  1.0,  0.8],
    ]
    Y = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    net = Network([4, 12, 3], weight_scale=0.1, seed=7)
    lr  = 0.3
    epochs = 150

    def mean_loss():
        total = 0.0
        for x, y in zip(X, Y):
            p = net.forward(x)
            total += categorical_cross_entropy(y, p)
        return total / len(X)

    initial = mean_loss()
    for _ in range(epochs):
        for x, y in zip(X, Y):
            p = net.forward(x)
            net.backward(p, y, lr)
    final = mean_loss()

    _run("loss decreased", final < initial)
    _run("loss decreased substantially (>70%)",
         final < 0.3 * initial)
    print(f"       initial loss = {initial:.4f}, "
          f"final loss = {final:.4f}")

    # all training samples correctly classified after training
    all_correct = all(
        net.predict(x) == max(range(3), key=lambda i: y[i])
        for x, y in zip(X, Y)
    )
    _run("all training samples correctly predicted", all_correct)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  network.py – sanity test suite")
    print("=" * 55)

    test_construction()
    test_forward()
    test_predict()
    test_learning()

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