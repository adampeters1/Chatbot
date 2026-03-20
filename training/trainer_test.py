"""
Brief sanity test for training/trainer.py

Run from the project root with:
    python -m training.trainer_test
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from model.network import Network
from training.trainer import Trainer


PASS = "  PASS"
FAIL = "  FAIL"
_failures = []


def _run(name, passed):
    status = PASS if passed else FAIL
    print(f"{status}  {name}")
    if not passed:
        _failures.append(name)


# ─────────────────────────────────────────────
# A tiny synthetic 3-class dataset
# ─────────────────────────────────────────────

def _make_toy_data():
    """
    6 samples in 4-D space, one-hot labels in 3 classes.
    Features are deliberately linearly-separable-ish so a small
    MLP converges fast.
    """
    X = np.array([
        [1.0,  0.2, -0.1,  0.3],
        [0.8,  0.1, -0.2,  0.4],   # class 0
        [-0.4, 1.2,  0.5, -0.2],
        [-0.3, 0.9,  0.4, -0.1],   # class 1
        [0.3, -0.5,  0.9,  1.1],
        [0.1, -0.6,  1.0,  0.8],   # class 2
    ], dtype=np.float32)

    y = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=np.float32)

    return X, y


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_trainer_runs_and_improves():
    print("\n--- trainer: runs and improves ---")

    X, y = _make_toy_data()
    net = Network([4, 12, 3], weight_scale=0.1, seed=7)
    trainer = Trainer(
        network       = net,
        X_train       = X,
        y_train       = y,
        learning_rate = 0.3,
        epochs        = 60,
        seed          = 0,
        verbose       = False,
    )

    history = trainer.fit()

    _run("history has loss per epoch",
         len(history["loss"]) == 60)
    _run("history has accuracy per epoch",
         len(history["accuracy"]) == 60)

    initial_loss = history["loss"][0]
    final_loss   = history["loss"][-1]
    final_acc    = history["accuracy"][-1]

    _run("loss decreased over training",
         final_loss < initial_loss)
    _run("loss decreased substantially (>70%)",
         final_loss < 0.3 * initial_loss)
    _run("final training accuracy == 100%",
         abs(final_acc - 1.0) < 1e-9)

    # evaluate() matches
    eval_result = trainer.evaluate(X, y)
    _run("evaluate accuracy == 100%",
         abs(eval_result["accuracy"] - 1.0) < 1e-9)


def test_shuffle_preserves_alignment():
    print("\n--- trainer: shuffle preserves X/y alignment ---")

    # If X and y were misaligned by shuffling, training would fail
    # catastrophically on this dataset — loss wouldn't drop and
    # accuracy wouldn't reach 1.0.  The convergence check therefore
    # doubles as an alignment check.  We additionally confirm the
    # arrays themselves are untouched.

    X, y = _make_toy_data()
    X_copy = X.copy()
    y_copy = y.copy()

    net = Network([4, 8, 3], weight_scale=0.1, seed=3)
    trainer = Trainer(net, X, y,
                      learning_rate=0.3, epochs=40,
                      seed=1, verbose=False)
    trainer.fit()

    _run("X_train not mutated by shuffling",
         np.array_equal(X, X_copy))
    _run("y_train not mutated by shuffling",
         np.array_equal(y, y_copy))


def test_accepts_plain_lists():
    print("\n--- trainer: accepts list-of-lists input ---")

    X, y = _make_toy_data()
    X_list = X.tolist()
    y_list = y.tolist()

    net = Network([4, 8, 3], weight_scale=0.1, seed=5)
    trainer = Trainer(net, X_list, y_list,
                      learning_rate=0.3, epochs=40,
                      seed=2, verbose=False)
    history = trainer.fit()

    _run("loss decreases with list inputs",
         history["loss"][-1] < history["loss"][0])


def test_reproducibility():
    print("\n--- trainer: reproducibility with fixed seed ---")

    X, y = _make_toy_data()

    def train_once():
        net = Network([4, 8, 3], weight_scale=0.1, seed=9)
        trainer = Trainer(net, X, y,
                          learning_rate=0.3, epochs=20,
                          seed=99, verbose=False)
        trainer.fit()
        return trainer.history["loss"]

    losses_a = train_once()
    losses_b = train_once()

    _run("same seed → identical loss trajectory",
         all(abs(a - b) < 1e-12 for a, b in zip(losses_a, losses_b)))


def test_validation():
    print("\n--- trainer: argument validation ---")

    X, y = _make_toy_data()
    net = Network([4, 8, 3], seed=0)

    try:
        Trainer(net, X, y[:3], epochs=10, verbose=False)
        _run("X/y length mismatch raises", False)
    except ValueError:
        _run("X/y length mismatch raises", True)

    try:
        Trainer(net, X, y, epochs=0, verbose=False)
        _run("epochs<1 raises", False)
    except ValueError:
        _run("epochs<1 raises", True)

    try:
        Trainer(net, X, y, learning_rate=-0.1, verbose=False)
        _run("negative lr raises", False)
    except ValueError:
        _run("negative lr raises", True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  trainer.py – sanity test suite")
    print("=" * 55)

    test_trainer_runs_and_improves()
    test_shuffle_preserves_alignment()
    test_accepts_plain_lists()
    test_reproducibility()
    test_validation()

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