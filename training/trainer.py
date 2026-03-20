"""
Training loop for the intent-classification network.

Manages the full SGD training process: per-epoch shuffling, per-sample
forward/backward passes, loss and accuracy tracking, and progress
reporting.  Designed to consume the NumPy arrays produced by the
Phase-3 feature pipeline while feeding plain Python lists into the
pure-Python Network class.
"""

import random

from model.losses import categorical_cross_entropy
from utils.matrix_ops import argmax


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _as_list(row):
    """
    Normalise a single feature / label row to a plain Python list.

    Accepts:
        - numpy.ndarray  → row.tolist()
        - list / tuple   → list(row)
    """
    if hasattr(row, "tolist"):
        return row.tolist()
    return list(row)


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Stochastic-gradient-descent trainer for `model.network.Network`.

    Parameters
    ----------
    network : Network
        The model to train.  Must expose `forward(x)`, `backward(p, y, lr)`
        and `predict(x)` with the signatures defined in model/network.py.
    X_train : array-like, shape (n_samples, n_features)
        Training feature vectors.  Rows may be NumPy arrays or lists.
    y_train : array-like, shape (n_samples, n_classes)
        One-hot encoded training labels.  Same row-type caveat as X.
    learning_rate : float, default 0.1
        SGD step size.
    epochs : int, default 50
        Number of full passes over the training data.
    seed : int or None, default None
        Seed for the trainer's private RNG (controls per-epoch shuffling).
        None → non-deterministic ordering.
    verbose : bool, default True
        If True, print one progress line per epoch.

    Attributes
    ----------
    history : dict
        Keys "loss" and "accuracy", each a list of length `epochs`
        holding the per-epoch mean training loss and training accuracy.
    """

    def __init__(
        self,
        network,
        X_train,
        y_train,
        learning_rate: float = 0.1,
        epochs: int = 50,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        # ── validate basic hyper-parameters ─────────────────
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(
                f"epochs must be a positive int, got {epochs!r}."
            )
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be a positive number, "
                f"got {learning_rate!r}."
            )

        # ── validate data shapes ────────────────────────────
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("X_train is empty.")
        if len(y_train) != n_samples:
            raise ValueError(
                "X_train and y_train must have the same number of rows: "
                f"X has {n_samples}, y has {len(y_train)}."
            )

        # ── stash everything ────────────────────────────────
        self.network       = network
        self.X_train       = X_train
        self.y_train       = y_train
        self.learning_rate = float(learning_rate)
        self.epochs        = epochs
        self.verbose       = verbose
        self._rng          = random.Random(seed)

        self.n_samples = n_samples
        self.history   = {"loss": [], "accuracy": []}

    # ─────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Run the full training loop.

        Returns
        -------
        dict
            The `self.history` dictionary containing per-epoch
            "loss" and "accuracy" lists.
        """
        indices = list(range(self.n_samples))

        if self.verbose:
            self._print_header()

        for epoch in range(1, self.epochs + 1):

            # Shuffle *indices*, never the arrays themselves, so X and
            # y stay aligned regardless of whether they are NumPy
            # arrays, Python lists, or anything else row-indexable.
            self._rng.shuffle(indices)

            epoch_loss = 0.0
            correct    = 0

            for i in indices:
                x = _as_list(self.X_train[i])
                y = _as_list(self.y_train[i])

                # ── forward ─────────────────────────
                p = self.network.forward(x)

                # ── loss & accuracy accounting ──────
                epoch_loss += categorical_cross_entropy(y, p)
                if argmax(p) == argmax(y):
                    correct += 1

                # ── backward + parameter update ─────
                self.network.backward(p, y, self.learning_rate)

            mean_loss = epoch_loss / self.n_samples
            accuracy  = correct / self.n_samples

            self.history["loss"].append(mean_loss)
            self.history["accuracy"].append(accuracy)

            if self.verbose:
                self._print_epoch(epoch, mean_loss, accuracy)

        if self.verbose:
            self._print_footer()

        return self.history

    # ─────────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────────

    def evaluate(self, X, y) -> dict:
        """
        Compute mean loss and accuracy on an arbitrary dataset
        *without* updating any parameters.

        Parameters
        ----------
        X : array-like, shape (n, n_features)
        y : array-like, shape (n, n_classes) — one-hot labels.

        Returns
        -------
        dict  {"loss": float, "accuracy": float}
        """
        n = len(X)
        if n == 0:
            raise ValueError("evaluate() received an empty dataset.")
        if len(y) != n:
            raise ValueError(
                f"evaluate(): X has {n} rows but y has {len(y)}."
            )

        total_loss = 0.0
        correct    = 0

        for i in range(n):
            xi = _as_list(X[i])
            yi = _as_list(y[i])
            p  = self.network.forward(xi)
            total_loss += categorical_cross_entropy(yi, p)
            if argmax(p) == argmax(yi):
                correct += 1

        return {
            "loss":     total_loss / n,
            "accuracy": correct / n,
        }

    # ─────────────────────────────────────────────
    # Console output
    # ─────────────────────────────────────────────

    def _print_header(self) -> None:
        print("\n" + "=" * 65)
        print(" TRAINING")
        print("=" * 65)
        print(f"  Samples       : {self.n_samples}")
        print(f"  Epochs        : {self.epochs}")
        print(f"  Learning rate : {self.learning_rate}")
        print(f"  Architecture  : {self.network}")
        print("-" * 65)
        print(f"  {'Epoch':>6}   {'Loss':>12}   {'Accuracy':>10}")
        print("-" * 65)

    def _print_epoch(self, epoch: int, loss: float, acc: float) -> None:
        print(f"  {epoch:>6}   {loss:>12.6f}   {acc * 100:>9.2f}%")

    def _print_footer(self) -> None:
        if not self.history["loss"]:
            print("=" * 65)
            return
        final_loss = self.history["loss"][-1]
        final_acc  = self.history["accuracy"][-1]
        print("-" * 65)
        print(
            f"  Final: loss = {final_loss:.6f}, "
            f"train accuracy = {final_acc * 100:.2f}%"
        )
        print("=" * 65)