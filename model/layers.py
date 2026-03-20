"""
A single fully-connected (dense) layer with forward and backward passes.

The layer stores its parameters (W, b), the activation function to apply,
and caches the intermediate values needed for backpropagation.

Depends only on:
    - utils.matrix_ops      (pure-Python linear algebra)
    - model.activations     (ReLU / sigmoid / softmax + derivatives)
    - random                (weight initialisation)
"""

import random

from utils.matrix_ops import (
    matrix_vector_multiply,
    vector_add,
    hadamard_product,
    outer_product,
    matrix_transpose,
    matrix_add,
    scalar_matrix_multiply,
    scalar_vector_multiply,
)
from model.activations import ACTIVATIONS


# ─────────────────────────────────────────────
# DenseLayer
# ─────────────────────────────────────────────

class DenseLayer:
    """
    Fully-connected layer: a = activation(W · x + b).

    Parameters
    ----------
    input_size : int
        Length of the input vector x.
    output_size : int
        Length of the output vector a (number of neurons).
    activation : str, default "relu"
        Name of the activation function.  Must be a key in
        model.activations.ACTIVATIONS ("relu", "sigmoid", "softmax").
    weight_scale : float, default 0.01
        Standard deviation of the Gaussian used to initialise W.
    seed : int or None, default None
        Optional seed for the layer's private RNG (reproducible tests).

    Attributes
    ----------
    W : list[list[float]]   — weight matrix, shape (output_size × input_size)
    b : list[float]         — bias vector, length output_size
    _input : list[float] or None
        Cached input x from the most recent forward().
    _z : list[float] or None
        Cached pre-activation z = W·x + b.
    _a : list[float] or None
        Cached post-activation a = activation(z).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        weight_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        # ── validate dimensions ─────────────────────────────
        if not (isinstance(input_size, int) and input_size > 0):
            raise ValueError(
                f"input_size must be a positive int, got {input_size!r}."
            )
        if not (isinstance(output_size, int) and output_size > 0):
            raise ValueError(
                f"output_size must be a positive int, got {output_size!r}."
            )

        # ── resolve activation ──────────────────────────────
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Expected one of: {sorted(ACTIVATIONS)}."
            )
        self.activation_name   = activation
        self._activation_fwd   = ACTIVATIONS[activation]["forward"]
        self._activation_deriv = ACTIVATIONS[activation]["derivative"]

        # ── sizes ───────────────────────────────────────────
        self.input_size  = input_size
        self.output_size = output_size

        # ── parameters ──────────────────────────────────────
        rng = random.Random(seed)
        self.W: list[list[float]] = [
            [rng.gauss(0.0, weight_scale) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.b: list[float] = [0.0] * output_size

        # ── forward-pass cache ─────────────────────────────
        self._input: list[float] | None = None
        self._z:     list[float] | None = None
        self._a:     list[float] | None = None

    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────

    def forward(self, input_vector: list) -> list:
        """
        Run a forward pass through the layer.

        Computes z = W · x + b, then a = activation(z), caching x, z, a
        for use in backward().

        Parameters
        ----------
        input_vector : list
            Input x of length `self.input_size`.

        Returns
        -------
        list
            Post-activation output a of length `self.output_size`.
        """
        if not isinstance(input_vector, list):
            raise TypeError(
                f"forward() expects a list input, "
                f"got {type(input_vector).__name__}."
            )
        if len(input_vector) != self.input_size:
            raise ValueError(
                "forward(): input length "
                f"{len(input_vector)} does not match "
                f"layer input_size {self.input_size}."
            )

        # z = W·x + b
        z = vector_add(
            matrix_vector_multiply(self.W, input_vector),
            self.b,
        )
        # a = σ(z)
        a = self._activation_fwd(z)

        # cache for backward
        self._input = input_vector
        self._z     = z
        self._a     = a

        return a

    # ─────────────────────────────────────────────
    # Backward
    # ─────────────────────────────────────────────

    def backward(self, output_gradient: list, learning_rate: float) -> list:
        """
        Backpropagate through the layer and update W, b in place.

        Parameters
        ----------
        output_gradient : list
            Gradient dL/da flowing back from the layer above.
            For a *softmax* output layer, this should be the already-
            combined gradient (y_pred − y_true) and will be treated
            directly as dL/dz — the activation derivative is skipped.
        learning_rate : float
            Step size for the parameter update.

        Returns
        -------
        list
            Gradient dL/dx with respect to this layer's input,
            of length `self.input_size`, to be fed into the preceding
            layer's backward().
        """
        if self._input is None or self._z is None:
            raise RuntimeError(
                "backward() called before forward(); "
                "no cached activations available."
            )
        if not isinstance(output_gradient, list):
            raise TypeError(
                "backward(): output_gradient must be a list, "
                f"got {type(output_gradient).__name__}."
            )
        if len(output_gradient) != self.output_size:
            raise ValueError(
                "backward(): output_gradient length "
                f"{len(output_gradient)} does not match "
                f"layer output_size {self.output_size}."
            )
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                "backward(): learning_rate must be numeric, "
                f"got {type(learning_rate).__name__}."
            )

        # ── dL/dz: pass through the activation ──────────────
        #
        # For softmax + cross-entropy the caller supplies the combined
        # gradient (y_pred − y_true), which is already dL/dz.  We do
        # *not* multiply by the softmax Jacobian here — that would
        # double-count it.
        #
        # For element-wise activations (ReLU, sigmoid) the derivative
        # is a vector and we combine with a Hadamard product.
        if self.activation_name == "softmax":
            dz = output_gradient
        else:
            act_grad = self._activation_deriv(self._z)
            dz = hadamard_product(output_gradient, act_grad)

        # ── gradients w.r.t. parameters ────────────────────
        # dL/dW  (output_size × input_size)  — outer product of dz and x
        dW = outer_product(dz, self._input)
        # dL/db  (output_size)               — just dz
        db = dz

        # ── gradient to propagate back ─────────────────────
        # dL/dx = Wᵀ · dz
        W_T  = matrix_transpose(self.W)
        dx   = matrix_vector_multiply(W_T, dz)

        # ── parameter updates (SGD step) ───────────────────
        # W ← W − η · dW
        self.W = matrix_add(
            self.W,
            scalar_matrix_multiply(-learning_rate, dW),
        )
        # b ← b − η · db
        self.b = vector_add(
            self.b,
            scalar_vector_multiply(-learning_rate, db),
        )

        return dx

    # ─────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DenseLayer(in={self.input_size}, out={self.output_size}, "
            f"activation='{self.activation_name}')"
        )