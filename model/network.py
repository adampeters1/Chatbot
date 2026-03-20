"""
Feed-forward neural network built from DenseLayer instances.

Chains an arbitrary number of fully-connected layers:
    input → [ReLU hidden]* → Softmax output

Depends on:
    - model.layers.DenseLayer
    - model.losses.softmax_cross_entropy_gradient
    - utils.matrix_ops.argmax
"""

from model.layers import DenseLayer
from model.losses import softmax_cross_entropy_gradient
from utils.matrix_ops import argmax


# ─────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────

class Network:
    """
    Multi-layer perceptron for multi-class classification.

    Parameters
    ----------
    layer_sizes : list[int]
        Sequence of layer widths, e.g. [vocab_size, 64, 32, n_classes].
        Must contain at least two integers (input and output).
        For a specification of length L, (L − 1) DenseLayer objects are
        created:

            layer_sizes[0] → layer_sizes[1]   (hidden, ReLU)
            layer_sizes[1] → layer_sizes[2]   (hidden, ReLU)
            …
            layer_sizes[-2] → layer_sizes[-1] (output, Softmax)

        With exactly two sizes the network collapses to a single
        softmax output layer (multinomial logistic regression).

    weight_scale : float, default 0.01
        σ of the Gaussian used to initialise each layer's weights.
        Passed straight through to every DenseLayer.

    seed : int or None, default None
        Base seed for reproducible initialisation.  If provided, layer
        k is seeded with (seed + k) so every layer draws from an
        independent but deterministic stream.  None leaves all layers
        unseeded.

    Attributes
    ----------
    layer_sizes : list[int]
        Copy of the size specification.
    layers : list[DenseLayer]
        The constituent layers, in forward order.
    """

    def __init__(
        self,
        layer_sizes: list,
        weight_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        # ── validate specification ──────────────────────────
        if not isinstance(layer_sizes, list):
            raise TypeError(
                "layer_sizes must be a list of ints, got "
                f"{type(layer_sizes).__name__}."
            )
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must contain at least two entries "
                "(input and output dimensions); got "
                f"{layer_sizes!r}."
            )
        for i, s in enumerate(layer_sizes):
            if not (isinstance(s, int) and s > 0):
                raise ValueError(
                    f"layer_sizes[{i}] must be a positive int, "
                    f"got {s!r}."
                )

        self.layer_sizes: list[int] = list(layer_sizes)

        # ── build layers ────────────────────────────────────
        self.layers: list[DenseLayer] = []
        n_layers = len(layer_sizes) - 1

        for k in range(n_layers):
            in_dim  = layer_sizes[k]
            out_dim = layer_sizes[k + 1]
            is_output = (k == n_layers - 1)
            activation = "softmax" if is_output else "relu"
            layer_seed = None if seed is None else (seed + k)

            self.layers.append(
                DenseLayer(
                    input_size   = in_dim,
                    output_size  = out_dim,
                    activation   = activation,
                    weight_scale = weight_scale,
                    seed         = layer_seed,
                )
            )

    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────

    def forward(self, input_vector: list) -> list:
        """
        Propagate an input through all layers.

        Parameters
        ----------
        input_vector : list
            1-D feature vector of length `layer_sizes[0]`.

        Returns
        -------
        list
            Softmax probability vector of length `layer_sizes[-1]`.
            Each element lies in (0, 1) and the vector sums to 1.
        """
        if not isinstance(input_vector, list):
            raise TypeError(
                "forward(): input_vector must be a list, "
                f"got {type(input_vector).__name__}."
            )
        if len(input_vector) != self.layer_sizes[0]:
            raise ValueError(
                "forward(): input length "
                f"{len(input_vector)} does not match network input "
                f"size {self.layer_sizes[0]}."
            )

        out = input_vector
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ─────────────────────────────────────────────
    # Backward
    # ─────────────────────────────────────────────

    def backward(
        self,
        predicted: list,
        actual: list,
        learning_rate: float,
    ) -> None:
        """
        Backpropagate and update every layer's parameters in place.

        Must be called immediately after `forward()` on the same sample
        so that each layer still holds its cached activations.

        Parameters
        ----------
        predicted : list
            Softmax output returned by the most recent `forward()`.
        actual : list
            One-hot (or soft-label) target vector, same length as
            `predicted`.
        learning_rate : float
            SGD step size applied uniformly to every layer.
        """
        if not isinstance(predicted, list) or not isinstance(actual, list):
            raise TypeError(
                "backward(): predicted and actual must both be lists."
            )
        out_dim = self.layer_sizes[-1]
        if len(predicted) != out_dim or len(actual) != out_dim:
            raise ValueError(
                "backward(): predicted/actual must have length "
                f"{out_dim}; got {len(predicted)} and {len(actual)}."
            )

        # The output layer uses the fused softmax + cross-entropy
        # gradient, which is already dL/dz at the output — see
        # DenseLayer.backward() for the convention.
        grad = softmax_cross_entropy_gradient(actual, predicted)

        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    # ─────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────

    def predict(self, input_vector: list) -> int:
        """
        Return the most-likely class index for a single input.

        Parameters
        ----------
        input_vector : list
            1-D feature vector.

        Returns
        -------
        int
            argmax of the softmax output.
        """
        probs = self.forward(input_vector)
        return argmax(probs)

    # ─────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────

    def __repr__(self) -> str:
        arch = " → ".join(str(s) for s in self.layer_sizes)
        return f"Network({arch})"

    def __len__(self) -> int:
        return len(self.layers)