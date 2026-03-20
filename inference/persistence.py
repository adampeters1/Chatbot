"""
Model persistence — save / load the full inference bundle as JSON.

A saved bundle contains everything needed to run inference on a raw
(preprocessed) token list without access to the original training
artefacts:

    - network architecture (layer_sizes + per-layer activation names)
    - every layer's weight matrix and bias vector
    - the vocabulary's token→index and index→token mappings
    - the TF-IDF vectorizer's IDF vector and document count
    - the LABEL_MAP (label → {category, description})

Depends only on:
    - json                  (serialisation)
    - os                    (directory creation / path checks)
    - model.network.Network (reconstruction target)
    - model.layers.DenseLayer
    - features.Vocabulary, TfidfVectorizer
"""

import json
import os

from model.network import Network
from model.layers import DenseLayer
from features import Vocabulary, TfidfVectorizer


# ─────────────────────────────────────────────
# Bundle schema version — bump if the JSON layout changes
# ─────────────────────────────────────────────

_BUNDLE_VERSION = 1


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

def save_model(
    path: str,
    network,
    vocabulary,
    tfidf_vectorizer,
    label_map: dict,
) -> None:
    """
    Serialise the full inference bundle to a single JSON file.

    Parameters
    ----------
    path : str
        Destination file path.  Parent directory is created if it
        doesn't already exist.
    network : Network
        A trained `model.network.Network` instance.
    vocabulary : Vocabulary
        The fitted vocabulary used to build feature vectors.
    tfidf_vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer (must have `.idf` populated).
    label_map : dict
        Integer-label → {"category": str, "description": str} mapping.
        Typically `config.LABEL_MAP`.

    Raises
    ------
    ValueError
        If the vectorizer hasn't been fit (no IDF values).
    TypeError
        If any argument is the wrong type.
    """
    # ── validate ────────────────────────────────────────
    if not isinstance(network, Network):
        raise TypeError(
            f"save_model: expected a Network, got {type(network).__name__}."
        )
    if not isinstance(vocabulary, Vocabulary):
        raise TypeError(
            f"save_model: expected a Vocabulary, got "
            f"{type(vocabulary).__name__}."
        )
    if not isinstance(tfidf_vectorizer, TfidfVectorizer):
        raise TypeError(
            f"save_model: expected a TfidfVectorizer, got "
            f"{type(tfidf_vectorizer).__name__}."
        )
    if tfidf_vectorizer.idf is None:
        raise ValueError(
            "save_model: TfidfVectorizer has no IDF values — "
            "call .fit() / .fit_dataset() before saving."
        )
    if not isinstance(label_map, dict):
        raise TypeError(
            f"save_model: expected a dict label_map, got "
            f"{type(label_map).__name__}."
        )

    # ── network section ─────────────────────────────────
    layers_payload = []
    for layer in network.layers:
        layers_payload.append({
            "input_size":  layer.input_size,
            "output_size": layer.output_size,
            "activation":  layer.activation_name,
            # W is already list[list[float]]; copy for safety
            "W": [list(row) for row in layer.W],
            "b": list(layer.b),
        })

    network_payload = {
        "layer_sizes": list(network.layer_sizes),
        "layers":      layers_payload,
    }

    # ── vocabulary section ──────────────────────────────
    vocab_payload = {
        "token_to_index": dict(vocabulary.token_to_index),
        "size":           vocabulary.size,
        "min_freq":       vocabulary.min_freq,
        "max_size":       vocabulary.max_size,
    }

    # ── TF-IDF section ──────────────────────────────────
    # .idf is a numpy array → .tolist() for JSON
    idf_list = (
        tfidf_vectorizer.idf.tolist()
        if hasattr(tfidf_vectorizer.idf, "tolist")
        else list(tfidf_vectorizer.idf)
    )
    tfidf_payload = {
        "idf":          idf_list,
        "n_documents":  int(tfidf_vectorizer.n_documents),
    }

    # ── label map section ───────────────────────────────
    # JSON requires string keys; we'll coerce back on load.
    label_map_payload = {
        str(k): v for k, v in label_map.items()
    }

    # ── assemble and write ──────────────────────────────
    bundle = {
        "version":    _BUNDLE_VERSION,
        "network":    network_payload,
        "vocabulary": vocab_payload,
        "tfidf":      tfidf_payload,
        "label_map":  label_map_payload,
    }

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh)


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_model(path: str):
    """
    Reconstruct the full inference bundle from a JSON file written
    by `save_model`.

    Parameters
    ----------
    path : str
        Path to the JSON bundle.

    Returns
    -------
    dict
        {
          "network":    Network,
          "vocabulary": Vocabulary,
          "tfidf":      TfidfVectorizer,
          "label_map":  dict[int, dict],
        }

    Raises
    ------
    FileNotFoundError
        If `path` doesn't exist.
    ValueError
        If the file is malformed or the version is unsupported.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"load_model: no such file: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    version = bundle.get("version")
    if version != _BUNDLE_VERSION:
        raise ValueError(
            f"load_model: unsupported bundle version {version!r} "
            f"(expected {_BUNDLE_VERSION})."
        )

    # ── vocabulary ──────────────────────────────────────
    vp = bundle["vocabulary"]
    vocab = Vocabulary(
        min_freq=vp.get("min_freq", 2),
        max_size=vp.get("max_size", None),
    )
    # Rehydrate mappings directly (no corpus re-scan needed)
    vocab.token_to_index = {
        tok: int(idx) for tok, idx in vp["token_to_index"].items()
    }
    vocab.index_to_token = {
        idx: tok for tok, idx in vocab.token_to_index.items()
    }
    vocab.size = len(vocab.token_to_index)
    # token_counts left empty — not needed for inference.

    # ── TF-IDF ─────────────────────────────────────────
    import numpy as np
    tp = bundle["tfidf"]
    tfidf = TfidfVectorizer(vocab)
    tfidf.idf = np.asarray(tp["idf"], dtype=np.float32)
    tfidf.n_documents = int(tp["n_documents"])

    # ── network ─────────────────────────────────────────
    npld = bundle["network"]
    layer_sizes = [int(s) for s in npld["layer_sizes"]]
    network = Network(layer_sizes)   # random init; weights overwritten next

    saved_layers = npld["layers"]
    if len(saved_layers) != len(network.layers):
        raise ValueError(
            "load_model: layer count mismatch — bundle has "
            f"{len(saved_layers)} layers, architecture implies "
            f"{len(network.layers)}."
        )

    for live, saved in zip(network.layers, saved_layers):
        # Sanity-check dimensions & activation
        if live.input_size != saved["input_size"] \
           or live.output_size != saved["output_size"]:
            raise ValueError(
                "load_model: dimension mismatch on a DenseLayer — "
                f"expected ({live.input_size}→{live.output_size}), "
                f"bundle has ({saved['input_size']}→{saved['output_size']})."
            )
        if live.activation_name != saved["activation"]:
            raise ValueError(
                "load_model: activation mismatch — expected "
                f"'{live.activation_name}', bundle has "
                f"'{saved['activation']}'."
            )
        live.W = [list(row) for row in saved["W"]]
        live.b = list(saved["b"])

    # ── label map ───────────────────────────────────────
    label_map = {
        int(k): v for k, v in bundle["label_map"].items()
    }

    return {
        "network":    network,
        "vocabulary": vocab,
        "tfidf":      tfidf,
        "label_map":  label_map,
    }