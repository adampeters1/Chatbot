"""
End-to-end inference pipeline: raw text → predicted intent.

Responsibilities
----------------
1.  Load a trained model bundle from disk, or — if none exists —
    run the full training pipeline from scratch and persist the
    resulting bundle for next time.
2.  For a raw input string:
        preprocess → TF-IDF vectorise → network.forward → argmax
    and return a structured prediction.

Depends on
----------
    preprocessing.preprocess / preprocess_dataset
    features.Vocabulary, TfidfVectorizer, LabelEncoder
    model.network.Network
    training.trainer.Trainer
    training.data_split.stratified_split
    utils.data_loader.load_data
    inference.persistence.save_model / load_model
    config.*
"""

import os

from config import (
    DATA_PATH,
    LABEL_MAP,
    NUM_CLASSES,
    PREPROCESSING,
    RANDOM_SEED,
    TEST_RATIO,
)

from preprocessing import preprocess, preprocess_dataset
from features import Vocabulary, TfidfVectorizer, LabelEncoder
from model.network import Network
from training.trainer import Trainer
from training.data_split import stratified_split
from utils.data_loader import load_data
from utils.matrix_ops import argmax

from inference.persistence import save_model, load_model


# ─────────────────────────────────────────────
# Default hyper-parameters for the cold-start training path.
# Pull from config if present; otherwise fall back to sensible defaults.
# ─────────────────────────────────────────────

try:
    from config import MODEL_PATH as _DEFAULT_MODEL_PATH
except ImportError:
    _DEFAULT_MODEL_PATH = "data/model_bundle.json"

try:
    from config import HIDDEN_SIZES as _DEFAULT_HIDDEN_SIZES
except ImportError:
    _DEFAULT_HIDDEN_SIZES = [64]

try:
    from config import LEARNING_RATE as _DEFAULT_LEARNING_RATE
except ImportError:
    _DEFAULT_LEARNING_RATE = 0.1

try:
    from config import EPOCHS as _DEFAULT_EPOCHS
except ImportError:
    _DEFAULT_EPOCHS = 50


# ─────────────────────────────────────────────
# InferencePipeline
# ─────────────────────────────────────────────

class InferencePipeline:
    """
    Raw-text → intent-label inference, with lazy cold-start training.

    Parameters
    ----------
    model_path : str, optional
        Path to a JSON bundle written by `persistence.save_model`.
        Defaults to `config.MODEL_PATH` if defined, else
        "data/model_bundle.json".
    preprocessing_config : dict, optional
        Keyword arguments forwarded to `preprocessing.preprocess`.
        Defaults to `config.PREPROCESSING`.
    auto_train : bool, default True
        If no bundle exists at `model_path`, train a model from
        scratch using `config.DATA_PATH` and then save it.  If False,
        a missing bundle raises FileNotFoundError.
    hidden_sizes : list[int], optional
        Hidden-layer widths for cold-start training.  Defaults to
        `config.HIDDEN_SIZES` if defined, else [64].
    learning_rate : float, optional
        SGD step size for cold-start training.
    epochs : int, optional
        Training epochs for cold-start training.
    verbose : bool, default True
        Print progress during load / train.

    Attributes
    ----------
    network      : Network
    vocabulary   : Vocabulary
    tfidf        : TfidfVectorizer
    label_map    : dict[int, dict]
    model_path   : str
    was_trained  : bool
        True if this pipeline cold-started a training run on
        construction; False if it loaded an existing bundle.
    """

    def __init__(
        self,
        model_path: str | None = None,
        preprocessing_config: dict | None = None,
        auto_train: bool = True,
        hidden_sizes: list | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> None:

        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self._pp_config = (
            dict(preprocessing_config)
            if preprocessing_config is not None
            else dict(PREPROCESSING)
        )
        self._hidden_sizes  = list(hidden_sizes) if hidden_sizes else list(_DEFAULT_HIDDEN_SIZES)
        self._learning_rate = learning_rate if learning_rate is not None else _DEFAULT_LEARNING_RATE
        self._epochs        = epochs if epochs is not None else _DEFAULT_EPOCHS
        self._verbose       = verbose

        self.network    = None
        self.vocabulary = None
        self.tfidf      = None
        self.label_map  = None
        self.was_trained = False

        # ── load or train ──────────────────────────────────
        if os.path.isfile(self.model_path):
            self._log(f"Loading model bundle from {self.model_path} …")
            self._load(self.model_path)
        elif auto_train:
            self._log(
                f"No bundle found at {self.model_path}; "
                f"training from scratch."
            )
            self._train_from_scratch()
            self._log(f"Saving trained bundle to {self.model_path} …")
            save_model(
                self.model_path,
                self.network,
                self.vocabulary,
                self.tfidf,
                self.label_map,
            )
            self.was_trained = True
        else:
            raise FileNotFoundError(
                f"No model bundle at {self.model_path} and "
                "auto_train is disabled."
            )

    # ─────────────────────────────────────────────
    # Public: prediction
    # ─────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Run the full inference chain on a single raw string.

        Parameters
        ----------
        text : str
            Raw user utterance.

        Returns
        -------
        dict
            {
              "label":         int,         # predicted class index
              "category":      str,         # e.g. "greeting"
              "description":   str,         # human-readable gloss
              "confidence":    float,       # probability of top class
              "probabilities": list[float], # full softmax output
              "tokens":        list[str],   # post-preprocessing tokens
            }
        """
        if not isinstance(text, str):
            raise TypeError(
                f"predict() expects a str, got {type(text).__name__}."
            )

        tokens = self._preprocess_text(text)
        vec    = self._vectorise(tokens)
        probs  = self.network.forward(vec)

        label_idx = argmax(probs)
        entry     = self.label_map.get(
            label_idx,
            {"category": f"label_{label_idx}", "description": ""},
        )

        return {
            "label":         label_idx,
            "category":      entry.get("category", f"label_{label_idx}"),
            "description":   entry.get("description", ""),
            "confidence":    probs[label_idx],
            "probabilities": list(probs),
            "tokens":        tokens,
        }

    def predict_top_k(self, text: str, k: int = 3) -> list:
        """
        Return the top-k most likely intents for a raw string.

        Parameters
        ----------
        text : str
        k : int, default 3

        Returns
        -------
        list[dict]
            Each entry: {"label", "category", "probability"},
            sorted descending by probability.
        """
        result = self.predict(text)
        probs  = result["probabilities"]
        ranked = sorted(
            range(len(probs)),
            key=lambda i: probs[i],
            reverse=True,
        )[:k]

        out = []
        for idx in ranked:
            entry = self.label_map.get(
                idx, {"category": f"label_{idx}", "description": ""}
            )
            out.append({
                "label":       idx,
                "category":    entry.get("category", f"label_{idx}"),
                "probability": probs[idx],
            })
        return out

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[InferencePipeline] {msg}")

    def _load(self, path: str) -> None:
        bundle = load_model(path)
        self.network    = bundle["network"]
        self.vocabulary = bundle["vocabulary"]
        self.tfidf      = bundle["tfidf"]
        self.label_map  = bundle["label_map"]

    # ── preprocessing bridge ───────────────────────────────
    def _preprocess_text(self, text: str) -> list:
        """
        Run the shared preprocessing chain on one raw string and
        return a list of tokens.

        Handles both possible return shapes of `preprocess`:
            - list[str]                (tokens directly)
            - dict with "tokens" key   (record-style)
        """
        result = preprocess(text, **self._pp_config)
        if isinstance(result, dict) and "tokens" in result:
            return list(result["tokens"])
        if isinstance(result, list):
            return result
        raise TypeError(
            "preprocessing.preprocess returned an unexpected type "
            f"{type(result).__name__}; expected list[str] or dict."
        )

    # ── vectorisation bridge ───────────────────────────────
    def _vectorise(self, tokens: list) -> list:
        """
        TF-IDF vectorise a token list and return a plain Python list
        (what Network.forward expects).
        """
        vec = self.tfidf.transform_single(tokens)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    # ── cold-start training ────────────────────────────────
    def _train_from_scratch(self) -> None:
        """
        Full Phase 1–5 pipeline: load → split → preprocess →
        vocab / TF-IDF / label-encode → build network → train.
        Populates self.network / vocabulary / tfidf / label_map.
        """
        # Phase 1 — load and split
        self._log(f"Loading dataset from {DATA_PATH} …")
        data = load_data(DATA_PATH)
        train_raw, _ = stratified_split(
            data, test_ratio=TEST_RATIO, random_seed=RANDOM_SEED
        )

        # Phase 2 — preprocess
        self._log("Preprocessing training data …")
        train_proc = preprocess_dataset(train_raw, **self._pp_config)

        # Phase 3 — features
        self._log("Building vocabulary and TF-IDF …")
        vocab = Vocabulary(min_freq=2, max_size=None)
        vocab.build_from_dataset(train_proc)

        tfidf = TfidfVectorizer(vocab)
        tfidf.fit_dataset(train_proc)

        X_train = tfidf.transform_dataset(train_proc)

        encoder = LabelEncoder(num_classes=NUM_CLASSES)
        y_train = encoder.encode_dataset(train_proc)

        # Phase 4 — build network
        layer_sizes = [vocab.size, *self._hidden_sizes, NUM_CLASSES]
        self._log(
            "Building network "
            f"{' → '.join(str(s) for s in layer_sizes)} …"
        )
        net = Network(layer_sizes, seed=RANDOM_SEED)

        # Phase 5 — train
        self._log(
            f"Training for {self._epochs} epochs "
            f"(lr={self._learning_rate}) …"
        )
        trainer = Trainer(
            network       = net,
            X_train       = X_train,
            y_train       = y_train,
            learning_rate = self._learning_rate,
            epochs        = self._epochs,
            seed          = RANDOM_SEED,
            verbose       = self._verbose,
        )
        trainer.fit()

        # stash artefacts
        self.network    = net
        self.vocabulary = vocab
        self.tfidf      = tfidf
        self.label_map  = dict(LABEL_MAP)

    # ─────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────

    def __repr__(self) -> str:
        origin = "trained" if self.was_trained else "loaded"
        arch = (
            " → ".join(str(s) for s in self.network.layer_sizes)
            if self.network else "∅"
        )
        return (
            f"InferencePipeline({origin}, arch={arch}, "
            f"vocab={len(self.vocabulary) if self.vocabulary else 0}, "
            f"classes={len(self.label_map) if self.label_map else 0})"
        )