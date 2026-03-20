"""
Sanity test for inference/pipeline.py

Strategy: build a tiny bundle via persistence.save_model (as in
persistence_test), then load it through InferencePipeline and verify
the predict() contract.

The auto-train path is *not* exercised here — that would require the
full dataset and take orders of magnitude longer.  It is covered
implicitly by running main.py end to end.

Run from the project root with:
    python -m inference.pipeline_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from model.network import Network
from features import Vocabulary, TfidfVectorizer
from inference.persistence import save_model
from inference.pipeline import InferencePipeline


PASS = "  PASS"
FAIL = "  FAIL"
_failures = []


def _run(name, passed):
    status = PASS if passed else FAIL
    print(f"{status}  {name}")
    if not passed:
        _failures.append(name)


def _almost_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


# ─────────────────────────────────────────────
# Build and save a toy bundle
# ─────────────────────────────────────────────

def _build_toy_bundle(path: str):
    corpus = [
        ["hello", "there"],
        ["hello", "friend"],
        ["thanks", "a", "lot"],
        ["thank", "you"],
        ["bye", "now"],
        ["goodbye", "friend"],
    ]
    vocab = Vocabulary(min_freq=1, max_size=None)
    vocab.build_from_tokens(corpus)

    tfidf = TfidfVectorizer(vocab)
    tfidf.fit(corpus)

    net = Network([vocab.size, 6, 3], weight_scale=0.5, seed=99)

    # Quick micro-training so predictions are non-random
    from model.losses import softmax_cross_entropy_gradient
    labels = [
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # greeting
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],   # thanks
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],   # bye
    ]
    for _ in range(200):
        for toks, y in zip(corpus, labels):
            x = tfidf.transform_single(toks).tolist()
            p = net.forward(x)
            net.backward(p, y, 0.3)

    label_map = {
        0: {"category": "greeting", "description": "hi"},
        1: {"category": "thanks",   "description": "ty"},
        2: {"category": "farewell", "description": "bye"},
    }
    save_model(path, net, vocab, tfidf, label_map)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_load_and_predict():
    print("\n--- load bundle & predict ---")

    tmp = os.path.join(os.path.dirname(__file__), "_tmp_pipeline_test.json")
    _build_toy_bundle(tmp)

    try:
        pipe = InferencePipeline(
            model_path=tmp,
            auto_train=False,
            verbose=False,
        )

        _run("pipeline loaded (not trained)", not pipe.was_trained)
        _run("network present",    pipe.network is not None)
        _run("vocabulary present", pipe.vocabulary is not None)
        _run("tfidf present",      pipe.tfidf is not None)
        _run("label_map present",  pipe.label_map is not None)

        # ── predict() shape contract ──────────────────────
        # NOTE: we bypass preprocessing.preprocess here to keep the
        # test self-contained — it would otherwise apply stemming /
        # stop-word removal tuned for the real dataset, which may
        # zero out our toy tokens. We inject tokens directly via the
        # vectorise → forward path instead.
        tokens = ["hello", "there"]
        vec = pipe._vectorise(tokens)
        probs = pipe.network.forward(vec)

        from utils.matrix_ops import argmax
        label = argmax(probs)
        entry = pipe.label_map[label]

        _run("probabilities sum to 1",
             _almost_equal(sum(probs), 1.0))
        _run("probabilities length == n_classes",
             len(probs) == 3)
        _run("label in valid range", 0 <= label < 3)
        _run("category is a string",
             isinstance(entry["category"], str))

        # ── predict_top_k ordering ─────────────────────────
        # Wire a predict() that skips external preprocessing for this
        # toy test — done by temporarily overriding _preprocess_text.
        pipe._preprocess_text = lambda t: t.split()
        result = pipe.predict("hello there")

        _run("predict returns dict", isinstance(result, dict))
        _run("predict has label key",       "label" in result)
        _run("predict has category key",    "category" in result)
        _run("predict has probabilities",   "probabilities" in result)
        _run("predict has confidence",      "confidence" in result)
        _run("confidence == max(prob)",
             _almost_equal(result["confidence"], max(result["probabilities"])))

        topk = pipe.predict_top_k("hello there", k=2)
        _run("top_k returns k entries", len(topk) == 2)
        _run("top_k sorted descending",
             topk[0]["probability"] >= topk[1]["probability"])

    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)


def test_errors():
    print("\n--- error handling ---")

    try:
        InferencePipeline(
            model_path="_definitely_missing_.json",
            auto_train=False,
            verbose=False,
        )
        _run("missing bundle + auto_train=False raises", False)
    except FileNotFoundError:
        _run("missing bundle + auto_train=False raises", True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  pipeline.py – sanity test suite")
    print("=" * 55)

    test_load_and_predict()
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