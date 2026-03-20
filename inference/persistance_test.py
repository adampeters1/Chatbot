"""
Sanity test for inference/persistence.py

Run from the project root with:
    python -m inference.persistence_test
"""

import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from model.network import Network
from features import Vocabulary, TfidfVectorizer
from inference.persistence import save_model, load_model


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
# Build a tiny end-to-end bundle
# ─────────────────────────────────────────────

def _make_bundle():
    # vocabulary from a toy corpus
    corpus = [
        ["hello", "there"],
        ["hello", "bot"],
        ["thanks", "bot"],
        ["bye", "bye", "bot"],
    ]
    vocab = Vocabulary(min_freq=1, max_size=None)
    vocab.build_from_tokens(corpus)

    # TF-IDF fitted on the same corpus
    tfidf = TfidfVectorizer(vocab)
    tfidf.fit(corpus)

    # tiny 3-class network
    net = Network([vocab.size, 5, 3], weight_scale=0.5, seed=123)

    label_map = {
        0: {"category": "greeting", "description": "hi"},
        1: {"category": "thanks",   "description": "ty"},
        2: {"category": "farewell", "description": "bye"},
    }

    return net, vocab, tfidf, label_map, corpus


# ─────────────────────────────────────────────
# Round-trip test
# ─────────────────────────────────────────────

def test_round_trip():
    print("\n--- save → load round trip ---")

    net, vocab, tfidf, label_map, corpus = _make_bundle()

    # Pick a fixed probe input; record forward() before saving.
    x_probe = tfidf.transform_single(corpus[0]).tolist()
    p_before = net.forward(x_probe)

    tmp_path = os.path.join(
        os.path.dirname(__file__), "_tmp_bundle_test.json"
    )
    try:
        save_model(tmp_path, net, vocab, tfidf, label_map)
        _run("bundle file written", os.path.isfile(tmp_path))

        loaded = load_model(tmp_path)
        net2   = loaded["network"]
        vocab2 = loaded["vocabulary"]
        tfidf2 = loaded["tfidf"]
        lm2    = loaded["label_map"]

        # ── network identity ────────────────────────
        _run("layer_sizes preserved",
             net2.layer_sizes == net.layer_sizes)

        weights_match = all(
            all(
                all(_almost_equal(a, b) for a, b in zip(r1, r2))
                for r1, r2 in zip(l1.W, l2.W)
            )
            for l1, l2 in zip(net.layers, net2.layers)
        )
        _run("all weight matrices preserved", weights_match)

        biases_match = all(
            all(_almost_equal(a, b) for a, b in zip(l1.b, l2.b))
            for l1, l2 in zip(net.layers, net2.layers)
        )
        _run("all bias vectors preserved", biases_match)

        # ── functional equivalence: same forward output ──
        x_probe2 = tfidf2.transform_single(corpus[0]).tolist()
        p_after = net2.forward(x_probe2)
        _run("forward() output identical after reload",
             all(_almost_equal(a, b) for a, b in zip(p_before, p_after)))

        _run("predict() identical after reload",
             net.predict(x_probe) == net2.predict(x_probe2))

        # ── vocabulary ──────────────────────────────
        _run("vocab token_to_index preserved",
             vocab2.token_to_index == vocab.token_to_index)
        _run("vocab size preserved", vocab2.size == vocab.size)

        # ── TF-IDF ─────────────────────────────────
        _run("idf vector preserved",
             np.allclose(tfidf2.idf, tfidf.idf))
        _run("n_documents preserved",
             tfidf2.n_documents == tfidf.n_documents)

        # ── label map ──────────────────────────────
        _run("label_map keys are ints",
             all(isinstance(k, int) for k in lm2.keys()))
        _run("label_map contents preserved",
             lm2 == label_map)

    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)


def test_errors():
    print("\n--- error handling ---")

    net, vocab, _, label_map, _ = _make_bundle()
    unfit = TfidfVectorizer(vocab)  # never fitted

    try:
        save_model("_x.json", net, vocab, unfit, label_map)
        _run("unfit TF-IDF raises", False)
    except ValueError:
        _run("unfit TF-IDF raises", True)

    try:
        load_model("_definitely_not_here_.json")
        _run("missing file raises", False)
    except FileNotFoundError:
        _run("missing file raises", True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 55)
    print("  persistence.py – sanity test suite")
    print("=" * 55)

    test_round_trip()
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