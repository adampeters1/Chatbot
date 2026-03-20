"""
main.py — Full Phase 1-6 verification + CLI chatbot entry point.
"""

from config import (
    DATA_PATH, TEST_RATIO, RANDOM_SEED, LABEL_MAP,
    PREPROCESSING, NUM_CLASSES
)
from utils.data_loader import load_data, print_data_summary
from training.data_split import stratified_split, print_split_summary
from preprocessing import preprocess, preprocess_dataset
from features import (
    Vocabulary,
    BagOfWordsVectorizer,
    TfidfVectorizer,
    LabelEncoder
)
import numpy as np


# ─────────────────────────────────────────────
# Phase 1
# ─────────────────────────────────────────────

def verify_phase_1():
    """Load data and perform stratified split."""
    print("\n" + "=" * 65)
    print(" PHASE 1 — Data Loading & Splitting")
    print("=" * 65)

    data = load_data(DATA_PATH)
    print_data_summary(data)

    train_data, test_data = stratified_split(
        data,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
    )
    print_split_summary(train_data, test_data)

    return data, train_data, test_data


# ─────────────────────────────────────────────
# Phase 2
# ─────────────────────────────────────────────

def verify_phase_2(train_data, test_data):
    """Run preprocessing pipeline."""
    print("\n" + "=" * 65)
    print(" PHASE 2 — Preprocessing Pipeline")
    print("=" * 65)

    train_processed = preprocess_dataset(train_data, **PREPROCESSING)
    test_processed  = preprocess_dataset(test_data,  **PREPROCESSING)

    print(f"  Training samples processed : {len(train_processed)}")
    print(f"  Testing samples processed  : {len(test_processed)}")

    all_train_tokens = []
    empty_count = 0
    for record in train_processed:
        all_train_tokens.extend(record["tokens"])
        if len(record["tokens"]) == 0:
            empty_count += 1

    unique_tokens = set(all_train_tokens)
    avg_tokens = (
        len(all_train_tokens) / len(train_processed)
        if train_processed else 0
    )

    print(f"\n  Training set token statistics:")
    print(f"    Total tokens           : {len(all_train_tokens)}")
    print(f"    Unique tokens          : {len(unique_tokens)}")
    print(f"    Avg tokens per sample  : {avg_tokens:.1f}")
    print(f"    Empty token lists      : {empty_count}")

    return train_processed, test_processed


# ─────────────────────────────────────────────
# Phase 3
# ─────────────────────────────────────────────

def verify_phase_3(train_processed, test_processed):
    """Build vocabulary, vectorize data, and encode labels."""
    print("\n" + "=" * 65)
    print(" PHASE 3 — Feature Extraction")
    print("=" * 65)

    # Step 1: Vocabulary
    print("\n[STEP 1] Building vocabulary...")
    vocab = Vocabulary(min_freq=2, max_size=None)
    vocab.build_from_dataset(train_processed)
    vocab.print_summary()

    # Step 2: Bag-of-Words
    print("\n[STEP 2] Bag-of-Words vectorization...")
    bow_vectorizer = BagOfWordsVectorizer(vocab)
    X_train_bow    = bow_vectorizer.transform_dataset(train_processed)
    X_test_bow     = bow_vectorizer.transform_dataset(test_processed)

    print(f"  Training BoW matrix shape   : {X_train_bow.shape}")
    print(f"  Testing BoW matrix shape    : {X_test_bow.shape}")
    print(f"  BoW matrix dtype            : {X_train_bow.dtype}")
    print(f"  Sample BoW vector (first 5) : {X_train_bow[0][:5]}")
    print(f"  BoW sparsity (% zeros)      : "
          f"{(X_train_bow == 0).sum() / X_train_bow.size * 100:.1f}%")

    # Step 3: TF-IDF
    print("\n[STEP 3] TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(vocab)
    tfidf_vectorizer.fit_dataset(train_processed)
    X_train_tfidf = tfidf_vectorizer.transform_dataset(train_processed)
    X_test_tfidf  = tfidf_vectorizer.transform_dataset(test_processed)

    print(f"  Training TF-IDF matrix shape   : {X_train_tfidf.shape}")
    print(f"  Testing TF-IDF matrix shape    : {X_test_tfidf.shape}")
    print(f"  TF-IDF matrix dtype            : {X_train_tfidf.dtype}")
    print(f"  Sample TF-IDF vector (first 5) : {X_train_tfidf[0][:5]}")
    print(f"  TF-IDF sparsity (% zeros)      : "
          f"{(X_train_tfidf == 0).sum() / X_train_tfidf.size * 100:.1f}%")

    print(f"\n  Comparison for sample 0:")
    print(f"    BoW vector norm    : {np.linalg.norm(X_train_bow[0]):.2f}")
    print(f"    TF-IDF vector norm : {np.linalg.norm(X_train_tfidf[0]):.2f}")

    # Step 4: Label encoding
    print("\n[STEP 4] Label encoding...")
    label_encoder = LabelEncoder(num_classes=NUM_CLASSES)
    y_train = label_encoder.encode_dataset(train_processed)
    y_test  = label_encoder.encode_dataset(test_processed)

    print(f"  Training labels shape    : {y_train.shape}")
    print(f"  Testing labels shape     : {y_test.shape}")
    print(f"  Labels dtype             : {y_train.dtype}")
    print(f"  Sample one-hot (label 0) : {y_train[0]}")

    original_label = train_processed[0]["label"]
    encoded        = label_encoder.encode_single(original_label)
    decoded        = label_encoder.decode_single(encoded)
    print(f"\n  Round-trip test:")
    print(f"    Original label : {original_label}")
    print(f"    Encoded vector : {encoded}")
    print(f"    Decoded label  : {decoded}")
    print(f"    Match          : {original_label == decoded}")

    print("\n" + "=" * 65)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"  ✓ Vocabulary built          : {vocab.size} tokens")
    print(f"  ✓ BoW vectors created       : "
          f"{X_train_bow.shape[0]} train, {X_test_bow.shape[0]} test")
    print(f"  ✓ TF-IDF vectors created    : "
          f"{X_train_tfidf.shape[0]} train, {X_test_tfidf.shape[0]} test")
    print(f"  ✓ Labels encoded            : "
          f"{y_train.shape[0]} train, {y_test.shape[0]} test")
    print(f"  ✓ Feature dimension         : {X_train_tfidf.shape[1]}")
    print(f"  ✓ Output dimension          : {y_train.shape[1]}")
    print("=" * 65)

    return vocab, X_train_tfidf, X_test_tfidf, y_train, y_test


# ─────────────────────────────────────────────
# Phase 4 — Math utilities & NN components
# ─────────────────────────────────────────────

def _run_component_suite(label: str, module_path: str):
    """
    Import a test module, execute run_all_tests() with stdout
    suppressed, and return (label, passed_bool, fail_count).
    """
    import importlib
    import io
    import contextlib

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        return label, False, f"import error: {exc}"

    if hasattr(mod, "_failures"):
        mod._failures.clear()

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.run_all_tests()
    except Exception as exc:
        return label, False, f"runtime error: {exc}"

    fail_count = len(getattr(mod, "_failures", [None]))
    return label, fail_count == 0, fail_count


def verify_phase_4():
    print("\n" + "=" * 65)
    print(" PHASE 4 — Math Utilities & Neural-Network Components")
    print("=" * 65)

    components = [
        ("matrix_ops   (utils/matrix_ops)",  "utils.matrix_ops_test"),
        ("activations  (model/activations)", "model.activations_test"),
        ("losses       (model/losses)",      "model.losses_test"),
        ("DenseLayer   (model/layers)",      "model.layers_test"),
        ("Network      (model/network)",     "model.network_test"),
    ]

    results = [_run_component_suite(lbl, mod) for lbl, mod in components]

    print()
    for label, passed, detail in results:
        mark   = "✓" if passed else "✗"
        status = "PASS" if passed else f"FAIL ({detail} failed)"
        print(f"  {mark} {label:<44} {status}")

    n_pass  = sum(1 for _, p, _ in results if p)
    n_total = len(results)

    print("\n" + "=" * 65)
    print("PHASE 4 VERIFICATION SUMMARY")
    print("=" * 65)
    if n_pass == n_total:
        print(f"  ✓ All {n_total} component suites passed.")
    else:
        print(f"  ✗ {n_total - n_pass} of {n_total} component suites FAILED.")
    print("=" * 65)

    return n_pass == n_total


# ─────────────────────────────────────────────
# Phase 5 — Training infrastructure
# ─────────────────────────────────────────────

def verify_phase_5():
    print("\n" + "=" * 65)
    print(" PHASE 5 — Training Infrastructure")
    print("=" * 65)

    components = [
        ("Trainer  (training/trainer)", "training.trainer_test"),
    ]

    results = [_run_component_suite(lbl, mod) for lbl, mod in components]

    print()
    for label, passed, detail in results:
        mark   = "✓" if passed else "✗"
        status = "PASS" if passed else f"FAIL ({detail} failed)"
        print(f"  {mark} {label:<44} {status}")

    n_pass  = sum(1 for _, p, _ in results if p)
    n_total = len(results)

    print("\n" + "=" * 65)
    print("PHASE 5 VERIFICATION SUMMARY")
    print("=" * 65)
    if n_pass == n_total:
        print(f"  ✓ All {n_total} component suites passed.")
    else:
        print(f"  ✗ {n_total - n_pass} of {n_total} component suites FAILED.")
    print("=" * 65)

    return n_pass == n_total


# ─────────────────────────────────────────────
# Phase 6 — Inference pipeline
# ─────────────────────────────────────────────

def verify_phase_6():
    print("\n" + "=" * 65)
    print(" PHASE 6 — Inference Pipeline & Persistence")
    print("=" * 65)

    components = [
        ("Persistence  (inference/persistence)", "inference.persistence_test"),
        ("Pipeline     (inference/pipeline)",    "inference.pipeline_test"),
    ]

    results = [_run_component_suite(lbl, mod) for lbl, mod in components]

    print()
    for label, passed, detail in results:
        mark   = "✓" if passed else "✗"
        status = "PASS" if passed else f"FAIL ({detail} failed)"
        print(f"  {mark} {label:<44} {status}")

    n_pass  = sum(1 for _, p, _ in results if p)
    n_total = len(results)

    print("\n" + "=" * 65)
    print("PHASE 6 VERIFICATION SUMMARY")
    print("=" * 65)
    if n_pass == n_total:
        print(f"  ✓ All {n_total} component suites passed.")
    else:
        print(f"  ✗ {n_total - n_pass} of {n_total} component suites FAILED.")
    print("=" * 65)

    return n_pass == n_total


# ─────────────────────────────────────────────
# CLI chatbot
# ─────────────────────────────────────────────

_DIVIDER      = "=" * 65
_THIN_DIVIDER = "-" * 65
_BAR_WIDTH    = 30


def _probability_bar(probability: float, width: int = _BAR_WIDTH) -> str:
    """Render a filled ASCII bar proportional to `probability`."""
    filled = round(probability * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _render_result(result: dict, top_k: list) -> str:
    """
    Format a full prediction result for terminal display.

    Parameters
    ----------
    result : dict   — primary prediction from pipeline.predict()
    top_k  : list   — ranked list from pipeline.predict_top_k()
    """
    lines = [
        "",
        _DIVIDER,
        " PREDICTION",
        _THIN_DIVIDER,
        f"  Intent     : {result['category']}",
        f"  Label      : {result['label']}",
        f"  Description: {result['description']}",
        f"  Confidence : {result['confidence'] * 100:>6.2f}%  "
          f"{_probability_bar(result['confidence'])}",
        _THIN_DIVIDER,
        " TOP-3 INTENTS",
        _THIN_DIVIDER,
    ]

    for rank, entry in enumerate(top_k, start=1):
        bar  = _probability_bar(entry["probability"])
        pct  = f"{entry['probability'] * 100:>6.2f}%"
        cat  = entry["category"]
        lines.append(f"  {rank}.  {cat:<22}  {pct}  {bar}")

    lines += [
        _THIN_DIVIDER,
        f"  Tokens     : {result['tokens']}",
        _DIVIDER,
        "",
    ]
    return "\n".join(lines)


def run_chatbot():
    """
    Instantiate the InferencePipeline (loading a saved bundle or
    training from scratch) and enter the interactive REPL.
    """
    from inference.pipeline import InferencePipeline

    print("\n" + _DIVIDER)
    print(" INTENT CLASSIFIER — Interactive CLI")
    print(_DIVIDER)
    print("  Type a message to classify its intent.")
    print("  Commands:  'quit' / 'exit' / 'q'  →  exit")
    print(_DIVIDER)

    # Pipeline construction handles load-or-train transparently.
    pipeline = InferencePipeline(verbose=True)

    print(f"\n  {pipeline}")
    print(f"\n  Ready.  Awaiting input …\n")

    while True:
        try:
            raw = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C → clean exit
            print("\n\n  Session ended.")
            break

        if not raw:
            continue

        if raw.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye.")
            break

        try:
            result = pipeline.predict(raw)
            top_k  = pipeline.predict_top_k(raw, k=3)
            print(_render_result(result, top_k))
        except Exception as exc:  # noqa: BLE001 — never crash the REPL
            print(f"\n  [error] {exc}\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    # ── Phase 1: Data loading ──────────────────────────────
    data, train_data, test_data = verify_phase_1()

    # ── Phase 2: Preprocessing ────────────────────────────
    train_processed, test_processed = verify_phase_2(train_data, test_data)

    # ── Phase 3: Feature extraction ───────────────────────
    vocab, X_train, X_test, y_train, y_test = verify_phase_3(
        train_processed, test_processed
    )

    # ── Phase 4: Math utilities + NN components ───────────
    verify_phase_4()

    # ── Phase 5: Training infrastructure ──────────────────
    verify_phase_5()

    # ── Phase 6: Inference pipeline & persistence ─────────
    verify_phase_6()

    # ── Interactive chatbot ───────────────────────────────
    run_chatbot()


if __name__ == "__main__":
    main()