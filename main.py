"""
main.py — Full Phase 3 verification
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


def verify_phase_2(train_data, test_data):
    """Run preprocessing pipeline."""
    print("\n" + "=" * 65)
    print(" PHASE 2 — Preprocessing Pipeline")
    print("=" * 65)

    train_processed = preprocess_dataset(train_data, **PREPROCESSING)
    test_processed = preprocess_dataset(test_data, **PREPROCESSING)

    print(f"  Training samples processed : {len(train_processed)}")
    print(f"  Testing samples processed  : {len(test_processed)}")

    all_train_tokens = []
    empty_count = 0
    for record in train_processed:
        all_train_tokens.extend(record["tokens"])
        if len(record["tokens"]) == 0:
            empty_count += 1

    unique_tokens = set(all_train_tokens)
    avg_tokens = len(all_train_tokens) / len(train_processed) if train_processed else 0

    print(f"\n  Training set token statistics:")
    print(f"    Total tokens           : {len(all_train_tokens)}")
    print(f"    Unique tokens          : {len(unique_tokens)}")
    print(f"    Avg tokens per sample  : {avg_tokens:.1f}")
    print(f"    Empty token lists      : {empty_count}")

    return train_processed, test_processed


def verify_phase_3(train_processed, test_processed):
    """Build vocabulary, vectorize data, and encode labels."""
    print("\n" + "=" * 65)
    print(" PHASE 3 — Feature Extraction")
    print("=" * 65)

    # ── Step 1: Build Vocabulary ──────────────────
    print("\n[STEP 1] Building vocabulary...")
    vocab = Vocabulary(min_freq=2, max_size=None)
    vocab.build_from_dataset(train_processed)
    vocab.print_summary()

    # ── Step 2: Bag-of-Words Vectorization ────────
    print("\n[STEP 2] Bag-of-Words vectorization...")
    bow_vectorizer = BagOfWordsVectorizer(vocab)
    
    X_train_bow = bow_vectorizer.transform_dataset(train_processed)
    X_test_bow = bow_vectorizer.transform_dataset(test_processed)
    
    print(f"  Training BoW matrix shape   : {X_train_bow.shape}")
    print(f"  Testing BoW matrix shape    : {X_test_bow.shape}")
    print(f"  BoW matrix dtype            : {X_train_bow.dtype}")
    print(f"  Sample BoW vector (first 5) : {X_train_bow[0][:5]}")
    print(f"  BoW sparsity (% zeros)      : {(X_train_bow == 0).sum() / X_train_bow.size * 100:.1f}%")

    # ── Step 3: TF-IDF Vectorization ──────────────
    print("\n[STEP 3] TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(vocab)
    tfidf_vectorizer.fit_dataset(train_processed)
    
    X_train_tfidf = tfidf_vectorizer.transform_dataset(train_processed)
    X_test_tfidf = tfidf_vectorizer.transform_dataset(test_processed)
    
    print(f"  Training TF-IDF matrix shape   : {X_train_tfidf.shape}")
    print(f"  Testing TF-IDF matrix shape    : {X_test_tfidf.shape}")
    print(f"  TF-IDF matrix dtype            : {X_train_tfidf.dtype}")
    print(f"  Sample TF-IDF vector (first 5) : {X_train_tfidf[0][:5]}")
    print(f"  TF-IDF sparsity (% zeros)      : {(X_train_tfidf == 0).sum() / X_train_tfidf.size * 100:.1f}%")
    
    # Compare BoW vs TF-IDF for same sample
    print(f"\n  Comparison for sample 0:")
    print(f"    BoW vector norm    : {np.linalg.norm(X_train_bow[0]):.2f}")
    print(f"    TF-IDF vector norm : {np.linalg.norm(X_train_tfidf[0]):.2f}")

    # ── Step 4: Label Encoding ────────────────────
    print("\n[STEP 4] Label encoding...")
    label_encoder = LabelEncoder(num_classes=NUM_CLASSES)
    
    y_train = label_encoder.encode_dataset(train_processed)
    y_test = label_encoder.encode_dataset(test_processed)
    
    print(f"  Training labels shape    : {y_train.shape}")
    print(f"  Testing labels shape     : {y_test.shape}")
    print(f"  Labels dtype             : {y_train.dtype}")
    print(f"  Sample one-hot (label 0) : {y_train[0]}")
    
    # Test encoding/decoding round-trip
    original_label = train_processed[0]["label"]
    encoded = label_encoder.encode_single(original_label)
    decoded = label_encoder.decode_single(encoded)
    print(f"\n  Round-trip test:")
    print(f"    Original label : {original_label}")
    print(f"    Encoded vector : {encoded}")
    print(f"    Decoded label  : {decoded}")
    print(f"    Match          : {original_label == decoded}")

    # ── Step 5: Verification Summary ──────────────
    print("\n" + "=" * 65)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"  ✓ Vocabulary built          : {vocab.size} tokens")
    print(f"  ✓ BoW vectors created       : {X_train_bow.shape[0]} train, {X_test_bow.shape[0]} test")
    print(f"  ✓ TF-IDF vectors created    : {X_train_tfidf.shape[0]} train, {X_test_tfidf.shape[0]} test")
    print(f"  ✓ Labels encoded            : {y_train.shape[0]} train, {y_test.shape[0]} test")
    print(f"  ✓ Feature dimension         : {X_train_tfidf.shape[1]}")
    print(f"  ✓ Output dimension          : {y_train.shape[1]}")
    print("=" * 65)

    return vocab, X_train_tfidf, X_test_tfidf, y_train, y_test


def main():
    # Phase 1: Data loading
    data, train_data, test_data = verify_phase_1()

    # Phase 2: Preprocessing
    train_processed, test_processed = verify_phase_2(train_data, test_data)

    # Phase 3: Feature extraction
    vocab, X_train, X_test, y_train, y_test = verify_phase_3(
        train_processed, test_processed
    )


if __name__ == "__main__":
    main()