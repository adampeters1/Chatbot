"""
main.py — Entry point for verifying Phase 1 and Phase 2 implementation.
"""

from config import DATA_PATH, TEST_RATIO, RANDOM_SEED, LABEL_MAP, PREPROCESSING
from utils.data_loader import load_data, print_data_summary
from training.data_split import stratified_split, print_split_summary
from preprocessing import preprocess, preprocess_dataset
from preprocessing.stemmer import stem


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
    """Run preprocessing pipeline and display verification output."""
    print("\n" + "=" * 65)
    print(" PHASE 2 — Preprocessing Pipeline")
    print("=" * 65)

    # ── 1. Stemmer unit tests ─────────────────
    print("\nSTEMMER VERIFICATION:")
    print("-" * 45)

    stem_tests = [
        ("running",   "run"),
        ("cats",      "cat"),
        ("played",    "play"),
        ("babies",    "baby"),
        ("grasses",   "grass"),
        ("boxes",     "box"),
        ("action",    "act"),
        ("helpful",   "help"),
        ("quickly",   "quick"),
        ("sadness",   "sad"),
        ("movement",  "move"),
        ("readable",  "read"),
        ("stopped",   "stop"),
        ("paintings", "paint"),
        ("the",       "the"),
    ]

    passed = 0
    for word, expected in stem_tests:
        result = stem(word)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {status}  stem('{word}') → '{result}'  (expected '{expected}')")

    print(f"\n  Results: {passed}/{len(stem_tests)} passed")

    # ── 2. Full pipeline examples ─────────────
    print("\n\nFULL PIPELINE EXAMPLES:")
    print("-" * 65)

    sample_texts = [
        "Hello! How are you doing today?",
        "Thank you SO much for your help!!!",
        "Can you speak in French, please?",
        "I DON'T think that's right...",
        "What features do you have? Can you do múltiple things??",
        "The résponse was    really    terrible!  Fix it.",
        "I have 3 suggestions for you to improve by 100%",
    ]

    for text in sample_texts:
        tokens = preprocess(text, **PREPROCESSING)
        print(f"  INPUT:  \"{text}\"")
        print(f"  OUTPUT: {tokens}")
        print()

    # ── 3. Process full datasets ──────────────
    print("DATASET PREPROCESSING:")
    print("-" * 65)

    train_processed = preprocess_dataset(train_data, **PREPROCESSING)
    test_processed = preprocess_dataset(test_data, **PREPROCESSING)

    print(f"  Training samples processed : {len(train_processed)}")
    print(f"  Testing samples processed  : {len(test_processed)}")

    # ── 4. Token statistics ───────────────────
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

    # ── 5. Samples per class after processing ─
    print(f"\n  Sample processed records (one per class):")
    print("-" * 65)

    shown_labels = set()
    for record in train_processed:
        label = record["label"]
        if label not in shown_labels:
            shown_labels.add(label)
            category = LABEL_MAP[label]["category"]
            display_text = record["text"]
            if len(display_text) > 50:
                display_text = display_text[:47] + "..."
            print(f"  [{label:>2}] {category:<20} {str(record['tokens'])[:60]}")
            print(f"       original: \"{display_text}\"")

        if len(shown_labels) == len(LABEL_MAP):
            break

    print("=" * 65)

    return train_processed, test_processed

def verify_vocabulary_builder(train_processed):
    """Test the Vocabulary class with the preprocessed training data."""
    from features import Vocabulary

    print("\n" + "=" * 65)
    print(" PHASE 3 — Vocabulary Builder")
    print("=" * 65)

    # ── Test 1: Build with default settings ──────
    print("\n[TEST 1] Building vocabulary with min_freq=2, no max_size")
    vocab = Vocabulary(min_freq=2, max_size=None)
    vocab.build_from_dataset(train_processed)
    vocab.print_summary()
    vocab.print_most_common(n=20)
    vocab.print_least_common(n=10)

    # ── Test 2: Verify token lookup ──────────────
    print("\n[TEST 2] Token lookup verification")
    print("-" * 50)
    test_tokens = ["hello", "thank", "help", "bot", "nonexistenttoken12345"]
    for token in test_tokens:
        idx = vocab.get_index(token)
        if idx is not None:
            reverse = vocab.get_token(idx)
            in_vocab = token in vocab
            print(f"  '{token}' → index {idx} → '{reverse}' | in vocab: {in_vocab}")
        else:
            print(f"  '{token}' → OUT OF VOCABULARY")

    # ── Test 3: Build with max_size constraint ───
    print("\n[TEST 3] Building vocabulary with min_freq=2, max_size=500")
    vocab_limited = Vocabulary(min_freq=2, max_size=500)
    vocab_limited.build_from_dataset(train_processed)
    vocab_limited.print_summary()

    # ── Test 4: Build with higher min_freq ───────
    print("\n[TEST 4] Building vocabulary with min_freq=5, no max_size")
    vocab_strict = Vocabulary(min_freq=5, max_size=None)
    vocab_strict.build_from_dataset(train_processed)
    vocab_strict.print_summary()

    print("\n" + "=" * 65)

    return vocab

def main():
    # Phase 1
    data, train_data, test_data = verify_phase_1()

    # Phase 2
    train_processed, test_processed = verify_phase_2(train_data, test_data)

    # Phase 3 - Vocabulary
    vocab = verify_vocabulary_builder(train_processed)


    
if __name__ == "__main__":
    main()