"""
main.py — Entry point for testing Phase 2 implementation.
"""

from config import DATA_PATH, TEST_RATIO, RANDOM_SEED
from data.data_loader import load_data, print_data_summary
from training.data_split import stratified_split, print_split_summary
from preprocessing.pipeline import PreprocessingPipeline


def test_preprocessing():
    """
    Test the preprocessing pipeline with sample inputs and real data.
    """
    
    print("\n" + "=" * 75)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 75)
    
    # ── Test with manual examples ─────────────
    pipeline = PreprocessingPipeline(use_stopwords=True, use_stemming=True)
    
    test_cases = [
        "Hello! How are you doing today?",
        "I have 25 cats and 3 dogs.",
        "Thanks so much for your help!",
        "Can you tell me what you're capable of?",
        "Goodbye, see you later!",
        "The running cats are jumping quickly over boxes."
    ]
    
    print("\nManual Test Cases:")
    print("-" * 75)
    for text in test_cases:
        tokens = pipeline.process(text)
        print(f"Input : {text}")
        print(f"Output: {tokens}")
        print()
    
    # ── Test with real dataset ────────────────
    print("-" * 75)
    print("Processing full dataset...")
    print("-" * 75)
    
    data = load_data(DATA_PATH)
    texts = [record["text"] for record in data]
    
    # Process entire dataset
    processed = pipeline.process_batch(texts)
    
    # Show statistics
    total_tokens = sum(len(tokens) for tokens in processed)
    avg_tokens = total_tokens / len(processed)
    empty_count = sum(1 for tokens in processed if len(tokens) == 0)
    
    print(f"\nDataset Processing Results:")
    print(f"  Total samples       : {len(processed)}")
    print(f"  Total tokens        : {total_tokens}")
    print(f"  Avg tokens/sample   : {avg_tokens:.2f}")
    print(f"  Empty after process : {empty_count}")
    
    # Show sample transformations from each class
    print("\nSample Transformations (one per class):")
    print("-" * 75)
    shown_labels = set()
    for i, record in enumerate(data):
        label = record["label"]
        if label not in shown_labels and len(processed[i]) > 0:
            shown_labels.add(label)
            print(f"[Label {label}]")
            print(f"  Original : {record['text'][:70]}")
            print(f"  Processed: {processed[i]}")
            print()
        if len(shown_labels) == 12:
            break
    
    print("=" * 75)
    
    return processed


def main():
    # ── Phase 1 verification ──────────────────
    print("\n" + "=" * 75)
    print("PHASE 1: DATA LOADING AND SPLITTING")
    print("=" * 75)
    
    data = load_data(DATA_PATH)
    print_data_summary(data)
    
    train_data, test_data = stratified_split(
        data,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )
    print_split_summary(train_data, test_data)
    
    # ── Phase 2 verification ──────────────────
    print("\n" + "=" * 75)
    print("PHASE 2: TEXT PREPROCESSING")
    print("=" * 75)
    
    test_preprocessing()


if __name__ == "__main__":
    main()