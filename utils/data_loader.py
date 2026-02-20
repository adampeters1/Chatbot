"""
Data loading and validation utilities.
Reads the intent classification CSV and returns structured records.
"""

import pandas as pd
import os

from config import LABEL_MAP, DATA_PATH, NUM_CLASSES


def load_data(filepath=DATA_PATH):
    """
    Load the intent classification dataset from a CSV file.

    Expects a CSV with two columns:
        - 'text':   the raw user utterance (string)
        - 'labels': the integer class label (0 through NUM_CLASSES-1)

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    records : list of dict
        Each dict has keys 'text' (str) and 'label' (int).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing from the CSV.
    """

    # ── Verify file exists ────────────────────
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    # ── Read CSV with pandas ──────────────────
    df = pd.read_csv(filepath)

    # ── Validate expected columns ─────────────
    # Handle common column name variations
    df.columns = df.columns.str.strip().str.lower()

    # Check for the two required columns
    if "text" not in df.columns:
        raise ValueError(
            f"Missing 'text' column. Found columns: {list(df.columns)}"
        )

    # Accept either 'labels' or 'label'
    if "labels" in df.columns:
        label_col = "labels"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(
            f"Missing 'labels'/'label' column. Found columns: {list(df.columns)}"
        )

    # ── Track validation statistics ───────────
    total_rows = len(df)
    rejected_empty = 0
    rejected_label = 0

    # ── Process row by row ────────────────────
    records = []

    for idx, row in df.iterrows():
        text = row["text"]
        label = row[label_col]

        # Reject rows where text is NaN, empty, or purely whitespace
        if pd.isna(text) or str(text).strip() == "":
            rejected_empty += 1
            continue

        # Clean the text value
        text = str(text).strip()

        # Validate label is a valid integer within range
        try:
            label = int(label)
        except (ValueError, TypeError):
            rejected_label += 1
            continue

        if label not in LABEL_MAP:
            rejected_label += 1
            continue

        records.append({
            "text": text,
            "label": label
        })

    # ── Report loading results ────────────────
    print("=" * 55)
    print("DATA LOADING COMPLETE")
    print("=" * 55)
    print(f"  Source file       : {filepath}")
    print(f"  Total rows in CSV : {total_rows}")
    print(f"  Valid records     : {len(records)}")
    print(f"  Rejected (empty)  : {rejected_empty}")
    print(f"  Rejected (label)  : {rejected_label}")
    print("=" * 55)

    if len(records) == 0:
        raise ValueError("No valid records found in dataset.")

    return records


def print_data_summary(data):
    """
    Print a detailed summary of the loaded dataset.
    Shows per-class distribution with category names and sample counts.

    Parameters
    ----------
    data : list of dict
        Each dict has keys 'text' (str) and 'label' (int).
    """

    # ── Count samples per label ───────────────
    label_counts = {}
    for record in data:
        label = record["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    # ── Compute text length statistics ────────
    text_lengths = [len(record["text"].split()) for record in data]
    avg_length = sum(text_lengths) / len(text_lengths)
    min_length = min(text_lengths)
    max_length = max(text_lengths)

    # ── Display summary ───────────────────────
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total samples       : {len(data)}")
    print(f"  Number of classes   : {len(label_counts)}")
    print(f"  Avg words per text  : {avg_length:.1f}")
    print(f"  Min words per text  : {min_length}")
    print(f"  Max words per text  : {max_length}")
    print("-" * 65)
    print(f"  {'Label':<7} {'Category':<22} {'Count':<8} {'Percent':<8} Bar")
    print("-" * 65)

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percent = (count / len(data)) * 100
        category = LABEL_MAP[label]["category"]
        bar = "█" * int(percent)
        print(f"  {label:<7} {category:<22} {count:<8} {percent:<7.1f}% {bar}")

    print("=" * 65)

    # ── Show a sample record from each class ──
    print("\nSAMPLE RECORDS (one per class):")
    print("-" * 65)

    shown_labels = set()
    for record in data:
        label = record["label"]
        if label not in shown_labels:
            shown_labels.add(label)
            category = LABEL_MAP[label]["category"]
            # Truncate long text for display
            display_text = record["text"]
            if len(display_text) > 60:
                display_text = display_text[:57] + "..."
            print(f"  [{label:>2}] {category:<20} \"{display_text}\"")

        if len(shown_labels) == NUM_CLASSES:
            break

    print("-" * 65)