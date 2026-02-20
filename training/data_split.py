"""
Dataset splitting utilities.
Provides stratified train/test splitting to maintain class distribution.
"""

import random
from collections import defaultdict

from config import LABEL_MAP


def stratified_split(data, test_ratio=0.2, random_seed=42):
    """
    Split data into training and testing sets using stratified sampling.

    Groups all records by their label first, then splits each group
    independently according to the given ratio. This ensures that every
    class is proportionally represented in both the train and test sets.

    Parameters
    ----------
    data : list of dict
        Each dict has keys 'text' (str) and 'label' (int).
    test_ratio : float
        Proportion of data to reserve for testing (0.0 to 1.0).
    random_seed : int or None
        Seed for reproducibility. Pass None for non-deterministic splits.

    Returns
    -------
    train_set : list of dict
        Training records.
    test_set : list of dict
        Testing records.
    """

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            f"test_ratio must be between 0 and 1 exclusive, got {test_ratio}"
        )

    if random_seed is not None:
        random.seed(random_seed)

    # ── Group records by label ────────────────
    groups = defaultdict(list)
    for record in data:
        groups[record["label"]].append(record)

    train_set = []
    test_set = []

    # ── Split each group independently ────────
    for label in sorted(groups.keys()):
        records = groups[label]
        random.shuffle(records)

        split_index = int(len(records) * (1 - test_ratio))

        # Ensure at least 1 sample in each set when possible
        if split_index == 0 and len(records) > 1:
            split_index = 1
        elif split_index == len(records) and len(records) > 1:
            split_index = len(records) - 1

        train_set.extend(records[:split_index])
        test_set.extend(records[split_index:])

    # ── Shuffle the combined sets ─────────────
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


def print_split_summary(train_data, test_data):
    """
    Print a comparison of class distributions across the train/test split.
    Useful for verifying that stratification worked correctly.

    Parameters
    ----------
    train_data : list of dict
        Training records.
    test_data : list of dict
        Testing records.
    """

    # ── Count per label in each set ───────────
    train_counts = {}
    for record in train_data:
        label = record["label"]
        train_counts[label] = train_counts.get(label, 0) + 1

    test_counts = {}
    for record in test_data:
        label = record["label"]
        test_counts[label] = test_counts.get(label, 0) + 1

    total_train = len(train_data)
    total_test = len(test_data)
    total_all = total_train + total_test

    # ── Display comparison table ──────────────
    print("\n" + "=" * 75)
    print("TRAIN / TEST SPLIT SUMMARY")
    print("=" * 75)
    print(f"  Total samples  : {total_all}")
    print(f"  Training set   : {total_train}  ({total_train/total_all*100:.1f}%)")
    print(f"  Testing set    : {total_test}  ({total_test/total_all*100:.1f}%)")
    print("-" * 75)
    print(f"  {'Label':<7} {'Category':<22} {'Train':<10} {'Test':<10} {'Train %':<10} {'Test %':<10}")
    print("-" * 75)

    all_labels = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

    for label in all_labels:
        category = LABEL_MAP[label]["category"]
        tr_count = train_counts.get(label, 0)
        te_count = test_counts.get(label, 0)

        tr_pct = (tr_count / total_train * 100) if total_train > 0 else 0
        te_pct = (te_count / total_test * 100) if total_test > 0 else 0

        print(
            f"  {label:<7} {category:<22} {tr_count:<10} {te_count:<10} "
            f"{tr_pct:<9.1f}% {te_pct:<9.1f}%"
        )

    print("=" * 75)