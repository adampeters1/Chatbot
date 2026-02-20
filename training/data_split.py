import random


def stratified_split(data, train_ratio=0.8, test_ratio=0.1, random_seed=42):
    """
    Split data into train, validation, and test sets with stratification.
    Ensures each label is proportionally represented in all sets.
    
    Args:
        data (list): List of dictionaries with 'text' and 'label' keys
        train_ratio (float): Proportion of data for training (default 0.8)
        test_ratio (float): Proportion of data for testing (default 0.1)
        random_seed (int): Random seed for reproducibility (default 42)
        
    Returns:
        tuple: (train_data, val_data, test_data) as lists
        
    Raises:
        ValueError: If ratios don't sum to <= 1.0 or data is empty
    """
    if not data:
        raise ValueError("Cannot split empty data")
    
    val_ratio = 1.0 - train_ratio - test_ratio
    
    if train_ratio + test_ratio + val_ratio > 1.0 or train_ratio <= 0 or test_ratio < 0 or val_ratio < 0:
        raise ValueError(f"Invalid split ratios: train={train_ratio}, test={test_ratio}, val={val_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Group data by label
    label_groups = {}
    for item in data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # Shuffle each group
    for label in label_groups:
        random.shuffle(label_groups[label])
    
    # Split each group proportionally
    train_data = []
    val_data = []
    test_data = []
    
    for label, items in label_groups.items():
        n_items = len(items)
        n_train = int(n_items * train_ratio)
        n_test = int(n_items * test_ratio)
        # Validation gets the remainder to ensure all data is used
        
        train_data.extend(items[:n_train])
        test_data.extend(items[n_train:n_train + n_test])
        val_data.extend(items[n_train + n_test:])
    
    # Final shuffle of the combined sets
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"\n=== Data Split Summary ===")
    print(f"Train set: {len(train_data)} samples ({len(train_data)/len(data)*100:.2f}%)")
    print(f"Validation set: {len(val_data)} samples ({len(val_data)/len(data)*100:.2f}%)")
    print(f"Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.2f}%)")
    print("=" * 40 + "\n")
    
    return train_data, val_data, test_data


def simple_split(data, train_ratio=0.8, random_seed=42):
    """
    Simple train/test split without validation set.
    
    Args:
        data (list): List of dictionaries with 'text' and 'label' keys
        train_ratio (float): Proportion of data for training (default 0.8)
        random_seed (int): Random seed for reproducibility (default 42)
        
    Returns:
        tuple: (train_data, test_data) as lists
    """
    test_ratio = 1.0 - train_ratio
    train_data, _, test_data = stratified_split(data, train_ratio, test_ratio, random_seed)
    
    return train_data, test_data


def verify_split_stratification(train_data, val_data, test_data):
    """
    Verify that the split maintains label proportions across all sets.
    
    Args:
        train_data (list): Training data
        val_data (list): Validation data
        test_data (list): Test data
    """
    def get_label_distribution(data):
        label_counts = {}
        for item in data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        total = len(data)
        return {label: count/total for label, count in label_counts.items()}
    
    train_dist = get_label_distribution(train_data)
    val_dist = get_label_distribution(val_data)
    test_dist = get_label_distribution(test_data)
    
    print("\n=== Stratification Verification ===")
    print(f"{'Label':<6} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
    print("-" * 40)
    
    all_labels = sorted(set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys()))
    for label in all_labels:
        train_pct = train_dist.get(label, 0) * 100
        val_pct = val_dist.get(label, 0) * 100
        test_pct = test_dist.get(label, 0) * 100
        print(f"{label:<6} {train_pct:<10.2f} {val_pct:<10.2f} {test_pct:<10.2f}")
    print("=" * 40 + "\n")