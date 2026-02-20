import csv
import os

# Label to category mapping based on your dataset
LABEL_CATEGORY_MAP = {
    0: "greeting",
    1: "farewell",
    2: "thank_you",
    3: "affirmation",
    4: "negation",
    5: "small_talk",
    6: "bot_capabilities",
    7: "feedback_positive",
    8: "feedback_negative",
    9: "clarification",
    10: "suggestion",
    11: "language_change"
}

CATEGORY_DESCRIPTION_MAP = {
    "greeting": "Greeting or saying hello.",
    "farewell": "Saying goodbye or farewell.",
    "thank_you": "Expressing gratitude or thanks.",
    "affirmation": "Agreeing or confirming something.",
    "negation": "Disagreeing or denying something.",
    "small_talk": "Engaging in casual or light conversation with no specific purpose.",
    "bot_capabilities": "Inquiries about the bot's features or abilities.",
    "feedback_positive": "Providing positive feedback about the bot, service, or experience.",
    "feedback_negative": "Providing negative feedback about the bot, service, or experience.",
    "clarification": "Asking for clarification or more information about a previous statement or question.",
    "suggestion": "Offering a suggestion or recommendation for improvement.",
    "language_change": "Requesting a change in the language being used by the bot or information about language options."
}


def load_data(filepath):
    """
    Load dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing 'text' and 'label' columns
        
    Returns:
        list: List of dictionaries with 'text' (str) and 'label' (int) keys
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the CSV format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        # Validate required columns
        if 'text' not in csv_reader.fieldnames or 'label' not in csv_reader.fieldnames:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        for row_num, row in enumerate(csv_reader, start=2):  # start=2 to account for header
            # Skip empty or malformed rows
            if not row.get('text') or not row.get('text').strip():
                print(f"Warning: Skipping empty text at row {row_num}")
                continue
            
            if not row.get('label') or not row.get('label').strip():
                print(f"Warning: Skipping empty label at row {row_num}")
                continue
            
            try:
                label = int(row['label'])
                
                # Validate label is in expected range
                if label < 0 or label > 11:
                    print(f"Warning: Label {label} out of range [0-11] at row {row_num}, skipping")
                    continue
                
                data.append({
                    'text': row['text'].strip(),
                    'label': label
                })
                
            except ValueError:
                print(f"Warning: Invalid label value '{row['label']}' at row {row_num}, skipping")
                continue
    
    if not data:
        raise ValueError("No valid data loaded from CSV file")
    
    print(f"Successfully loaded {len(data)} records from {filepath}")
    return data


def get_category_name(label):
    """
    Get the category name for a given label.
    
    Args:
        label (int): Numeric label (0-11)
        
    Returns:
        str: Category name
    """
    return LABEL_CATEGORY_MAP.get(label, "unknown")


def get_category_description(category):
    """
    Get the description for a given category.
    
    Args:
        category (str): Category name
        
    Returns:
        str: Category description
    """
    return CATEGORY_DESCRIPTION_MAP.get(category, "No description available")


def print_data_summary(data):
    """
    Print a summary of the loaded data including label distribution.
    
    Args:
        data (list): List of data dictionaries
    """
    print("\n=== Data Summary ===")
    print(f"Total samples: {len(data)}")
    
    # Count labels
    label_counts = {}
    for item in data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        category = get_category_name(label)
        count = label_counts[label]
        percentage = (count / len(data)) * 100
        print(f"  Label {label} ({category}): {count} samples ({percentage:.2f}%)")
    print("=" * 40 + "\n")