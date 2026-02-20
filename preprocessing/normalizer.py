"""
Text normalization utilities.
Cleans raw text prior to tokenization through a sequence of
transformations: lowercasing, accent removal, punctuation stripping,
number handling, and whitespace normalization.
"""

import re
import string

# ──────────────────────────────────────────────
# Accented character → ASCII equivalent mapping
# Only lowercase forms needed since lowercasing
# is applied before accent removal in the pipeline.
# ──────────────────────────────────────────────
ACCENT_MAP = {
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
    "è": "e", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "ý": "y", "ÿ": "y",
    "ñ": "n", "ç": "c",
}

# O(1) lookup set for punctuation characters
PUNCTUATION_SET = set(string.punctuation)


def to_lowercase(text):
    """Convert entire string to lowercase."""
    return text.lower()


def remove_accents(text):
    """Replace accented characters with their ASCII equivalents.
    Characters not found in the mapping are left unchanged."""
    return "".join(ACCENT_MAP.get(char, char) for char in text)


def remove_punctuation(text):
    """Strip all characters present in string.punctuation."""
    return "".join(char for char in text if char not in PUNCTUATION_SET)


def normalize_whitespace(text):
    """Collapse all whitespace sequences (spaces, tabs, newlines)
    into single spaces and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def handle_numbers(text, strategy="remove"):
    """
    Process numeric characters according to the chosen strategy.

    Parameters
    ----------
    text : str
        Input text potentially containing digits.
    strategy : str
        'remove'  — delete all digit sequences entirely.
        'replace' — replace each digit sequence with the token NUM.
        'keep'    — leave digits unchanged.

    Returns
    -------
    str
        Text with numbers handled according to strategy.

    Rationale for default 'remove': in intent classification the
    presence of specific numbers rarely affects the intent category.
    "I have 3 questions" carries the same intent as "I have questions".
    """
    if strategy == "remove":
        return re.sub(r"\d+", "", text)
    elif strategy == "replace":
        return re.sub(r"\d+", "NUM", text)
    elif strategy == "keep":
        return text
    else:
        raise ValueError(f"Unknown number strategy: {strategy}")


def normalize(text, number_strategy="remove"):
    """
    Full normalization pipeline applied in order:
        1. Lowercase conversion
        2. Accent / unicode replacement
        3. Punctuation removal
        4. Number handling
        5. Whitespace normalization

    Parameters
    ----------
    text : str
        Raw input text.
    number_strategy : str
        Passed to handle_numbers().

    Returns
    -------
    str
        Cleaned, normalized text ready for tokenization.
    """
    text = to_lowercase(text)
    text = remove_accents(text)
    text = remove_punctuation(text)
    text = handle_numbers(text, strategy=number_strategy)
    text = normalize_whitespace(text)
    return text