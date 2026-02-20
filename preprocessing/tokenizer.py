"""
Text tokenization utilities.
Splits a normalized text string into a list of individual tokens.
"""

import re


def tokenize_whitespace(text):
    """
    Baseline tokenizer — splits on whitespace boundaries.

    Parameters
    ----------
    text : str
        Normalized text.

    Returns
    -------
    list of str
        Token list. Empty list for empty / whitespace-only input.
    """
    if not text or not text.strip():
        return []
    return text.split()


def tokenize_regex(text):
    """
    Regex-based tokenizer — extracts contiguous sequences of
    lowercase letters and digits.  More robust than whitespace
    splitting against residual special characters or edge cases.

    Parameters
    ----------
    text : str
        Normalized text.

    Returns
    -------
    list of str
        Token list. Empty list for empty / whitespace-only input.
    """
    if not text or not text.strip():
        return []
    return re.findall(r"[a-z0-9]+", text)


def tokenize(text, method="regex"):
    """
    Tokenize text using the specified method.

    Parameters
    ----------
    text : str
        Pre-normalized text to tokenize.
    method : str
        'whitespace' — simple str.split().
        'regex'      — regex extraction of alphanumeric sequences.

    Returns
    -------
    list of str
        Token list.
    """
    if method == "whitespace":
        return tokenize_whitespace(text)
    elif method == "regex":
        return tokenize_regex(text)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")