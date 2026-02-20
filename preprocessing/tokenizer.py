"""
Tokenization module.
Splits normalized text into a list of individual tokens.
"""

import re


def tokenize(text):
    """
    Tokenize a normalized text string into a list of tokens.
    
    Uses a regex pattern to extract alphanumeric sequences and
    special tokens (like <NUM>).
    
    Parameters
    ----------
    text : str
        Normalized text string.
    
    Returns
    -------
    list of str
        List of tokens.
    
    Examples
    --------
    >>> tokenize("hello world")
    ['hello', 'world']
    
    >>> tokenize("i have <NUM> cats")
    ['i', 'have', '<NUM>', 'cats']
    
    >>> tokenize("")
    []
    """
    
    if not text or not isinstance(text, str):
        return []
    
    # Extract all alphanumeric sequences and special tokens
    # Pattern matches: letters, numbers, and angle-bracketed tokens
    tokens = re.findall(r'<[A-Z]+>|[a-z0-9]+', text)
    
    return tokens


def tokenize_batch(texts):
    """
    Tokenize a batch of texts.
    
    Parameters
    ----------
    texts : list of str
        List of normalized text strings.
    
    Returns
    -------
    list of list of str
        List of token lists.
    """
    return [tokenize(text) for text in texts]