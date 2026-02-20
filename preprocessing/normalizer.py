"""
Text normalization module.
Cleans and standardizes raw text input before tokenization.
"""

import re
import string


def normalize_text(text):
    """
    Normalize raw text through a series of cleaning operations.
    
    Operations applied in order:
    1. Lowercase conversion
    2. Unicode/accent handling
    3. Punctuation removal
    4. Number handling (replace with token)
    5. Whitespace normalization
    
    Parameters
    ----------
    text : str
        Raw input text.
    
    Returns
    -------
    str
        Normalized text.
    """
    
    if not isinstance(text, str):
        text = str(text)
    
    # ── Step 1: Lowercase ─────────────────────
    text = text.lower()
    
    # ── Step 2: Unicode/Accent handling ───────
    # Encode to ASCII, ignoring characters that can't be represented
    # This strips accents: café → cafe
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # ── Step 3: Punctuation removal ───────────
    # Remove all punctuation characters
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # ── Step 4: Number handling ───────────────
    # Replace all digit sequences with a <NUM> token
    # This preserves that a number existed without the specific value
    # "I have 25 cats" → "I have <NUM> cats"
    text = re.sub(r'\d+', '<NUM>', text)
    
    # ── Step 5: Whitespace normalization ──────
    # Collapse multiple spaces, tabs, newlines into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_batch(texts):
    """
    Normalize a batch of texts.
    
    Parameters
    ----------
    texts : list of str
        List of raw text strings.
    
    Returns
    -------
    list of str
        List of normalized texts.
    """
    return [normalize_text(text) for text in texts]