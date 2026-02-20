"""
Stemming module.
Implements a simplified suffix-stripping stemmer based on Porter Stemmer rules.
This is not a complete Porter implementation but captures the most impactful rules.
"""


def stem_word(word):
    """
    Apply stemming rules to a single word.
    
    Rules applied in priority order:
    1. Plural handling
    2. Verb suffixes
    3. Adjective/adverb suffixes
    
    Parameters
    ----------
    word : str
        A single token (assumed to be lowercase).
    
    Returns
    -------
    str
        The stemmed word.
    
    Examples
    --------
    >>> stem_word("running")
    'run'
    
    >>> stem_word("cats")
    'cat'
    
    >>> stem_word("happily")
    'happi'
    """
    
    if len(word) <= 2:
        return word
    
    original = word
    
    # ────────────────────────────────────────────
    # Step A: Plural Handling
    # ────────────────────────────────────────────
    
    # "ies" → "y" (e.g., berries → berry)
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
        return word
    
    # "sses" → "ss" (e.g., stresses → stress)
    if word.endswith("sses"):
        word = word[:-2]
        return word
    
    # "es" → "" (e.g., boxes → box, but only if stem remains valid)
    if word.endswith("es") and len(word) > 3:
        # Only remove if the letter before 'es' is a sibilant or common pattern
        if word[-3] in ['s', 'x', 'z'] or word[-4:-2] in ['ch', 'sh']:
            word = word[:-2]
            return word
    
    # "s" → "" (e.g., cats → cat)
    # Only if not "ss" and stem is long enough
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        word = word[:-1]
        return word
    
    # ────────────────────────────────────────────
    # Step B: Verb Suffixes
    # ────────────────────────────────────────────
    
    # "ing" → "" (e.g., running → run, but need doubling handling)
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        # Check for doubled consonant before "ing" (e.g., running → run)
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            word = stem[:-1]
        else:
            word = stem
        return word
    
    # "ed" → "" (e.g., played → play, stopped → stop)
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        # Check for doubled consonant
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            word = stem[:-1]
        else:
            word = stem
        return word
    
    # "tion" / "sion" → "t" / "s" (e.g., creation → creat)
    if word.endswith("tion") and len(word) > 5:
        word = word[:-3]
        return word
    
    if word.endswith("sion") and len(word) > 5:
        word = word[:-3]
        return word
    
    # ────────────────────────────────────────────
    # Step C: Adjective/Adverb Suffixes
    # ────────────────────────────────────────────
    
    # "ly" → "" (e.g., quickly → quick)
    if word.endswith("ly") and len(word) > 4:
        word = word[:-2]
        return word
    
    # "ness" → "" (e.g., happiness → happi)
    if word.endswith("ness") and len(word) > 5:
        word = word[:-4]
        return word
    
    # "ful" → "" (e.g., beautiful → beauti)
    if word.endswith("ful") and len(word) > 5:
        word = word[:-3]
        return word
    
    # "ment" → "" (e.g., government → govern)
    if word.endswith("ment") and len(word) > 5:
        word = word[:-4]
        return word
    
    # "able" / "ible" → "" (e.g., comfortable → comfort)
    if word.endswith("able") and len(word) > 5:
        word = word[:-4]
        return word
    
    if word.endswith("ible") and len(word) > 5:
        word = word[:-4]
        return word
    
    # "ous" → "" (e.g., dangerous → danger)
    if word.endswith("ous") and len(word) > 5:
        word = word[:-3]
        return word
    
    # "ive" → "" (e.g., active → act)
    if word.endswith("ive") and len(word) > 5:
        word = word[:-3]
        return word
    
    # "er" → "" (e.g., bigger → big)
    # Only apply if doubled consonant pattern
    if word.endswith("er") and len(word) > 4:
        stem = word[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            word = stem[:-1]
            return word
    
    # If no rules applied, return original
    return word


def stem_tokens(tokens, use_stemming=True):
    """
    Apply stemming to a list of tokens.
    
    Parameters
    ----------
    tokens : list of str
        List of tokens.
    use_stemming : bool
        If False, return tokens unchanged. Allows easy toggling.
    
    Returns
    -------
    list of str
        List of stemmed tokens.
    """
    
    if not use_stemming:
        return tokens
    
    return [stem_word(token) for token in tokens]


def stem_tokens_batch(token_lists, use_stemming=True):
    """
    Apply stemming to a batch of token lists.
    
    Parameters
    ----------
    token_lists : list of list of str
        List of token lists.
    use_stemming : bool
        If False, return unchanged.
    
    Returns
    -------
    list of list of str
        List of stemmed token lists.
    """
    return [stem_tokens(tokens, use_stemming) for tokens in token_lists]