"""
Stop word removal module.
Filters out common English words that typically don't contribute to intent.
"""

# ──────────────────────────────────────────────
# Curated Stop Word List
# ──────────────────────────────────────────────
# This is a manually curated list of ~100 common English stop words.
# These words are typically removed because they add little semantic value
# for intent classification tasks.

STOP_WORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Pronouns
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again',
    'further', 'then', 'once',
    
    # Conjunctions
    'and', 'or', 'but', 'if', 'while', 'because', 'as', 'until',
    'although', 'nor', 'so',
    
    # Auxiliary/Modal verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'should', 'could', 'shall',
    'may', 'might', 'must', 'can',
    
    # Common verbs
    'get', 'got', 'getting',
    
    # Determiners/Quantifiers
    'this', 'that', 'these', 'those',
    'some', 'any', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'another', 'such', 'only', 'own', 'same',
    
    # Adverbs
    'very', 'too', 'just', 'now', 'here', 'there', 'where', 'when',
    'why', 'how', 'than', 'too', 'also',
    
    # Negation (keep "no" and "not" as they can be important for intent)
    # Removed from stop words intentionally
    
    # Other common words
    'what', 'which', 'who', 'whom', 'whose',
    'out', 'by',
}


def remove_stopwords(tokens, use_stopwords=True):
    """
    Remove stop words from a list of tokens.
    
    Parameters
    ----------
    tokens : list of str
        List of tokens.
    use_stopwords : bool
        If False, return tokens unchanged. Allows easy toggling.
    
    Returns
    -------
    list of str
        Filtered token list with stop words removed.
    
    Examples
    --------
    >>> remove_stopwords(['i', 'love', 'this', 'bot'])
    ['love', 'bot']
    
    >>> remove_stopwords(['hello', 'world'])
    ['hello', 'world']
    
    >>> remove_stopwords([], use_stopwords=True)
    []
    """
    
    if not use_stopwords:
        return tokens
    
    # Filter out any token that appears in the stop word set
    filtered = [token for token in tokens if token not in STOP_WORDS]
    
    # Edge case: if all tokens were stop words, return original
    # This prevents losing all information from very short queries
    if len(filtered) == 0 and len(tokens) > 0:
        return tokens
    
    return filtered


def remove_stopwords_batch(token_lists, use_stopwords=True):
    """
    Remove stop words from a batch of token lists.
    
    Parameters
    ----------
    token_lists : list of list of str
        List of token lists.
    use_stopwords : bool
        If False, return unchanged.
    
    Returns
    -------
    list of list of str
        Filtered token lists.
    """
    return [remove_stopwords(tokens, use_stopwords) for tokens in token_lists]