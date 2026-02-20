"""
Stop word removal utilities.
Contains a manually curated set of common English stop words
and a filter function to remove them from token lists.

NOTE: Some words in this set (e.g. 'not', 'no') carry significant
meaning for certain intent classes such as negation.  The removal
step is therefore made toggleable in the pipeline so its effect
on classification accuracy can be measured empirically.
"""

STOP_WORDS = {
    # ── Articles ──────────────────────────────
    "a", "an", "the",

    # ── Prepositions ──────────────────────────
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "out",

    # ── Conjunctions ─────────────────────────
    "and", "but", "or", "nor", "so", "yet",

    # ── Pronouns ──────────────────────────────
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",

    # ── Common verbs / auxiliaries ─────────────
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "could", "should",
    "may", "might", "can", "shall",

    # ── Demonstratives / determiners ──────────
    "this", "that", "these", "those",

    # ── Question words ────────────────────────
    "what", "which", "who", "whom", "where", "when", "how", "why",

    # ── Subordinating conjunctions ────────────
    "if", "then", "than", "as", "while", "because", "although",

    # ── Adverbs / quantifiers ─────────────────
    "not", "no", "just", "only", "very", "too", "also",
    "there", "here",

    # ── Indefinite / distributive ─────────────
    "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "any",
    "same", "own",
}


def remove_stop_words(tokens):
    """
    Filter out stop words from a token list.

    Parameters
    ----------
    tokens : list of str
        Input tokens (expected to already be lowercased).

    Returns
    -------
    list of str
        Tokens with all stop words removed.
    """
    return [token for token in tokens if token not in STOP_WORDS]