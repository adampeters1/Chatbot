"""
Simplified Porter-like stemmer.
Implements the three most impactful suffix-stripping rule groups
without the full complexity of the complete Porter algorithm.

Rule groups are applied sequentially:
    A) Plural suffixes    (-sses, -ies, -es, -s)
    B) Verb suffixes      (-tion, -ing, -ed)
    C) Adj/Adv suffixes   (-ness, -ment, -able, -ible, -ful, -ly)

Minimum-length guards prevent over-stemming of short words.
A double-consonant stripping step cleans stems after -ing / -ed
removal (e.g. "running" → "runn" → "run").
"""

VOWELS = set("aeiou")


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def _is_consonant(char):
    """Return True if the character is not a vowel."""
    return char not in VOWELS


def _ends_with_double_consonant(word):
    """Check whether word ends with two identical consonant chars."""
    if len(word) >= 2:
        return word[-1] == word[-2] and _is_consonant(word[-1])
    return False


def _strip_double_consonant(word):
    """Remove one character from a trailing double consonant.
    'runn' → 'run',  'stopp' → 'stop'"""
    if _ends_with_double_consonant(word):
        return word[:-1]
    return word


# ──────────────────────────────────────────────
# Step A — Plural Suffixes
# ──────────────────────────────────────────────

def _step_a_plurals(word):
    """
    Strip plural suffixes.
    Rules checked in order of specificity (longest match first):
        -sses → -ss     (grasses  → grass)
        -ies  → -y      (babies   → baby)
        -es   → remove  (boxes    → box)
        -s    → remove  (cats     → cat)
    """
    if word.endswith("sses"):
        return word[:-2]

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("es") and len(word) > 4:
        # Only strip -es where it is a genuine plural suffix
        # (after s, x, z, sh, ch)
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "sh", "ch")):
            return stem

    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


# ──────────────────────────────────────────────
# Step B — Verb Suffixes
# ──────────────────────────────────────────────

def _step_b_verb_suffixes(word):
    """
    Strip verb suffixes.
    Rules checked in order of specificity:
        -tion → keep 't', remove 'ion'  (action    → act)
        -ing  → remove, clean doubles    (running   → run)
        -ed   → remove, clean doubles    (stopped   → stop)
    """
    if word.endswith("tion") and len(word) > 5:
        return word[:-3]

    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        return _strip_double_consonant(stem)

    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        return _strip_double_consonant(stem)

    return word


# ──────────────────────────────────────────────
# Step C — Adjective / Adverb Suffixes
# ──────────────────────────────────────────────

def _step_c_adj_adv_suffixes(word):
    """
    Strip adjective and adverb suffixes.
    Rules checked longest-first to avoid partial matches:
        -ness → remove  (sadness   → sad)
        -ment → remove  (movement  → move)
        -able → remove  (readable  → read)
        -ible → remove  (flexible  → flex)
        -ful  → remove  (helpful   → help)
        -ly   → remove  (quickly   → quick)
    """
    if word.endswith("ness") and len(word) > 6:
        return word[:-4]

    if word.endswith("ment") and len(word) > 6:
        return word[:-4]

    if word.endswith("able") and len(word[:-4]) >= 4:
        return word[:-4]
    if word.endswith("ible") and len(word[:-4]) >= 4:
        return word[:-4]

    if word.endswith("ful") and len(word) > 5:
        return word[:-3]

    if word.endswith("ly") and len(word) > 4:
        return word[:-2]

    return word


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def stem(word):
    """
    Apply simplified Porter-like stemming to a single word.

    Steps A → B → C are applied sequentially so that plural
    stripping feeds into verb-suffix stripping and so on.
    Words of three or fewer characters are returned unchanged.

    Parameters
    ----------
    word : str
        A single lowercase token.

    Returns
    -------
    str
        The stemmed token.
    """
    if len(word) <= 3:
        return word

    word = _step_a_plurals(word)
    word = _step_b_verb_suffixes(word)
    word = _step_c_adj_adv_suffixes(word)

    return word


def stem_tokens(tokens):
    """
    Apply stemming to every token in a list.

    Parameters
    ----------
    tokens : list of str

    Returns
    -------
    list of str
        Stemmed tokens in the same order.
    """
    return [stem(token) for token in tokens]