"""
Preprocessing pipeline assembly.
Chains normalizer → tokenizer → stop word removal → stemmer into
a single configurable function, and provides a dataset-level
batch processing utility.
"""

from preprocessing.normalizer import normalize
from preprocessing.tokenizer import tokenize
from preprocessing.stop_words import remove_stop_words
from preprocessing.stemmer import stem_tokens


def preprocess(text,
               use_stop_words=True,
               use_stemming=True,
               number_strategy="remove",
               tokenize_method="regex"):
    """
    Run the full preprocessing pipeline on a single text string.

    Parameters
    ----------
    text : str
        Raw user input.
    use_stop_words : bool
        If True, remove tokens found in the stop-word set.
    use_stemming : bool
        If True, apply suffix-stripping stemmer to each token.
    number_strategy : str
        'remove' | 'replace' | 'keep'  — forwarded to normalizer.
    tokenize_method : str
        'regex' | 'whitespace' — forwarded to tokenizer.

    Returns
    -------
    list of str
        Final processed token list.
    """
    # Step 1: Normalize raw text
    text = normalize(text, number_strategy=number_strategy)

    # Step 2: Split into tokens
    tokens = tokenize(text, method=tokenize_method)

    # Step 3: Optionally remove stop words
    if use_stop_words:
        tokens = remove_stop_words(tokens)

    # Step 4: Optionally apply stemming
    if use_stemming:
        tokens = stem_tokens(tokens)

    return tokens


def preprocess_dataset(data,
                       use_stop_words=True,
                       use_stemming=True,
                       number_strategy="remove",
                       tokenize_method="regex"):
    """
    Apply the preprocessing pipeline to every record in a dataset.

    Parameters
    ----------
    data : list of dict
        Each dict must have keys 'text' (str) and 'label' (int).
    use_stop_words : bool
        Forwarded to preprocess().
    use_stemming : bool
        Forwarded to preprocess().
    number_strategy : str
        Forwarded to preprocess().
    tokenize_method : str
        Forwarded to preprocess().

    Returns
    -------
    list of dict
        Each dict contains:
            'text'   — original raw text (str)
            'label'  — integer class label (int)
            'tokens' — preprocessed token list (list of str)
    """
    processed = []
    for record in data:
        tokens = preprocess(
            record["text"],
            use_stop_words=use_stop_words,
            use_stemming=use_stemming,
            number_strategy=number_strategy,
            tokenize_method=tokenize_method,
        )
        processed.append({
            "text": record["text"],
            "label": record["label"],
            "tokens": tokens,
        })
    return processed