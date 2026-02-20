"""
Vocabulary builder for text classification.
Constructs a token-to-index mapping from preprocessed training data
with configurable frequency filtering and size limits.
"""

from collections import Counter


class Vocabulary:
    """
    Builds and manages a vocabulary mapping from tokens to integer indices.

    The vocabulary is constructed from a training corpus with optional
    constraints on minimum token frequency and maximum vocabulary size.
    Once built, it provides bidirectional lookup between tokens and indices.

    Attributes
    ----------
    token_to_index : dict
        Maps token strings to unique integer indices.
    index_to_token : dict
        Reverse mapping from indices back to token strings.
    size : int
        Total number of unique tokens in the vocabulary.
    token_counts : Counter
        Frequency count of each token in the training corpus.
    min_freq : int
        Minimum frequency threshold used during construction.
    max_size : int or None
        Maximum vocabulary size constraint used during construction.
    """

    def __init__(self, min_freq=2, max_size=None):
        """
        Initialize an empty vocabulary with filtering constraints.

        Parameters
        ----------
        min_freq : int
            Minimum number of occurrences required for a token to be
            included in the vocabulary. Tokens appearing fewer times
            are treated as out-of-vocabulary. Default is 2 to filter
            rare tokens that may be typos or dataset noise.
        max_size : int or None
            Maximum number of tokens to keep in the vocabulary.
            If set, only the top-K most frequent tokens are retained.
            If None, all tokens meeting min_freq are kept.
        """
        self.min_freq = min_freq
        self.max_size = max_size

        self.token_to_index = {}
        self.index_to_token = {}
        self.token_counts = Counter()
        self.size = 0

    def build_from_tokens(self, token_lists):
        """
        Construct the vocabulary from a list of preprocessed token lists.

        This method should be called once on the training data only.
        The same vocabulary is then used for both training and test data
        to ensure consistency.

        Parameters
        ----------
        token_lists : list of list of str
            Each inner list represents the preprocessed tokens from
            one training sample.

        Returns
        -------
        self
            Returns self to allow method chaining.
        """
        # ── Step 1: Count all token occurrences ───────
        for tokens in token_lists:
            self.token_counts.update(tokens)

        # ── Step 2: Filter by minimum frequency ───────
        filtered_tokens = [
            token for token, count in self.token_counts.items()
            if count >= self.min_freq
        ]

        # ── Step 3: Sort by frequency (descending) ────
        # In case max_size is set, we want to keep the most frequent
        filtered_tokens.sort(key=lambda t: self.token_counts[t], reverse=True)

        # ── Step 4: Apply max_size constraint ─────────
        if self.max_size is not None and len(filtered_tokens) > self.max_size:
            filtered_tokens = filtered_tokens[:self.max_size]

        # ── Step 5: Build bidirectional mappings ──────
        self.token_to_index = {token: idx for idx, token in enumerate(filtered_tokens)}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        self.size = len(self.token_to_index)

        return self

    def build_from_dataset(self, data):
        """
        Convenience method to build vocabulary directly from a dataset.

        Parameters
        ----------
        data : list of dict
            Each dict must have a 'tokens' key containing a list of str.

        Returns
        -------
        self
            Returns self to allow method chaining.
        """
        token_lists = [record["tokens"] for record in data]
        return self.build_from_tokens(token_lists)

    def get_index(self, token):
        """
        Get the integer index for a token.

        Parameters
        ----------
        token : str

        Returns
        -------
        int or None
            The token's index if it exists in the vocabulary, else None.
        """
        return self.token_to_index.get(token, None)

    def get_token(self, index):
        """
        Get the token string for an integer index.

        Parameters
        ----------
        index : int

        Returns
        -------
        str or None
            The token at the given index if it exists, else None.
        """
        return self.index_to_token.get(index, None)

    def contains(self, token):
        """
        Check whether a token is in the vocabulary.

        Parameters
        ----------
        token : str

        Returns
        -------
        bool
        """
        return token in self.token_to_index

    def get_summary(self):
        """
        Generate a summary dictionary of vocabulary statistics.

        Returns
        -------
        dict
            Contains keys: size, min_freq, max_size, total_tokens_seen,
            unique_tokens_seen, coverage (fraction of unique tokens retained).
        """
        total_unique = len(self.token_counts)
        coverage = (self.size / total_unique * 100) if total_unique > 0 else 0

        return {
            "size": self.size,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "total_tokens_seen": sum(self.token_counts.values()),
            "unique_tokens_seen": total_unique,
            "coverage_percent": coverage,
        }

    def print_summary(self):
        """
        Print a formatted summary of the vocabulary to the console.
        """
        summary = self.get_summary()

        print("\n" + "=" * 65)
        print("VOCABULARY SUMMARY")
        print("=" * 65)
        print(f"  Vocabulary size              : {summary['size']}")
        print(f"  Minimum frequency threshold  : {summary['min_freq']}")
        print(f"  Maximum size limit           : {summary['max_size']}")
        print(f"  Total tokens seen in corpus  : {summary['total_tokens_seen']}")
        print(f"  Unique tokens before filter  : {summary['unique_tokens_seen']}")
        print(f"  Coverage (retained / unique) : {summary['coverage_percent']:.1f}%")
        print("=" * 65)

    def print_most_common(self, n=20):
        """
        Print the N most frequent tokens in the vocabulary.

        Parameters
        ----------
        n : int
            Number of tokens to display.
        """
        print(f"\nTOP {n} MOST FREQUENT TOKENS:")
        print("-" * 50)
        print(f"  {'Rank':<6} {'Token':<20} {'Count':<10} {'Index':<10}")
        print("-" * 50)

        # Get tokens sorted by frequency
        sorted_tokens = sorted(
            self.token_to_index.keys(),
            key=lambda t: self.token_counts[t],
            reverse=True
        )

        for rank, token in enumerate(sorted_tokens[:n], start=1):
            count = self.token_counts[token]
            index = self.token_to_index[token]
            print(f"  {rank:<6} {token:<20} {count:<10} {index:<10}")

        print("-" * 50)

    def print_least_common(self, n=20):
        """
        Print the N least frequent tokens in the vocabulary.

        Useful for identifying boundary cases at the minimum frequency threshold.

        Parameters
        ----------
        n : int
            Number of tokens to display.
        """
        print(f"\nBOTTOM {n} LEAST FREQUENT TOKENS:")
        print("-" * 50)
        print(f"  {'Token':<20} {'Count':<10} {'Index':<10}")
        print("-" * 50)

        # Get tokens sorted by frequency ascending
        sorted_tokens = sorted(
            self.token_to_index.keys(),
            key=lambda t: self.token_counts[t]
        )

        for token in sorted_tokens[:n]:
            count = self.token_counts[token]
            index = self.token_to_index[token]
            print(f"  {token:<20} {count:<10} {index:<10}")

        print("-" * 50)

    def __len__(self):
        """Return vocabulary size."""
        return self.size

    def __contains__(self, token):
        """Support 'token in vocab' syntax."""
        return self.contains(token)

    def __repr__(self):
        """String representation for debugging."""
        return (f"Vocabulary(size={self.size}, min_freq={self.min_freq}, "
                f"max_size={self.max_size})")