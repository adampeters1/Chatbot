"""
Text vectorization utilities.
Implements Bag-of-Words and TF-IDF vector representations.
"""

import numpy as np
from collections import Counter


class BagOfWordsVectorizer:
    """
    Converts token lists into fixed-length Bag-of-Words vectors.

    Each vector has length equal to the vocabulary size, with each
    position representing the count of the corresponding token.

    Attributes
    ----------
    vocabulary : Vocabulary
        The vocabulary instance used for token-to-index mapping.
    """

    def __init__(self, vocabulary):
        """
        Initialize vectorizer with a pre-built vocabulary.

        Parameters
        ----------
        vocabulary : Vocabulary
            Must be already built from training data.
        """
        self.vocabulary = vocabulary

    def transform_single(self, tokens):
        """
        Convert a single token list into a BoW vector.

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens from one sample.

        Returns
        -------
        np.ndarray
            1D array of shape (vocab_size,) with token counts.
        """
        vector = np.zeros(self.vocabulary.size, dtype=np.float32)
        
        for token in tokens:
            idx = self.vocabulary.get_index(token)
            if idx is not None:
                vector[idx] += 1.0
        
        return vector

    def transform(self, token_lists):
        """
        Convert multiple token lists into BoW vectors.

        Parameters
        ----------
        token_lists : list of list of str
            List of preprocessed token lists.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, vocab_size).
        """
        vectors = np.zeros((len(token_lists), self.vocabulary.size), dtype=np.float32)
        
        for i, tokens in enumerate(token_lists):
            vectors[i] = self.transform_single(tokens)
        
        return vectors

    def transform_dataset(self, data):
        """
        Convenience method to vectorize a dataset.

        Parameters
        ----------
        data : list of dict
            Each dict must have a 'tokens' key.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, vocab_size).
        """
        token_lists = [record["tokens"] for record in data]
        return self.transform(token_lists)


class TfidfVectorizer:
    """
    Converts token lists into TF-IDF weighted vectors.

    TF (Term Frequency): count(term) / total_terms_in_document
    IDF (Inverse Document Frequency): log(total_docs / (1 + docs_containing_term))

    The vectorizer must be fit on training data before transforming any data.

    Attributes
    ----------
    vocabulary : Vocabulary
        The vocabulary instance used for token-to-index mapping.
    idf : np.ndarray or None
        IDF values for each vocabulary token, computed during fit().
    n_documents : int
        Number of documents used during fitting.
    """

    def __init__(self, vocabulary):
        """
        Initialize TF-IDF vectorizer with a pre-built vocabulary.

        Parameters
        ----------
        vocabulary : Vocabulary
            Must be already built from training data.
        """
        self.vocabulary = vocabulary
        self.idf = None
        self.n_documents = 0

    def fit(self, token_lists):
        """
        Compute IDF values from training token lists.

        Must be called once on training data before transform() can be used.

        Parameters
        ----------
        token_lists : list of list of str
            Training token lists.

        Returns
        -------
        self
        """
        self.n_documents = len(token_lists)
        document_frequency = np.zeros(self.vocabulary.size, dtype=np.float32)

        # Count how many documents contain each token
        for tokens in token_lists:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                idx = self.vocabulary.get_index(token)
                if idx is not None:
                    document_frequency[idx] += 1.0

        # Compute IDF: log(N / (1 + df))
        # The +1 prevents division by zero for tokens that appear in vocabulary
        # but might not appear in the current split
        self.idf = np.log(self.n_documents / (1.0 + document_frequency))

        return self

    def fit_dataset(self, data):
        """
        Convenience method to fit from a dataset.

        Parameters
        ----------
        data : list of dict
            Each dict must have a 'tokens' key.

        Returns
        -------
        self
        """
        token_lists = [record["tokens"] for record in data]
        return self.fit(token_lists)

    def transform_single(self, tokens):
        """
        Convert a single token list into a TF-IDF vector.

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens from one sample.

        Returns
        -------
        np.ndarray
            1D array of shape (vocab_size,) with TF-IDF weights.
        """
        if self.idf is None:
            raise RuntimeError("TfidfVectorizer must be fit before transform")

        vector = np.zeros(self.vocabulary.size, dtype=np.float32)
        
        if len(tokens) == 0:
            return vector

        # Compute term frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        for token, count in token_counts.items():
            idx = self.vocabulary.get_index(token)
            if idx is not None:
                tf = count / total_tokens
                vector[idx] = tf * self.idf[idx]

        return vector

    def transform(self, token_lists):
        """
        Convert multiple token lists into TF-IDF vectors.

        Parameters
        ----------
        token_lists : list of list of str
            List of preprocessed token lists.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, vocab_size).
        """
        if self.idf is None:
            raise RuntimeError("TfidfVectorizer must be fit before transform")

        vectors = np.zeros((len(token_lists), self.vocabulary.size), dtype=np.float32)
        
        for i, tokens in enumerate(token_lists):
            vectors[i] = self.transform_single(tokens)
        
        return vectors

    def transform_dataset(self, data):
        """
        Convenience method to vectorize a dataset.

        Parameters
        ----------
        data : list of dict
            Each dict must have a 'tokens' key.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, vocab_size).
        """
        token_lists = [record["tokens"] for record in data]
        return self.transform(token_lists)