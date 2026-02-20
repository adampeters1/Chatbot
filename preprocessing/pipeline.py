"""
Preprocessing pipeline module.
Chains all text preprocessing steps into a single, configurable pipeline.
"""

from preprocessing.normalizer import normalize_text, normalize_batch
from preprocessing.tokenizer import tokenize, tokenize_batch
from preprocessing.stopwords import remove_stopwords, remove_stopwords_batch
from preprocessing.stemmer import stem_tokens, stem_tokens_batch


class PreprocessingPipeline:
    """
    Unified text preprocessing pipeline.
    
    Applies the following operations in sequence:
    1. Text normalization (lowercase, punctuation removal, etc.)
    2. Tokenization
    3. Stop word removal (optional)
    4. Stemming (optional)
    
    Parameters
    ----------
    use_stopwords : bool, default=True
        Whether to remove stop words.
    use_stemming : bool, default=True
        Whether to apply stemming.
    
    Attributes
    ----------
    use_stopwords : bool
    use_stemming : bool
    
    Examples
    --------
    >>> pipeline = PreprocessingPipeline()
    >>> tokens = pipeline.process("Hello! How are you doing?")
    >>> print(tokens)
    ['hello', 'do']
    """
    
    def __init__(self, use_stopwords=True, use_stemming=True):
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
    
    def process(self, text):
        """
        Process a single text string through the full pipeline.
        
        Parameters
        ----------
        text : str
            Raw input text.
        
        Returns
        -------
        list of str
            Final processed tokens.
        """
        
        # Step 1: Normalize
        normalized = normalize_text(text)
        
        # Step 2: Tokenize
        tokens = tokenize(normalized)
        
        # Step 3: Remove stop words (optional)
        tokens = remove_stopwords(tokens, use_stopwords=self.use_stopwords)
        
        # Step 4: Stem (optional)
        tokens = stem_tokens(tokens, use_stemming=self.use_stemming)
        
        return tokens
    
    def process_batch(self, texts):
        """
        Process a batch of texts through the pipeline.
        
        Parameters
        ----------
        texts : list of str
            List of raw text strings.
        
        Returns
        -------
        list of list of str
            List of processed token lists.
        """
        
        # Step 1: Normalize batch
        normalized = normalize_batch(texts)
        
        # Step 2: Tokenize batch
        tokens = tokenize_batch(normalized)
        
        # Step 3: Remove stop words batch
        tokens = remove_stopwords_batch(tokens, use_stopwords=self.use_stopwords)
        
        # Step 4: Stem batch
        tokens = stem_tokens_batch(tokens, use_stemming=self.use_stemming)
        
        return tokens
    
    def __repr__(self):
        return (
            f"PreprocessingPipeline("
            f"use_stopwords={self.use_stopwords}, "
            f"use_stemming={self.use_stemming})"
        )