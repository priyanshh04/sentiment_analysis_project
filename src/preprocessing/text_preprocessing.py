"""
Text Preprocessing Module
Handles comprehensive text cleaning and preprocessing for sentiment analysis
"""

import pandas as pd
import numpy as np
import re
import string
import logging
from typing import List, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

# Try to import NLP libraries with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Some advanced features will be disabled.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Some advanced features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive text preprocessing class for sentiment analysis
    """

    def __init__(self, 
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_html: bool = True,
                 remove_punctuation: bool = True,
                 remove_digits: bool = True,
                 remove_extra_whitespace: bool = True,
                 remove_stopwords: bool = True,
                 apply_lemmatization: bool = True,
                 min_word_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize TextPreprocessor

        Args:
            lowercase (bool): Convert text to lowercase
            remove_urls (bool): Remove URLs and email addresses
            remove_html (bool): Remove HTML tags
            remove_punctuation (bool): Remove punctuation marks
            remove_digits (bool): Remove digit characters
            remove_extra_whitespace (bool): Remove extra whitespace
            remove_stopwords (bool): Remove stopwords
            apply_lemmatization (bool): Apply lemmatization
            min_word_length (int): Minimum word length to keep
            custom_stopwords (List[str], optional): Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_stopwords = remove_stopwords
        self.apply_lemmatization = apply_lemmatization
        self.min_word_length = min_word_length
        self.custom_stopwords = custom_stopwords or []

        # Initialize components
        self.stopwords_set = set()
        self.lemmatizer = None
        self.is_fitted = False

        # Download required NLTK data
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data"""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                logger.info("Downloading required NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('omw-1.4', quiet=True)

    def fit(self, X: Union[pd.Series, List[str]], y=None):
        """
        Fit the preprocessor (initialize stopwords and lemmatizer)

        Args:
            X: Input texts
            y: Target variable (ignored)

        Returns:
            self: Returns the instance itself
        """
        logger.info("Fitting text preprocessor...")

        # Initialize stopwords
        if self.remove_stopwords and NLTK_AVAILABLE:
            try:
                self.stopwords_set = set(stopwords.words('english'))
                self.stopwords_set.update(self.custom_stopwords)

                # Add domain-specific stopwords for musical instruments
                domain_stopwords = [
                    'guitar', 'bass', 'drum', 'piano', 'keyboard', 'instrument',
                    'sound', 'music', 'play', 'playing', 'song', 'songs'
                ]
                self.stopwords_set.update(domain_stopwords)

            except Exception as e:
                logger.warning(f"Could not load stopwords: {e}")
                self.stopwords_set = set(self.custom_stopwords)

        # Initialize lemmatizer
        if self.apply_lemmatization and NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"Could not initialize lemmatizer: {e}")
                self.lemmatizer = None

        self.is_fitted = True
        logger.info("Text preprocessor fitted successfully")
        return self

    def transform(self, X: Union[pd.Series, List[str]]) -> Union[pd.Series, List[str]]:
        """
        Transform texts by applying preprocessing steps

        Args:
            X: Input texts

        Returns:
            Preprocessed texts
        """
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before transform")

        logger.info(f"Preprocessing {len(X)} texts...")

        # Convert to appropriate format
        is_series = isinstance(X, pd.Series)
        texts = X.tolist() if is_series else list(X)

        # Apply preprocessing steps
        processed_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} texts...")

            processed_text = self._preprocess_single_text(text)
            processed_texts.append(processed_text)

        logger.info("Text preprocessing completed")

        # Return in original format
        if is_series:
            return pd.Series(processed_texts, index=X.index)
        else:
            return processed_texts

    def fit_transform(self, X: Union[pd.Series, List[str]], y=None) -> Union[pd.Series, List[str]]:
        """
        Fit and transform in one step

        Args:
            X: Input texts
            y: Target variable (ignored)

        Returns:
            Preprocessed texts
        """
        return self.fit(X, y).transform(X)

    def _preprocess_single_text(self, text: str) -> str:
        """
        Preprocess a single text

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to string
        text = str(text)

        # Step 1: Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Step 2: Remove URLs and emails
        if self.remove_urls:
            text = self._remove_urls_emails(text)

        # Step 3: Remove HTML tags
        if self.remove_html:
            text = self._remove_html_tags(text)

        # Step 4: Remove extra whitespace (early normalization)
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        # Step 5: Handle contractions
        text = self._expand_contractions(text)

        # Step 6: Remove punctuation
        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        # Step 7: Remove digits
        if self.remove_digits:
            text = re.sub(r'\d+', '', text)

        # Step 8: Tokenization and advanced processing
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)

            # Step 9: Remove short words
            tokens = [token for token in tokens if len(token) >= self.min_word_length]

            # Step 10: Remove stopwords
            if self.remove_stopwords and self.stopwords_set:
                tokens = [token for token in tokens if token not in self.stopwords_set]

            # Step 11: Lemmatization
            if self.apply_lemmatization and self.lemmatizer:
                tokens = self._lemmatize_tokens(tokens)

            # Rejoin tokens
            text = ' '.join(tokens)
        else:
            # Basic tokenization fallback
            words = text.split()
            words = [word for word in words if len(word) >= self.min_word_length]
            if self.remove_stopwords and self.stopwords_set:
                words = [word for word in words if word not in self.stopwords_set]
            text = ' '.join(words)

        # Final whitespace cleanup
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        return text

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks"""
        # Keep some punctuation that might be meaningful
        punctuation_to_keep = ''
        translator = str.maketrans('', '', string.punctuation.replace(punctuation_to_keep, ''))
        return text.translate(translator)

    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions"""
        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
            "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are",
            "you've": "you have"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if not self.lemmatizer:
            return tokens

        try:
            # Get POS tags for better lemmatization
            pos_tags = pos_tag(tokens)

            lemmatized = []
            for word, pos in pos_tags:
                # Convert POS tag to WordNet POS tag
                wordnet_pos = self._get_wordnet_pos(pos)
                lemmatized_word = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
                lemmatized.append(lemmatized_word)

            return lemmatized

        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return tokens

    def _get_wordnet_pos(self, pos_tag: str) -> str:
        """Convert POS tag to WordNet POS tag"""
        if pos_tag.startswith('J'):
            return 'a'  # adjective
        elif pos_tag.startswith('V'):
            return 'v'  # verb
        elif pos_tag.startswith('N'):
            return 'n'  # noun
        elif pos_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun

# Utility functions
def preprocess_texts(texts: Union[pd.Series, List[str]], 
                    **kwargs) -> Union[pd.Series, List[str]]:
    """
    Convenience function for text preprocessing

    Args:
        texts: Input texts to preprocess
        **kwargs: Preprocessing parameters

    Returns:
        Preprocessed texts
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.fit_transform(texts)

def clean_text_basic(text: str) -> str:
    """
    Basic text cleaning function (no dependencies)

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_text_stats(texts: Union[pd.Series, List[str]]) -> Dict[str, Any]:
    """
    Get statistics about text data

    Args:
        texts: Input texts

    Returns:
        Dict[str, Any]: Text statistics
    """
    if isinstance(texts, pd.Series):
        texts = texts.dropna().tolist()
    else:
        texts = [t for t in texts if pd.notna(t) and t != '']

    if not texts:
        return {}

    lengths = [len(str(text)) for text in texts]
    word_counts = [len(str(text).split()) for text in texts]

    stats = {
        'total_texts': len(texts),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_word_count': np.mean(word_counts),
        'median_word_count': np.median(word_counts),
        'min_word_count': min(word_counts),
        'max_word_count': max(word_counts),
        'empty_texts': sum(1 for text in texts if str(text).strip() == ''),
        'vocabulary_size': len(set(' '.join(str(t) for t in texts).split()))
    }

    return stats
