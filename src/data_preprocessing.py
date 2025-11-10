"""Data preprocessing helpers (NLTK-based).

This module provides text cleaning, tokenization and stopword removal.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Attempt to download quietly if resources are missing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Lowercase, remove URLs, punctuation, digits, and extra spaces.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)        # Remove punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()    # Remove multiple spaces
    return text


def tokenize_text(text):
    """
    Tokenize text using NLTK.
    """
    return word_tokenize(text)


def remove_stopwords(tokens):
    """
    Remove English stopwords from token list.
    """
    return [token for token in tokens if token not in stop_words]


def preprocess(text):
    """
    Complete preprocessing pipeline.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    filtered_tokens = remove_stopwords(tokens)
    return filtered_tokens


def preprocess_bulk(texts):
    """
    Apply preprocessing pipeline to a list/series of texts.
    """
    return [preprocess(text) for text in texts]
