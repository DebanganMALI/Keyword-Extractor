"""spaCy-based noun phrase extraction helpers."""

import spacy
from collections import Counter

# Load spaCy English model once
nlp = spacy.load('en_core_web_sm')


def extract_noun_phrases(text):
    """
    Extracts noun phrases from a single text string.
    Returns a list of phrases.
    """
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]


def extract_bulk_noun_phrases(texts):
    """
    Extracts noun phrases for a list of text strings.
    Returns a list of lists of phrases.
    """
    return [extract_noun_phrases(text) for text in texts]


def top_noun_phrases(texts, top_n=10):
    """
    Gets most common noun phrases across all texts.
    Returns a list of top N phrases.
    """
    # Flatten all noun phrases into one list
    phrases = []
    for text in texts:
        phrases.extend(extract_noun_phrases(text))
    phrase_counts = Counter(phrases)
    return [phrase for phrase, count in phrase_counts.most_common(top_n)]
