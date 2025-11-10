"""NLTK-based keyword extraction helpers."""

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_freq_keywords(tokens, top_n=10):
    """
    Get top N frequent keywords from a tokenized text.
    """
    freq_dist = Counter(tokens)
    keywords = [word for word, _ in freq_dist.most_common(top_n)]
    return keywords


def extract_bulk_freq_keywords(token_lists, top_n=10):
    """
    Get top N frequent keywords for each document (token list).
    Returns a list of keyword lists.
    """
    return [extract_freq_keywords(tokens, top_n) for tokens in token_lists]


def extract_tfidf_keywords(raw_texts, top_n=10):
    """
    Extract keywords using TF-IDF from raw (uncleaned) texts.
    Returns a list of keyword lists for each document.
    """
    vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(raw_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    keywords_per_doc = []
    for doc_idx in range(tfidf_matrix.shape[0]):
        # Get top indices for this doc
        row = tfidf_matrix.getrow(doc_idx)
        sorted_indices = row.toarray()[0].argsort()[::-1][:top_n]
        keywords = [feature_names[idx] for idx in sorted_indices if row[0, idx] > 0]
        keywords_per_doc.append(keywords)
    return keywords_per_doc


def extract_top_tfidf_keywords(raw_texts, top_n=10):
    """
    Extracts top global TF-IDF keywords across all texts (not per document).
    """
    vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(raw_texts)
    summed_tfidf = tfidf_matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = summed_tfidf.argsort()[::-1][:top_n]
    top_keywords = [feature_names[i] for i in sorted_indices]
    return top_keywords
