"""Entry point for the Keyword Extractor package.

This module is designed to be run as a module (python -m src.main) or imported.
"""

from pathlib import Path

import pandas as pd

from .data_preprocessing import preprocess_bulk
from .keyword_extraction_nltk import extract_freq_keywords, extract_tfidf_keywords
from .keyword_extraction_spacy import extract_noun_phrases, top_noun_phrases


def main():
    # 1. Load data (path resolved relative to project root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    csv_path = project_root / 'data' / 'bbc_news.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Put bbc_news.csv in the project's data/ folder.")

    df = pd.read_csv(csv_path)

    # Determine which column holds the article text. Common names: 'text', 'content', 'description', 'article', 'body', 'summary'
    preferred_text_cols = ['text', 'content', 'description', 'article', 'body', 'summary', 'full_text', 'clean_text']
    text_col = next((c for c in preferred_text_cols if c in df.columns), None)
    if text_col is None:
        # If there's a title+description available, combine them as a fallback
        if 'title' in df.columns and 'description' in df.columns:
            texts = (df['title'].fillna('') + '. ' + df['description'].fillna('')).astype(str).tolist()
            print("Using combined 'title' + 'description' as text input.")
        else:
            raise KeyError(f"No suitable text column found in dataset. Available columns: {list(df.columns)}")
    else:
        texts = df[text_col].astype(str).tolist()

    # 2. Preprocess data (tokenization, cleaning, stopword removal)
    print("Preprocessing texts using NLTK...")
    tokenized_texts = preprocess_bulk(texts)

    # 3. NLTK Keyword Extraction
    print("\nExtracting top keywords with frequency (NLTK) for first 3 docs:")
    for i in range(3):
        print(f"Doc {i+1}:", extract_freq_keywords(tokenized_texts[i], top_n=10))

    print("\nExtracting top TF-IDF keywords (NLTK) for first 3 docs:")
    tfidf_keywords = extract_tfidf_keywords(texts, top_n=10)
    for i in range(3):
        print(f"Doc {i+1}:", tfidf_keywords[i])

    # 4. spaCy Keyword Extraction
    print("\nExtracting noun phrase keywords with spaCy for first 3 docs:")
    for i in range(3):
        print(f"Doc {i+1}:", extract_noun_phrases(texts[i])[:10])

    print("\nTop 10 most frequent noun phrases in corpus (spaCy):")
    common_np = top_noun_phrases(texts, top_n=10)
    print(common_np)


if __name__ == '__main__':
    main()
