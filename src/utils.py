"""Utility helpers: data loading, plotting and saving keywords."""

import pandas as pd
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_keywords(keywords, title="Keyword Frequency", top_n=10):
    """
    Plots a bar chart of keyword frequencies.
    'keywords' can be a dict (keyword: freq) or a list of keywords.
    """
    if isinstance(keywords, dict):
        # Already a frequency dict
        freq_dict = keywords
    else:
        # Assume list, count frequency
        from collections import Counter
        freq_dict = dict(Counter(keywords))

    # Select top N
    sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, values = zip(*sorted_items)
    
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def save_keywords(keywords, filepath):
    """
    Saves a list of keywords to a text file (one per line).
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for kw in keywords:
                f.write(str(kw) + "\n")
        print(f"Keywords saved to {filepath}")
    except Exception as e:
        print(f"Error saving keywords: {e}")
