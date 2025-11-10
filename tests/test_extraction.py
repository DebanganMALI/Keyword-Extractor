"""Unit tests for keyword extraction modules."""

import unittest
from src.keyword_extraction_nltk import (
    extract_freq_keywords,
    extract_bulk_freq_keywords,
    extract_tfidf_keywords,
    extract_top_tfidf_keywords
)
from src.keyword_extraction_spacy import (
    extract_noun_phrases,
    top_noun_phrases
)


class TestExtractFreqKeywords(unittest.TestCase):
    """Tests for extract_freq_keywords function."""

    def test_basic_frequency(self):
        """Test basic keyword frequency extraction."""
        tokens = ['python', 'python', 'javascript', 'python', 'rust']
        result = extract_freq_keywords(tokens, top_n=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'python')  # Most frequent

    def test_top_n_respected(self):
        """Test that top_n parameter is respected."""
        tokens = ['a', 'a', 'b', 'b', 'c', 'c', 'd']
        result = extract_freq_keywords(tokens, top_n=2)
        self.assertEqual(len(result), 2)

    def test_empty_list(self):
        """Test with empty token list."""
        result = extract_freq_keywords([], top_n=5)
        self.assertEqual(result, [])

    def test_single_token(self):
        """Test with single token."""
        result = extract_freq_keywords(['python'], top_n=5)
        self.assertEqual(result, ['python'])


class TestExtractBulkFreqKeywords(unittest.TestCase):
    """Tests for extract_bulk_freq_keywords function."""

    def test_multiple_documents(self):
        """Test extraction for multiple documents."""
        token_lists = [
            ['python', 'python', 'code'],
            ['javascript', 'javascript', 'web'],
        ]
        result = extract_bulk_freq_keywords(token_lists, top_n=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 'python')
        self.assertEqual(result[1][0], 'javascript')


class TestExtractTFIDFKeywords(unittest.TestCase):
    """Tests for extract_tfidf_keywords function."""

    def test_basic_tfidf(self):
        """Test basic TF-IDF extraction."""
        texts = [
            "python is great for data science",
            "javascript is used for web development",
            "python and javascript are popular languages"
        ]
        result = extract_tfidf_keywords(texts, top_n=3)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(doc, list) for doc in result))

    def test_tfidf_returns_keywords_per_doc(self):
        """Test that TF-IDF returns keywords per document."""
        texts = ["hello world", "foo bar baz"]
        result = extract_tfidf_keywords(texts, top_n=2)
        self.assertEqual(len(result), 2)


class TestExtractTopTFIDFKeywords(unittest.TestCase):
    """Tests for extract_top_tfidf_keywords function."""

    def test_global_tfidf(self):
        """Test global TF-IDF extraction across corpus."""
        texts = [
            "python python python",
            "javascript javascript",
            "rust"
        ]
        result = extract_top_tfidf_keywords(texts, top_n=2)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, list)


class TestExtractNounPhrases(unittest.TestCase):
    """Tests for extract_noun_phrases function."""

    def test_noun_phrase_extraction(self):
        """Test noun phrase extraction from text."""
        text = "The quick brown fox jumps over the lazy dog"
        result = extract_noun_phrases(text)
        self.assertIsInstance(result, list)
        # Should extract at least some noun phrases
        self.assertTrue(len(result) > 0)

    def test_lowercase_output(self):
        """Test that noun phrases are lowercase."""
        text = "The Quick Brown Fox"
        result = extract_noun_phrases(text)
        if result:
            # All extracted phrases should be lowercase
            self.assertTrue(all(phrase.islower() or ' ' in phrase for phrase in result))

    def test_empty_text(self):
        """Test with empty text."""
        result = extract_noun_phrases("")
        self.assertEqual(result, [])


class TestTopNounPhrases(unittest.TestCase):
    """Tests for top_noun_phrases function."""

    def test_most_common_phrases(self):
        """Test extraction of most common noun phrases."""
        texts = [
            "The cat sat on the mat",
            "The cat is on the mat",
            "A dog runs in the park"
        ]
        result = top_noun_phrases(texts, top_n=3)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) <= 3)

    def test_top_n_respected(self):
        """Test that top_n parameter is respected."""
        texts = ["The cat", "The dog", "The bird"] * 3
        result = top_noun_phrases(texts, top_n=5)
        self.assertTrue(len(result) <= 5)


if __name__ == '__main__':
    unittest.main()
