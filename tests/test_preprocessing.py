"""Unit tests for data preprocessing module."""

import unittest
from src.data_preprocessing import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    preprocess,
    preprocess_bulk
)


class TestCleanText(unittest.TestCase):
    """Tests for clean_text function."""

    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        result = clean_text("HELLO World")
        self.assertEqual(result, "hello world")

    def test_url_removal(self):
        """Test that URLs are removed."""
        result = clean_text("Check this out: http://example.com now")
        self.assertNotIn("http", result)

    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        result = clean_text("Hello, World! How are you?")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)

    def test_digit_removal(self):
        """Test that digits are removed."""
        result = clean_text("Test 123 data 456")
        self.assertNotIn("123", result)
        self.assertNotIn("456", result)

    def test_multiple_spaces_removed(self):
        """Test that multiple spaces are collapsed."""
        result = clean_text("hello    world")
        self.assertEqual(result, "hello world")


class TestTokenizeText(unittest.TestCase):
    """Tests for tokenize_text function."""

    def test_basic_tokenization(self):
        """Test basic tokenization."""
        result = tokenize_text("hello world")
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_punctuation_as_token(self):
        """Test that punctuation is tokenized separately."""
        result = tokenize_text("hello, world!")
        # NLTK tokenizes punctuation as separate tokens
        self.assertTrue(len(result) >= 2)


class TestRemoveStopwords(unittest.TestCase):
    """Tests for remove_stopwords function."""

    def test_stopword_removal(self):
        """Test that stopwords are removed."""
        tokens = ["the", "quick", "brown", "fox"]
        result = remove_stopwords(tokens)
        self.assertNotIn("the", result)
        self.assertIn("quick", result)
        self.assertIn("brown", result)
        self.assertIn("fox", result)

    def test_empty_list(self):
        """Test with empty list."""
        result = remove_stopwords([])
        self.assertEqual(result, [])


class TestPreprocess(unittest.TestCase):
    """Tests for preprocess function."""

    def test_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "The quick, brown fox jumps over the lazy dog!"
        result = preprocess(text)
        # Should be a list of tokens without stopwords
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        # Common stopwords should be removed
        self.assertNotIn("the", result)
        self.assertNotIn("over", result)

    def test_empty_string(self):
        """Test with empty string."""
        result = preprocess("")
        self.assertEqual(result, [])


class TestPreprocessBulk(unittest.TestCase):
    """Tests for preprocess_bulk function."""

    def test_multiple_documents(self):
        """Test preprocessing multiple documents."""
        texts = ["Hello world", "Python programming", "Data science"]
        result = preprocess_bulk(texts)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(doc, list) for doc in result))

    def test_empty_list(self):
        """Test with empty list."""
        result = preprocess_bulk([])
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
