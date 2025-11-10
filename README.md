# 🔍 NLP Keyword Extractor

A modular Python project for extracting keywords from BBC News articles using **NLTK**, **spaCy**, and **scikit-learn**. Extract keywords via word frequency, TF-IDF scores, and noun phrases.

## Features

- 📊 **Multiple Extraction Methods**
  - Word Frequency (NLTK)
  - TF-IDF Scores (scikit-learn)
  - Noun Phrases (spaCy)
- 🧹 **Text Preprocessing**: Tokenization, cleaning, stopword removal (NLTK)
- 📈 **Data Visualization**: Built-in plotting utilities
- 🏗️ **Modular Design**: Reusable functions and clear package structure
- 📓 **Exploratory Analysis**: Jupyter notebook for interactive exploration
- ✅ **Well Tested**: Unit tests for core functionality

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Keyword-Extractor
```

### 2. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
& '.\venv\Scripts\Activate.ps1'

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Language Models
```bash
# NLTK data (tokenization, stopwords)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# spaCy English model
python -m spacy download en_core_web_sm
```

### 5. Prepare Data
The project expects `data/bbc_news.csv` with columns like `title`, `description`, etc.
- Download the [BBC News dataset](https://www.kaggle.com/datasets/gpreda/bbc-news) from Kaggle
- Or use any CSV with a `text`, `description`, or `content` column

### 6. Run the Pipeline
```bash
# From project root (recommended)
python run.py

# Or run as a module
python -m src.main
```

## Example Output

```
Preprocessing texts using NLTK...

Extracting top keywords with frequency (NLTK) for first 3 docs:
Doc 1: ['ukrainian', 'president', 'says', 'country', 'forgive', 'forget', 'murder', 'civilians']
Doc 2: ['jeremy', 'bowen', 'frontline', 'irpin', 'residents', 'came', 'russian', 'fire', 'trying', 'flee']
Doc 3: ['one', 'worlds', 'biggest', 'fertiliser', 'firms', 'says', 'conflict', 'could', 'deliver', 'shock']

Extracting top TF-IDF keywords (NLTK) for first 3 docs:
Doc 1: ['forgive', 'forget', 'civilians', 'murder', 'ukrainian', 'country', 'president', 'says']
...
```

## Project Structure

```
Keyword-Extractor/
├── src/                              # Main package
│   ├── __init__.py                   # Package exports
│   ├── main.py                       # Entry point
│   ├── data_preprocessing.py         # Text cleaning & tokenization
│   ├── keyword_extraction_nltk.py    # NLTK-based extraction
│   ├── keyword_extraction_spacy.py   # spaCy noun phrase extraction
│   └── utils.py                      # Data loading, plotting, saving
├── notebooks/
│   └── exploratory_analysis.ipynb    # Interactive data exploration
├── data/
│   └── bbc_news.csv                  # BBC News dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py         # Tests for preprocessing
│   └── test_extraction.py            # Tests for extraction
├── .github/
│   └── workflows/
│       └── test.yml                  # GitHub Actions CI/CD
├── run.py                            # Convenience runner
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## Module Documentation

### `src.data_preprocessing`
Handles text cleaning and tokenization:
```python
from src.data_preprocessing import preprocess, preprocess_bulk

tokens = preprocess("Hello world!")  # ['hello', 'world']
tokenized = preprocess_bulk(["Doc 1", "Doc 2"])
```

### `src.keyword_extraction_nltk`
Extract keywords using frequency and TF-IDF:
```python
from src.keyword_extraction_nltk import extract_freq_keywords, extract_tfidf_keywords

keywords = extract_freq_keywords(tokens, top_n=10)
tfidf_kws = extract_tfidf_keywords(texts, top_n=10)
```

### `src.keyword_extraction_spacy`
Extract noun phrases:
```python
from src.keyword_extraction_spacy import extract_noun_phrases, top_noun_phrases

phrases = extract_noun_phrases("The quick brown fox jumps")
top = top_noun_phrases(texts, top_n=10)
```

### `src.utils`
Utilities for data I/O and visualization:
```python
from src.utils import load_data, plot_keywords, save_keywords

df = load_data('data/articles.csv')
plot_keywords(keywords, title="Keyword Frequency", top_n=10)
save_keywords(['python', 'javascript'], 'output/keywords.txt')
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## Dependencies

- **pandas**: Data loading and manipulation
- **nltk**: Tokenization and stopword removal
- **spacy**: NLP and noun phrase extraction
- **scikit-learn**: TF-IDF vectorization
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **jupyter**: Interactive notebooks

See `requirements.txt` for exact versions.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Commit changes**: `git commit -m "Add my feature"`
4. **Push to branch**: `git push origin feature/my-feature`
5. **Open a Pull Request**

### Code Style
- Follow PEP 8
- Add docstrings to all functions
- Write unit tests for new features
- Ensure tests pass: `pytest tests/`

## Issues & Feedback

Found a bug? Have a suggestion? Please open an [issue](https://github.com/debangan/keyword-extractor/issues).

## License

This project is licensed under the MIT License — see `LICENSE` for details.

## Author

**Debangan Mali**

---

**Happy Keyword Extracting! 🚀**