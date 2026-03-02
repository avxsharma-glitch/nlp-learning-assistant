# ============================================================
# preprocessing.py — NLTK-based text normalisation
# ============================================================

import nltk

def _ensure_corpora():
    """Download required NLTK corpora if they are not already present."""
    for download_id in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(download_id, quiet=True)

_ensure_corpora()

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer

_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()


def normalize(text: str) -> str:
    """
    Tokenize *text*, keep only alphabetic tokens, remove English stop-words,
    and lemmatize each remaining token.

    Returns a single space-joined string ready for vectorisation.
    """
    tokens   = word_tokenize(text.lower())
    filtered = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in _stop_words
    ]
    return " ".join(filtered)
