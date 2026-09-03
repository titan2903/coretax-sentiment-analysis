"""Indonesian review text preprocessing."""

import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


SLANG_NORMALIZATION = {
    "bgt": "banget",
    "bangetnya": "banget",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "ngk": "tidak",
    "aja": "saja",
    "apk": "aplikasi",
    "apknya": "aplikasinya",
    "blm": "belum",
    "belom": "belum",
    "bener": "benar",
    "krn": "karena",
    "krena": "karena",
    "lg": "lagi",
    "msh": "masih",
    "ribet": "rumit",
    "udh": "sudah",
    "udah": "sudah",
    "yg": "yang",
}

_STEMMER = StemmerFactory().create_stemmer()
_STOPWORD_REMOVER = StopWordRemoverFactory().create_stop_word_remover()


def clean_text(text: object) -> str:
    """Normalize URLs, mentions, punctuation, digits, and whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]


def normalize_slang(text: str) -> str:
    return " ".join(SLANG_NORMALIZATION.get(token, token) for token in text.split())


def preprocess(text: object) -> str:
    text = normalize_slang(clean_text(text))
    text = _STOPWORD_REMOVER.remove(text)
    return _STEMMER.stem(text)


def preprocess_many(texts: list[object]) -> list[str]:
    """Preprocess a corpus without invoking the slow corpus-wide stemmer."""
    normalized = []
    for text in texts:
        value = _STOPWORD_REMOVER.remove(normalize_slang(clean_text(text)))
        tokens = value.split()
        normalized.append(tokens)
    return [" ".join(tokens) for tokens in normalized]
