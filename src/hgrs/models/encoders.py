"""Text encoders that turn game descriptions into dense vectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceBertEncoder:
    """Sentence-BERT embeddings (the model used in the project: ``all-MiniLM-L6-v2``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on an optional extra
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install -e .[bert]`, "
                "or set content.encoder to 'lsa' for a lightweight encoder."
            ) from e
        model = SentenceTransformer(self.name)
        return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


class LsaEncoder:
    """TF-IDF followed by truncated SVD: a dependency-free stand-in for Sentence-BERT."""

    name = "lsa"

    def __init__(self, n_components: int = 64, seed: int = 42):
        self.n_components = n_components
        self.seed = seed

    def encode(self, texts: list[str]) -> np.ndarray:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
        k = max(1, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        reduced = TruncatedSVD(n_components=k, random_state=self.seed).fit_transform(tfidf)
        return normalize(reduced)


def make_encoder(name: str, seed: int = 42) -> Encoder:
    return LsaEncoder(seed=seed) if name == "lsa" else SentenceBertEncoder(name)
