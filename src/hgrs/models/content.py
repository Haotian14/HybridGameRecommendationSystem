"""Content-based recommenders: compare games a user played with the rest of the catalogue."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hgrs.data import UserHistory
from hgrs.models.base import HistoryRecommender, empty_scores, ranked
from hgrs.models.catalog import GameCatalog
from hgrs.models.encoders import Encoder


class ContentRecommender(HistoryRecommender):
    """Shared plumbing: map a history onto catalogue positions, pick candidates, rank."""

    def __init__(self, catalog: GameCatalog):
        super().__init__()
        self.catalog = catalog

    def _fit(self, interactions: pd.DataFrame) -> None:
        self._build_features()

    def score_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        played, known = self.catalog.positions(history.app_ids)
        if played.size == 0:
            return empty_scores(self.name)
        if candidates is None:
            pool = np.setdiff1d(np.arange(len(self.catalog)), played)
        else:
            pool, _ = self.catalog.positions(list(candidates))
        if pool.size == 0:
            return empty_scores(self.name)
        scores = self._score(played, history.hours[known], pool)
        return ranked(self.catalog.app_ids[pool], scores, self.name)

    def _build_features(self) -> None:
        raise NotImplementedError

    def _score(self, played: np.ndarray, hours: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """Score catalogue positions ``pool`` for a user who played ``played``."""
        raise NotImplementedError


class TfidfRecommender(ContentRecommender):
    """Cosine similarity between the user's mean TF-IDF vector and each candidate."""

    name = "tfidf"

    def __init__(self, catalog: GameCatalog, max_features: int = 5000, text: str = "combined"):
        super().__init__(catalog)
        self.max_features = max_features
        self.text = text

    def _build_features(self) -> None:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        self.matrix = vectorizer.fit_transform(self.catalog.text(self.text))

    def _score(self, played, hours, pool):
        profile = np.asarray(self.matrix[played].mean(axis=0))
        return cosine_similarity(profile, self.matrix[pool]).ravel()


class BertRecommender(ContentRecommender):
    """Sentence embeddings, genre overlap and price affinity, weighted by hours played.

    score = w_content · sim(embedding, profile)
          + w_genre   · sim(genres, genre profile)
          + w_price   · (1 - |price - preferred price|)

    ``similarity="cosine"`` (default) keeps both similarity terms in [-1, 1]. The original
    notebooks used raw dot products (``"dot"``): a game's genre score then grows with the
    number of genre tags it has, reaching ~3 against ~0.5 for the content term, so games
    tagged with many genres crowded out everything else.
    """

    name = "bert"

    def __init__(
        self,
        catalog: GameCatalog,
        encoder: Encoder,
        weights: dict[str, float] | None = None,
        cache_dir: str | Path | None = None,
        similarity: str = "cosine",
    ):
        super().__init__(catalog)
        if similarity not in ("cosine", "dot"):
            raise ValueError(f"similarity must be 'cosine' or 'dot', not {similarity!r}")
        self.encoder = encoder
        self.weights = {"content": 0.7, "genre": 0.2, "price": 0.1, **(weights or {})}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.similarity = similarity

    def _build_features(self) -> None:
        cached = self._load_cached_embeddings()
        if cached is not None:
            self.embeddings = cached
            return
        texts = self.catalog.text("combined")
        self.embeddings = np.asarray(self.encoder.encode(texts), dtype=float)
        self._save_cached_embeddings()

    def _score(self, played, hours, pool):
        weights = hours if hours.sum() > 0 else None  # all-zero hours: plain average
        content_profile = np.average(self.embeddings[played], axis=0, weights=weights)
        genre_profile = np.average(self.catalog.genre_matrix[played], axis=0, weights=weights)
        preferred_price = self.catalog.price_norm[played].mean()

        content = self._similarity(self.embeddings[pool], content_profile)
        genre = self._similarity(self.catalog.genre_matrix[pool], genre_profile)
        price = 1 - np.abs(self.catalog.price_norm[pool] - preferred_price)
        w = self.weights
        return w["content"] * content + w["genre"] * genre + w["price"] * price

    def _similarity(self, rows: np.ndarray, profile: np.ndarray) -> np.ndarray:
        if self.similarity == "dot":
            return rows @ profile
        if not profile.any():
            return np.zeros(len(rows))
        return cosine_similarity(profile.reshape(1, -1), rows).ravel()

    # Encoding a few thousand descriptions with BERT takes minutes on a CPU, so the
    # embeddings are cached and reused as long as the catalogue is unchanged.

    def _cache_file(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"embeddings_{self.encoder.name.replace('/', '_')}.npz"

    def _load_cached_embeddings(self) -> np.ndarray | None:
        path = self._cache_file()
        if path is None or not path.exists():
            return None
        cached = np.load(path, allow_pickle=False)
        if not np.array_equal(cached["app_ids"], self.catalog.app_ids.astype(str)):
            return None
        return cached["embeddings"]

    def _save_cached_embeddings(self) -> None:
        path = self._cache_file()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, app_ids=self.catalog.app_ids.astype(str), embeddings=self.embeddings
        )
