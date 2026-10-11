"""Content-based recommenders: TF-IDF and Sentence-BERT user profiles over game metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from hgrs.data import GAME_TEXT_COLUMNS, normalise_games
from hgrs.models.base import Recommender, empty_scores, ranked


def split_tags(value: str) -> list[str]:
    return [t.strip() for t in str(value).replace(";", ",").split(",") if t.strip()]


class GameCatalog:
    """Game metadata plus the derived features shared by the content models."""

    def __init__(self, games: pd.DataFrame):
        self.games = normalise_games(games)
        self.app_ids = self.games["app_id"].to_numpy()
        self.index_of = {a: i for i, a in enumerate(self.app_ids)}

        genres = self.games["genre"].map(split_tags)
        self.genre_matrix = MultiLabelBinarizer().fit_transform(genres).astype(float)

        price = self.games["Price"].to_numpy(dtype=float)
        span = price.max() - price.min() if len(price) else 0.0
        self.price_norm = (price - price.min()) / span if span > 0 else np.zeros_like(price)

    def __len__(self) -> int:
        return len(self.app_ids)

    def text(self, kind: str = "combined") -> list[str]:
        if kind == "about":
            return self.games["About the game"].tolist()
        if kind == "combined":
            return self.games[GAME_TEXT_COLUMNS].agg(" ".join, axis=1).tolist()
        raise ValueError(f"unknown text kind {kind!r}; use 'combined' or 'about'")


class _ContentRecommender(Recommender):
    """Builds a profile from the games a user played and scores candidates against it."""

    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog
        self._history: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, interactions: pd.DataFrame) -> "_ContentRecommender":
        known = interactions[interactions["app_id"].isin(self.catalog.index_of)]
        idx = known["app_id"].map(self.catalog.index_of).to_numpy(dtype=int)
        hours = known["hours"].to_numpy(dtype=float)
        self._history = {
            str(u): (idx[rows], hours[rows]) for u, rows in known.groupby("user_id").indices.items()
        }
        self._build_features()
        return self

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        history = self._history.get(str(user_id))
        if history is None:
            return empty_scores(self.name)
        if candidates is None:
            cand = np.setdiff1d(np.arange(len(self.catalog)), history[0])
        else:
            cand = np.array([self.catalog.index_of[a] for a in candidates if a in self.catalog.index_of], dtype=int)
        if cand.size == 0:
            return empty_scores(self.name)
        return ranked(self.catalog.app_ids[cand], self._score(history, cand), self.name)

    def _build_features(self) -> None:
        raise NotImplementedError

    def _score(self, history: tuple[np.ndarray, np.ndarray], cand: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TfidfRecommender(_ContentRecommender):
    """Cosine similarity between the mean TF-IDF vector of played games and each candidate."""

    name = "tfidf"

    def __init__(self, catalog: GameCatalog, max_features: int = 5000, text: str = "combined"):
        super().__init__(catalog)
        self.max_features = max_features
        self.text = text

    def _build_features(self) -> None:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        self.matrix = vectorizer.fit_transform(self.catalog.text(self.text))

    def _score(self, history, cand):
        profile = np.asarray(self.matrix[history[0]].mean(axis=0))
        return cosine_similarity(profile, self.matrix[cand]).ravel()


class BertRecommender(_ContentRecommender):
    """Sentence embeddings + genre overlap + price affinity, weighted by hours played.

    score = content * <emb, profile> + genre * <genres, genre_profile> + price * (1 - |price - preferred|)
    """

    name = "bert"

    def __init__(
        self,
        catalog: GameCatalog,
        encoder: "Encoder",
        weights: dict[str, float] | None = None,
        cache_dir: str | Path | None = None,
    ):
        super().__init__(catalog)
        self.encoder = encoder
        self.weights = {"content": 0.7, "genre": 0.2, "price": 0.1, **(weights or {})}
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _build_features(self) -> None:
        self.embeddings = self._load_cached() if self.cache_dir else None
        if self.embeddings is None:
            self.embeddings = np.asarray(self.encoder.encode(self.catalog.text("combined")), dtype=float)
            if self.cache_dir:
                self._save_cache()

    def _score(self, history, cand):
        idx, hours = history
        weights = hours if hours.sum() > 0 else None
        content = np.average(self.embeddings[idx], axis=0, weights=weights)
        genre = np.average(self.catalog.genre_matrix[idx], axis=0, weights=weights)
        price_pref = self.catalog.price_norm[idx].mean()
        w = self.weights
        return (
            w["content"] * (self.embeddings[cand] @ content)
            + w["genre"] * (self.catalog.genre_matrix[cand] @ genre)
            + w["price"] * (1 - np.abs(self.catalog.price_norm[cand] - price_pref))
        )

    def _cache_file(self) -> Path:
        safe = self.encoder.name.replace("/", "_")
        return self.cache_dir / f"embeddings_{safe}.npz"

    def _load_cached(self) -> np.ndarray | None:
        path = self._cache_file()
        if not path.exists():
            return None
        cached = np.load(path, allow_pickle=False)
        ids = self.catalog.app_ids.astype(str)
        if cached["app_ids"].shape != ids.shape or not (cached["app_ids"] == ids).all():
            return None
        return cached["embeddings"]

    def _save_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._cache_file(), app_ids=self.catalog.app_ids.astype(str), embeddings=self.embeddings)


class Encoder:
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceBertEncoder(Encoder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install -e .[bert]`, "
                "or set content.encoder to 'lsa' in the config for a lightweight encoder."
            ) from e
        model = SentenceTransformer(self.name)
        return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


class LsaEncoder(Encoder):
    """TF-IDF + truncated SVD: a dependency-free stand-in for Sentence-BERT (demo and tests)."""

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
