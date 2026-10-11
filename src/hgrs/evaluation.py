"""Offline evaluation under one protocol for every model.

Two experiments:

* **Warm users.** Hold out 20% of each user's games and rank the full catalogue.
  Besides accuracy, measure *what kind* of games each model recommends: catalogue
  coverage, genre diversity, novelty and how popular the picks are.
* **Cold start.** Remove some users from training entirely, reveal only their first
  n games, and see how well each model predicts the rest as n grows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from hgrs.data import UserHistory
from hgrs.models import GameCatalog, HistoryRecommender, Recommender


def precision_recall_ndcg(
    recommended: Sequence[str], relevant: set[str], k: int
) -> tuple[float, float, float]:
    top = list(recommended)[:k]
    gains = np.array([1.0 if game in relevant else 0.0 for game in top])
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    ideal = (1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
    hits = gains.sum()
    ndcg = float((gains * discounts).sum() / ideal) if ideal > 0 else 0.0
    return hits / k, hits / len(relevant), ndcg


def liked_games(interactions: pd.DataFrame) -> dict[str, set[str]]:
    """A held-out game counts as relevant when the user recommended it."""
    liked = interactions[interactions["is_recommended"] >= 1]
    return {user: set(games["app_id"]) for user, games in liked.groupby("user_id")}


class Evaluator:
    """Computes accuracy and beyond-accuracy metrics against one training set."""

    def __init__(self, train: pd.DataFrame, catalog: GameCatalog, k: int = 10):
        self.k = k
        self.catalog = catalog
        self.n_users = train["user_id"].nunique()
        self.n_games = len(catalog) or train["app_id"].nunique()
        players = train.groupby("app_id")["user_id"].nunique()
        self.players = players
        self.popularity_percentile = players.rank(pct=True)

    # --------------------------------------------------------------- warm users

    def evaluate(
        self,
        models: Mapping[str, Recommender],
        test: pd.DataFrame,
        n_users: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        relevant = liked_games(test)
        users = _sample(sorted(relevant), n_users, seed)

        rows = []
        for name, model in models.items():
            per_user = []
            shown: set[str] = set()
            for user in users:
                recs = list(model.recommend(user, self.k).index)
                shown.update(recs)
                per_user.append(
                    (*precision_recall_ndcg(recs, relevant[user], self.k), *self.describe(recs))
                )
            scores = np.array(per_user, dtype=float)
            means = np.nanmean(scores, axis=0)
            k = self.k
            rows.append(
                {
                    "model": name,
                    f"precision@{k}": means[0],
                    f"recall@{k}": means[1],
                    f"ndcg@{k}": means[2],
                    # standard error of the mean over users: differences smaller than about
                    # twice this are not distinguishable from noise
                    "ndcg_se": _standard_error(scores[:, 2]),
                    "coverage": len(shown) / self.n_games,
                    "diversity": means[3],
                    "novelty": means[4],
                    "popularity_pct": means[5],
                }
            )
        return pd.DataFrame(rows).set_index("model")

    def describe(self, recs: Sequence[str]) -> tuple[float, float, float]:
        """Beyond-accuracy properties of one recommendation list.

        diversity       1 - mean pairwise cosine similarity of the games' genre vectors
        novelty         mean self-information  -log2(players / users); higher = less mainstream
        popularity_pct  mean popularity percentile; 1.0 = only the most played games
        """
        players = self.players.reindex(recs).fillna(0).clip(lower=1).to_numpy()
        novelty = float(np.mean(-np.log2(players / self.n_users))) if len(recs) else np.nan
        popularity = float(self.popularity_percentile.reindex(recs).fillna(0).mean())
        return self._diversity(recs), novelty, popularity

    def _diversity(self, recs: Sequence[str]) -> float:
        positions, _ = self.catalog.positions(list(recs))
        genres = self.catalog.genre_matrix[positions]
        genres = genres[genres.sum(axis=1) > 0]
        if len(genres) < 2:
            return np.nan
        unit = genres / np.linalg.norm(genres, axis=1, keepdims=True)
        similarity = unit @ unit.T
        n = len(unit)
        mean_off_diagonal = (similarity.sum() - n) / (n * (n - 1))
        return float(1.0 - mean_off_diagonal)

    # --------------------------------------------------------------- cold start

    def cold_start(
        self,
        models: Mapping[str, HistoryRecommender],
        histories: Mapping[str, UserHistory],
        sizes: Sequence[int],
    ) -> pd.DataFrame:
        """Recall@k when a new user has revealed only their first ``n`` games.

        The relevant set is the games the user liked among those *not* revealed.
        """
        table = {}
        for name, model in models.items():
            row = {}
            for n in sizes:
                recalls = []
                for history in histories.values():
                    hidden_liked = set(history.app_ids[n:][history.recommended[n:] >= 1])
                    if len(history) <= n or not hidden_liked:
                        continue
                    recs = list(model.recommend_history(history.head(n), self.k).index)
                    recalls.append(precision_recall_ndcg(recs, hidden_liked, self.k)[1])
                row[f"n={n}"] = float(np.mean(recalls)) if recalls else np.nan
            table[name] = row
        result = pd.DataFrame(table).T
        result.index.name = "model"
        return result


def _standard_error(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def _sample(users: list[str], n: int | None, seed: int) -> list[str]:
    if not n or len(users) <= n:
        return users
    return sorted(np.random.default_rng(seed).choice(users, size=n, replace=False).tolist())


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """Render a metrics table as GitHub markdown (no ``tabulate`` dependency)."""
    columns = list(df.columns)
    lines = [
        f"| {df.index.name or ''} | " + " | ".join(map(str, columns)) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name in df.index:
        cells = []
        for column in columns:
            value = df.at[name, column]
            if isinstance(value, float | np.floating):
                cells.append("–" if np.isnan(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
