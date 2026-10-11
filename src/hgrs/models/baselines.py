"""Reference points every real model has to beat."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hgrs.data import UserHistory
from hgrs.models.base import HistoryRecommender, empty_scores, ranked


def _candidate_pool(
    known_items: pd.Index, history: UserHistory, candidates: Sequence[str] | None
) -> pd.Index:
    if candidates is None:
        return known_items.difference(pd.Index(history.app_ids), sort=False)
    return known_items.intersection(pd.Index([str(c) for c in candidates]), sort=False)


class PopularityRecommender(HistoryRecommender):
    """Recommend the games most users recommended. Not personalised at all."""

    name = "popularity"

    def _fit(self, interactions: pd.DataFrame) -> None:
        liked = interactions.loc[interactions["is_recommended"] >= 1, "app_id"]
        every_game = pd.Index(interactions["app_id"].unique())
        self.popularity = liked.value_counts().reindex(every_game, fill_value=0)

    def score_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        pool = _candidate_pool(self.popularity.index, history, candidates)
        if pool.empty:
            return empty_scores(self.name)
        return ranked(pool, self.popularity.reindex(pool).to_numpy(), self.name)


class RandomRecommender(HistoryRecommender):
    """A random ranking, seeded per user so it is reproducible."""

    name = "random"

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed

    def _fit(self, interactions: pd.DataFrame) -> None:
        self.items = pd.Index(interactions["app_id"].unique())

    def score_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        pool = _candidate_pool(self.items, history, candidates)
        if pool.empty:
            return empty_scores(self.name)
        user_key = zlib.crc32(",".join(map(str, history.app_ids)).encode())
        rng = np.random.default_rng([self.seed, user_key])
        return ranked(pool, rng.random(len(pool)), self.name)
