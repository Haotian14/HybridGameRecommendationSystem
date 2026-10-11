"""The interfaces every recommender implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hgrs.data import UserHistory, histories_by_user


class Recommender(ABC):
    """A model that scores games for a user.

    ``score`` returns a Series indexed by ``app_id``, best first. With ``candidates=None``
    the model ranks every game it knows that the user has not played; otherwise it scores
    only the given candidates (ones it does not know are dropped).
    """

    name: str = "recommender"

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> Recommender: ...

    @abstractmethod
    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series: ...

    def recommend(
        self, user_id: str, k: int = 10, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        return self.score(user_id, candidates).head(k)


class HistoryRecommender(Recommender):
    """A recommender that only needs a user's play history, not their id.

    It can therefore serve users it never saw during training (the cold-start case):
    ``score_history`` works on any ``UserHistory``, and ``score(user_id)`` simply looks up
    the history remembered at ``fit`` time.
    """

    def __init__(self) -> None:
        self.histories: dict[str, UserHistory] = {}

    def fit(self, interactions: pd.DataFrame) -> HistoryRecommender:
        self.histories = histories_by_user(interactions)
        self._fit(interactions)
        return self

    def _fit(self, interactions: pd.DataFrame) -> None:
        """Learn model parameters; histories are already stored."""

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        history = self.histories.get(str(user_id))
        if history is None:
            return empty_scores(self.name)
        return self.score_history(history, candidates)

    @abstractmethod
    def score_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.Series: ...

    def recommend_history(
        self, history: UserHistory, k: int = 10, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        return self.score_history(history, candidates).head(k)


def ranked(app_ids: Sequence[str], scores: Sequence[float], name: str) -> pd.Series:
    """Best-first score Series. Ties keep their input order, so results are deterministic."""
    index = pd.Index(list(app_ids), name="app_id")
    series = pd.Series(np.asarray(scores, dtype=float), index=index, name=name)
    return series.sort_values(ascending=False, kind="mergesort")


def empty_scores(name: str) -> pd.Series:
    return pd.Series([], index=pd.Index([], name="app_id"), name=name, dtype=float)
