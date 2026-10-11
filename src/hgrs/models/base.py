"""Common interface for every recommender, base or hybrid."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd


class Recommender(ABC):
    """A model that scores games for a user.

    ``score`` returns a Series indexed by ``app_id`` and sorted best-first. With
    ``candidates=None`` the model ranks every game it knows that the user has not
    played; otherwise it scores only the given candidates (unknown ones are dropped).
    """

    name: str = "recommender"

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "Recommender": ...

    @abstractmethod
    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series: ...

    def recommend(self, user_id: str, k: int = 10, candidates: Sequence[str] | None = None) -> pd.Series:
        return self.score(user_id, candidates).head(k)


def ranked(app_ids: Sequence[str], scores: Sequence[float], name: str) -> pd.Series:
    """Build a best-first score Series; ties keep their input order so results are deterministic."""
    s = pd.Series(np.asarray(scores, dtype=float), index=pd.Index(list(app_ids), name="app_id"), name=name)
    return s.sort_values(ascending=False, kind="mergesort")


def empty_scores(name: str) -> pd.Series:
    return pd.Series([], index=pd.Index([], name="app_id"), name=name, dtype=float)
