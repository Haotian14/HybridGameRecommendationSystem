"""The two hybrid strategies: a multi-stage cascade and a weighted score fusion."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from hgrs.models.base import Recommender, empty_scores


def _fit_unique(models: Sequence[Recommender], interactions: pd.DataFrame) -> None:
    seen: set[int] = set()
    for m in models:
        if id(m) not in seen:
            m.fit(interactions)
            seen.add(id(m))


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return s * 0.0
    return (s - lo) / (hi - lo)


class CascadeHybrid(Recommender):
    """Each stage re-ranks the survivors of the previous stage and keeps its top ``n``.

    Default: TF-IDF 500 -> BERT 300 -> KNN 100 -> SVD 10. Cheap content models narrow the
    catalogue first; the collaborative models make the final, personalised cut.
    """

    name = "pipeline"

    def __init__(self, stages: Sequence[tuple[Recommender, int]]):
        if not stages:
            raise ValueError("a cascade needs at least one stage")
        self.stages = list(stages)

    def fit(self, interactions: pd.DataFrame) -> "CascadeHybrid":
        _fit_unique([m for m, _ in self.stages], interactions)
        return self

    def explain(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.DataFrame:
        """Final ranking plus every stage's score for the surviving games."""
        stage_scores = []
        for model, keep in self.stages:
            s = model.score(user_id, candidates).head(keep)
            if s.empty:
                return pd.DataFrame(columns=["score"])
            stage_scores.append(s)
            candidates = list(s.index)
        final = stage_scores[-1]
        table = pd.DataFrame({"score": final})
        for s in stage_scores:
            table[f"{s.name}_score"] = s.reindex(final.index)
        return table

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        table = self.explain(user_id, candidates)
        return table["score"].rename(self.name) if not table.empty else empty_scores(self.name)


class WeightedHybrid(Recommender):
    """One model proposes candidates; the others score them and the scores are blended.

    Each component score is min-max normalised per user before weighting, so models with
    different scales (KNN in [0, 1], SVD in [1, 5], cosine similarities) are comparable.
    """

    name = "weighted"

    def __init__(
        self,
        candidate_model: Recommender,
        n_candidates: int,
        components: dict[str, tuple[Recommender, float]],
        top_k: int | None = None,
    ):
        if not components:
            raise ValueError("a weighted hybrid needs at least one component")
        self.candidate_model = candidate_model
        self.n_candidates = n_candidates
        self.components = components
        self.top_k = top_k

    def fit(self, interactions: pd.DataFrame) -> "WeightedHybrid":
        _fit_unique([self.candidate_model, *(m for m, _ in self.components.values())], interactions)
        return self

    def explain(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.DataFrame:
        pool = self.candidate_model.score(user_id, candidates).head(self.n_candidates)
        if pool.empty:
            return pd.DataFrame(columns=["score"])
        table = pd.DataFrame(index=pool.index)
        total = pd.Series(0.0, index=pool.index)
        for name, (model, weight) in self.components.items():
            raw = model.score(user_id, list(pool.index)).reindex(pool.index)
            table[f"{name}_score"] = raw
            total += weight * minmax(raw).fillna(0.0)
        table.insert(0, "score", total)
        table = table.sort_values("score", ascending=False, kind="mergesort")
        return table.head(self.top_k) if self.top_k else table

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        table = self.explain(user_id, candidates)
        return table["score"].rename(self.name) if not table.empty else empty_scores(self.name)
