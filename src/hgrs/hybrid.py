"""Hybrid strategies: compositions of the base models.

Fitting a hybrid fits its parts. When several hybrids share the same base models (as in
``ModelSuite``), fit the base models once and use the hybrids without refitting.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hgrs.data import UserHistory
from hgrs.models.base import HistoryRecommender, Recommender, empty_scores


def minmax(scores: pd.Series) -> pd.Series:
    """Rescale to [0, 1] so scores from different models can be added."""
    low, high = scores.min(), scores.max()
    if pd.isna(low) or high == low:
        return scores * 0.0
    return (scores - low) / (high - low)


def _fit_each_once(models: Sequence[Recommender], interactions: pd.DataFrame) -> None:
    fitted: set[int] = set()
    for model in models:
        if id(model) not in fitted:
            model.fit(interactions)
            fitted.add(id(model))


def _scores_from(table: pd.DataFrame, name: str) -> pd.Series:
    return table["score"].rename(name) if not table.empty else empty_scores(name)


class CascadeHybrid(Recommender):
    """Each stage re-ranks the survivors of the previous stage and keeps its top ``n``.

    Default: TF-IDF 500 → BERT 300 → KNN 100 → SVD 10. Cheap content models narrow the
    catalogue first; the collaborative models make the final, personalised cut.
    """

    name = "pipeline"

    def __init__(self, stages: Sequence[tuple[Recommender, int]]):
        if not stages:
            raise ValueError("a cascade needs at least one stage")
        self.stages = list(stages)

    def fit(self, interactions: pd.DataFrame) -> CascadeHybrid:
        _fit_each_once([model for model, _ in self.stages], interactions)
        return self

    def explain(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.DataFrame:
        """The final ranking, with every stage's score for each surviving game."""
        survivors_per_stage = []
        for model, keep in self.stages:
            survivors = model.score(user_id, candidates).head(keep)
            if survivors.empty:
                return pd.DataFrame(columns=["score"])
            survivors_per_stage.append(survivors)
            candidates = list(survivors.index)

        final = survivors_per_stage[-1]
        table = pd.DataFrame({"score": final})
        for stage in survivors_per_stage:
            table[f"{stage.name}_score"] = stage.reindex(final.index)
        return table

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        return _scores_from(self.explain(user_id, candidates), self.name)


class WeightedHybrid(Recommender):
    """One model proposes candidates; the others score them and the scores are blended.

    Component scores are min-max normalised per user before weighting, so models on
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

    def fit(self, interactions: pd.DataFrame) -> WeightedHybrid:
        parts = [self.candidate_model, *(model for model, _ in self.components.values())]
        _fit_each_once(parts, interactions)
        return self

    def explain(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.DataFrame:
        """The blended ranking, with each component's raw score."""
        pool = self.candidate_model.score(user_id, candidates).head(self.n_candidates)
        if pool.empty:
            return pd.DataFrame(columns=["score"])

        table = pd.DataFrame(index=pool.index)
        blended = pd.Series(0.0, index=pool.index)
        for name, (model, weight) in self.components.items():
            raw = model.score(user_id, list(pool.index)).reindex(pool.index)
            table[f"{name}_score"] = raw
            blended += weight * minmax(raw).fillna(0.0)

        table.insert(0, "score", blended)
        table = table.sort_values("score", ascending=False, kind="mergesort")
        return table.head(self.top_k) if self.top_k else table

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        return _scores_from(self.explain(user_id, candidates), self.name)


class AdaptiveHybrid(HistoryRecommender):
    """Trust collaborative filtering in proportion to how much we know about the user.

        cf_weight = n / (n + shrinkage)        n = number of games the user played
        score     = (1 - cf_weight) · content + cf_weight · collaborative

    A newcomer with 1-2 games is served almost entirely by the content models, which only
    need game descriptions. As the history grows, the collaborative model, which needs
    behaviour to learn from, takes over. With ``shrinkage = 10`` the two count equally
    at 10 games.
    """

    name = "adaptive"

    def __init__(
        self,
        content_models: Sequence[HistoryRecommender],
        cf_model: HistoryRecommender,
        shrinkage: float = 10.0,
    ):
        super().__init__()
        if not content_models:
            raise ValueError("the adaptive hybrid needs at least one content model")
        self.content_models = list(content_models)
        self.cf_model = cf_model
        self.shrinkage = shrinkage

    def fit(self, interactions: pd.DataFrame) -> AdaptiveHybrid:
        _fit_each_once([*self.content_models, self.cf_model], interactions)
        return self

    def cf_weight(self, n_games: int) -> float:
        return n_games / (n_games + self.shrinkage) if n_games else 0.0

    # Known users get the collaborative model's trained factors; anyone else is folded in.

    def explain(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.DataFrame:
        history = self.cf_model.histories.get(str(user_id))
        if history is None:
            return pd.DataFrame(columns=["score"])
        cf_scores = self.cf_model.score(user_id, candidates)
        return self._blend(history, cf_scores, candidates)

    def explain_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.DataFrame:
        cf_scores = self.cf_model.score_history(history, candidates)
        return self._blend(history, cf_scores, candidates)

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        return _scores_from(self.explain(user_id, candidates), self.name)

    def score_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        return _scores_from(self.explain_history(history, candidates), self.name)

    def _blend(
        self, history: UserHistory, cf_scores: pd.Series, candidates: Sequence[str] | None
    ) -> pd.DataFrame:
        content_parts = [
            minmax(model.score_history(history, candidates)) for model in self.content_models
        ]
        content = pd.concat(content_parts, axis=1).mean(axis=1)
        collaborative = minmax(cf_scores)

        games = content.index.union(collaborative.index, sort=False)
        if games.empty:
            return pd.DataFrame(columns=["score"])
        table = pd.DataFrame(
            {
                "content_score": content.reindex(games).fillna(0.0),
                "cf_score": collaborative.reindex(games).fillna(0.0),
            }
        )
        w = self.cf_weight(len(history))
        table.insert(0, "score", (1 - w) * table["content_score"] + w * table["cf_score"])
        table.attrs["cf_weight"] = w
        return table.sort_values("score", ascending=False, kind="mergesort")
