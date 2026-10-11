"""Collaborative-filtering recommenders built on scikit-surprise."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader

from hgrs.data import assign_rating
from hgrs.models.base import Recommender, empty_scores, ranked


class _SurpriseRecommender(Recommender):
    rating_scale: tuple[float, float]

    def _ratings(self, interactions: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _make_algo(self):
        raise NotImplementedError

    def fit(self, interactions: pd.DataFrame) -> "_SurpriseRecommender":
        df = pd.DataFrame(
            {
                "user_id": interactions["user_id"].astype(str).to_numpy(),
                "app_id": interactions["app_id"].astype(str).to_numpy(),
                "rating": self._ratings(interactions).to_numpy(dtype=float),
            }
        )
        data = Dataset.load_from_df(df, Reader(rating_scale=self.rating_scale))
        self.trainset = data.build_full_trainset()
        self.algo = self._make_algo()
        self.algo.fit(self.trainset)
        return self

    def _inner_uid(self, user_id: str) -> int | None:
        try:
            return self.trainset.to_inner_uid(str(user_id))
        except ValueError:
            return None

    def _unseen_items(self, inner_uid: int) -> list[str]:
        rated = {j for j, _ in self.trainset.ur[inner_uid]}
        return [self.trainset.to_raw_iid(j) for j in self.trainset.all_items() if j not in rated]

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        inner = self._inner_uid(user_id)
        if inner is None:
            return empty_scores(self.name)
        items = self._unseen_items(inner) if candidates is None else [str(c) for c in candidates]
        if not items:
            return empty_scores(self.name)
        return ranked(items, self._predict(str(user_id), inner, items), self.name)

    def _predict(self, user_id: str, inner_uid: int, items: list[str]) -> np.ndarray:
        return np.array([self.algo.predict(user_id, iid).est for iid in items])


class KNNRecommender(_SurpriseRecommender):
    """KNN collaborative filtering on the binary ``is_recommended`` signal."""

    name = "knn"
    rating_scale = (0.0, 1.0)

    def __init__(self, k: int = 40, sim_name: str = "msd", user_based: bool = True):
        self.k = k
        self.sim_options = {"name": sim_name, "user_based": user_based}

    def _ratings(self, interactions):
        return interactions["is_recommended"].astype(float)

    def _make_algo(self):
        return KNNBasic(k=self.k, sim_options=self.sim_options, verbose=False)


class SVDRecommender(_SurpriseRecommender):
    """Funk-SVD matrix factorisation on the 1-5 rating derived from hours and recommendation."""

    name = "svd"
    rating_scale = (1.0, 5.0)

    def __init__(self, n_factors: int = 100, reg_all: float = 0.02, seed: int = 42):
        self.n_factors = n_factors
        self.reg_all = reg_all
        self.seed = seed

    def _ratings(self, interactions):
        return assign_rating(interactions)

    def _make_algo(self):
        return SVD(n_factors=self.n_factors, reg_all=self.reg_all, random_state=self.seed)

    def _predict(self, user_id, inner_uid, items):
        # Vectorised version of SVD.estimate: mu + b_u + b_i + q_i . p_u, clipped to the rating scale.
        # Items unseen in training fall back to mu + b_u, exactly as Surprise does.
        ts, algo = self.trainset, self.algo
        est = np.full(len(items), ts.global_mean + algo.bu[inner_uid])
        inner_items = np.array([ts._raw2inner_id_items.get(i, -1) for i in items])
        known = inner_items >= 0
        ii = inner_items[known]
        est[known] += algo.bi[ii] + algo.qi[ii] @ algo.pu[inner_uid]
        return np.clip(est, *self.rating_scale)
