"""Collaborative filtering: KNN and SVD, built on scikit-surprise."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader
from surprise.trainset import Trainset

from hgrs.data import UserHistory, assign_rating
from hgrs.models.base import HistoryRecommender, Recommender, empty_scores, ranked


def build_trainset(
    interactions: pd.DataFrame, ratings: pd.Series, scale: tuple[float, float]
) -> Trainset:
    frame = pd.DataFrame(
        {
            "user_id": interactions["user_id"].astype(str).to_numpy(),
            "app_id": interactions["app_id"].astype(str).to_numpy(),
            "rating": ratings.to_numpy(dtype=float),
        }
    )
    return Dataset.load_from_df(frame, Reader(rating_scale=scale)).build_full_trainset()


def inner_user_id(trainset: Trainset, user_id: str) -> int | None:
    try:
        return trainset.to_inner_uid(str(user_id))
    except ValueError:
        return None


class KNNRecommender(Recommender):
    """Neighbourhood collaborative filtering on the binary ``is_recommended`` signal."""

    name = "knn"
    rating_scale = (0.0, 1.0)

    def __init__(self, k: int = 40, sim_name: str = "msd", user_based: bool = True):
        self.k = k
        self.sim_options = {"name": sim_name, "user_based": user_based}

    def fit(self, interactions: pd.DataFrame) -> KNNRecommender:
        self.trainset = build_trainset(
            interactions, interactions["is_recommended"], self.rating_scale
        )
        self.algo = KNNBasic(k=self.k, sim_options=self.sim_options, verbose=False)
        self.algo.fit(self.trainset)
        return self

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        inner = inner_user_id(self.trainset, user_id)
        if inner is None:
            return empty_scores(self.name)
        if candidates is None:
            rated = {item for item, _ in self.trainset.ur[inner]}
            candidates = [
                self.trainset.to_raw_iid(i) for i in self.trainset.all_items() if i not in rated
            ]
        items = [str(c) for c in candidates]
        if not items:
            return empty_scores(self.name)
        estimates = [self.algo.predict(str(user_id), item).est for item in items]
        return ranked(items, estimates, self.name)


class SVDRecommender(HistoryRecommender):
    """Funk-SVD on a 1-5 rating derived from recommendation and hours played.

    Known users are scored with their trained latent factors. A user unseen in training
    is *folded in*: with item factors held fixed, their bias and latent vector are fitted
    to the games they played by ridge regression, so SVD also works for cold-start users.
    """

    name = "svd"
    rating_scale = (1.0, 5.0)

    def __init__(
        self, n_factors: int = 100, reg_all: float = 0.02, fold_in_reg: float = 1.0, seed: int = 42
    ):
        super().__init__()
        self.n_factors = n_factors
        self.reg_all = reg_all
        self.fold_in_reg = fold_in_reg
        self.seed = seed

    def _fit(self, interactions: pd.DataFrame) -> None:
        self.trainset = build_trainset(interactions, assign_rating(interactions), self.rating_scale)
        self.algo = SVD(n_factors=self.n_factors, reg_all=self.reg_all, random_state=self.seed)
        self.algo.fit(self.trainset)
        self.items = np.array(
            [self.trainset.to_raw_iid(i) for i in self.trainset.all_items()], dtype=object
        )
        self.item_index = {item: i for i, item in enumerate(self.items)}

    def score(self, user_id: str, candidates: Sequence[str] | None = None) -> pd.Series:
        inner = inner_user_id(self.trainset, user_id)
        if inner is None:
            return empty_scores(self.name)
        history = self.histories[str(user_id)]
        return self._rank(self.algo.bu[inner], self.algo.pu[inner], history, candidates)

    def score_history(
        self, history: UserHistory, candidates: Sequence[str] | None = None
    ) -> pd.Series:
        factors = self.fold_in(history)
        if factors is None:
            return empty_scores(self.name)
        return self._rank(*factors, history, candidates)

    def fold_in(self, history: UserHistory) -> tuple[float, np.ndarray] | None:
        """Fit (user bias, user vector) to ``history`` with the item factors held fixed.

        Solves  min_w ||X w - y||² + λ||w||²  with  X = [q_i, 1],  y = r - μ - b_i.
        """
        known = np.array([a in self.item_index for a in history.app_ids], dtype=bool)
        if not known.any():
            return None
        items = np.array([self.item_index[a] for a in history.app_ids[known]])
        ratings = assign_rating(history.to_frame()).to_numpy()[known]

        X = np.hstack([self.algo.qi[items], np.ones((len(items), 1))])
        y = ratings - self.trainset.global_mean - self.algo.bi[items]
        A = X.T @ X + self.fold_in_reg * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        return float(w[-1]), w[:-1]

    def _rank(
        self,
        user_bias: float,
        user_vector: np.ndarray,
        history: UserHistory,
        candidates: Sequence[str] | None,
    ) -> pd.Series:
        if candidates is None:
            played = set(history.app_ids)
            items = [item for item in self.items if item not in played]
        else:
            items = [str(c) for c in candidates]
        if not items:
            return empty_scores(self.name)
        return ranked(items, self._estimate(user_bias, user_vector, items), self.name)

    def _estimate(self, user_bias: float, user_vector: np.ndarray, items: list[str]) -> np.ndarray:
        """Vectorised ``μ + b_u + b_i + q_i·p_u``, clipped to the rating scale.

        Games unseen in training fall back to ``μ + b_u``, as Surprise does.
        """
        estimates = np.full(len(items), self.trainset.global_mean + user_bias)
        index = np.array([self.item_index.get(item, -1) for item in items])
        known = index >= 0
        rows = index[known]
        estimates[known] += self.algo.bi[rows] + self.algo.qi[rows] @ user_vector
        return np.clip(estimates, *self.rating_scale)
