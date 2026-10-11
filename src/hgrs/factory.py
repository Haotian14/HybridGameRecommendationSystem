"""Assemble every model described by a config into one ``ModelSuite``."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hgrs.config import Config
from hgrs.hybrid import AdaptiveHybrid, CascadeHybrid, WeightedHybrid
from hgrs.models import (
    BertRecommender,
    GameCatalog,
    KNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    Recommender,
    SVDRecommender,
    TfidfRecommender,
    make_encoder,
)

BASELINES = ("popularity", "random")
BASE_MODELS = ("tfidf", "bert", "knn", "svd")
HYBRIDS = ("pipeline", "weighted", "adaptive")
ALL_MODELS = (*BASELINES, *BASE_MODELS, *HYBRIDS)
# Models that can serve a user they never saw in training (they only need a play history).
COLD_START_MODELS = ("popularity", "random", "tfidf", "bert", "svd", "adaptive")


@dataclass
class ModelSuite:
    """Baselines, base models and the hybrids built from them, sharing fitted parts."""

    models: dict[str, Recommender]

    def __getitem__(self, name: str) -> Recommender:
        return self.models[name]

    def select(self, names: Iterable[str]) -> dict[str, Recommender]:
        return {name: self.models[name] for name in names}

    def fit(self, interactions: pd.DataFrame, only: Iterable[str] | None = None) -> ModelSuite:
        """Fit the baselines and base models (hybrids just reuse them)."""
        wanted = set(only) if only is not None else set(BASELINES + BASE_MODELS)
        for name in (*BASELINES, *BASE_MODELS):
            if name in wanted:
                started = time.perf_counter()
                self.models[name].fit(interactions)
                elapsed = time.perf_counter() - started
                print(f"  fitted {name:<10} {elapsed:6.1f}s", file=sys.stderr)
        return self


def build_models(cfg: Config, catalog: GameCatalog, cache_dir: Path | None = None) -> ModelSuite:
    """Create every model described by ``cfg``. Nothing is fitted yet."""
    base: dict[str, Recommender] = {
        "popularity": PopularityRecommender(),
        "random": RandomRecommender(seed=cfg.seed),
        "tfidf": TfidfRecommender(catalog, cfg.content.tfidf_max_features, cfg.content.tfidf_text),
        "bert": BertRecommender(
            catalog,
            make_encoder(cfg.content.encoder, cfg.seed),
            cfg.content.bert_weights,
            cache_dir,
            similarity=cfg.content.bert_similarity,
        ),
        "knn": KNNRecommender(cfg.knn.k, cfg.knn.sim_name, cfg.knn.user_based),
        "svd": SVDRecommender(
            cfg.svd.n_factors, cfg.svd.reg_all, cfg.svd.fold_in_reg, seed=cfg.seed
        ),
    }

    pipeline = CascadeHybrid([(base[name], n) for name, n in cfg.pipeline.stages])
    weighted = WeightedHybrid(
        candidate_model=base[cfg.weighted.candidate_model],
        n_candidates=cfg.weighted.n_candidates,
        components={name: (base[name], w) for name, w in cfg.weighted.weights.items()},
        top_k=cfg.weighted.top_k,
    )
    adaptive = AdaptiveHybrid(
        content_models=[base[name] for name in cfg.adaptive.content_models],
        cf_model=base[cfg.adaptive.cf_model],
        shrinkage=cfg.adaptive.shrinkage,
    )
    return ModelSuite({**base, "pipeline": pipeline, "weighted": weighted, "adaptive": adaptive})
