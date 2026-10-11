"""Offline evaluation: hold out part of each user's history and rank the full catalogue.

Every model is scored with the same protocol, so the numbers are directly comparable
(the original notebooks used a different protocol per model; see the README).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from hgrs.models.base import Recommender


def precision_recall_ndcg(recommended: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    top = recommended[:k]
    gains = np.array([1.0 if a in relevant else 0.0 for a in top])
    hits = gains.sum()
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    ideal = (1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
    ndcg = float((gains * discounts).sum() / ideal) if ideal > 0 else 0.0
    return hits / k, hits / len(relevant), ndcg


def relevant_items(test: pd.DataFrame) -> dict[str, set[str]]:
    """A held-out game counts as relevant when the user recommended it."""
    liked = test[test["is_recommended"] >= 1.0]
    return {u: set(g["app_id"]) for u, g in liked.groupby("user_id")}


def evaluate(
    models: Mapping[str, Recommender],
    test: pd.DataFrame,
    k: int = 10,
    n_users: int | None = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Precision@k, Recall@k, NDCG@k and HitRate@k for already-fitted models."""
    relevant = relevant_items(test)
    users = sorted(relevant)
    if n_users and len(users) > n_users:
        users = sorted(np.random.default_rng(seed).choice(users, size=n_users, replace=False).tolist())

    rows = []
    for name, model in models.items():
        p, r, n, h = [], [], [], []
        for u in users:
            recs = list(model.recommend(u, k).index)
            prec, rec, ndcg = precision_recall_ndcg(recs, relevant[u], k)
            p.append(prec)
            r.append(rec)
            n.append(ndcg)
            h.append(float(prec > 0))
        rows.append(
            {
                "model": name,
                f"precision@{k}": np.mean(p),
                f"recall@{k}": np.mean(r),
                f"ndcg@{k}": np.mean(n),
                f"hit_rate@{k}": np.mean(h),
                "users": len(users),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    """Render a metrics table as GitHub markdown without needing the tabulate package."""
    cols = list(df.columns)
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name in df.index:
        cells = [
            f"{df.at[name, c]:.{digits}f}" if df[c].dtype.kind == "f" else str(df.at[name, c]) for c in cols
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
