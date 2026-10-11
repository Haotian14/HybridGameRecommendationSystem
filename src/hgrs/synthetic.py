"""A small synthetic Steam-like dataset with the same schema as the real one.

Used by ``hgrs demo`` and the test-suite so the whole system runs in seconds without
downloading several GB of data. Users have one or two favourite genres and mostly play,
recommend and sink hours into games from those genres, so the models have signal to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GENRES = {
    "Action": "shooter combat fast explosive weapons arena",
    "RPG": "quest character story loot dungeon fantasy",
    "Sports": "football basketball league team season stadium",
    "Racing": "cars drift track speed championship circuit",
    "Strategy": "empire tactics resources army turn base",
    "Simulation": "farm city build manage economy realistic",
    "Puzzle": "logic tiles brain riddle levels clever",
    "Horror": "dark survival monsters fear night haunted",
}
REAL_USERS = {
    "11111111": ["Action"],
    "22222222": ["Sports"],
}  # same ids as the real FPS / sports profiles


def make_games(n_games: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(GENRES)
    rows = []
    for i in range(n_games):
        main = names[i % len(names)]
        extra = rng.choice(
            [g for g in names if g != main], size=rng.integers(0, 2), replace=False
        ).tolist()
        genres = [main, *extra]
        words = " ".join(GENRES[g] for g in genres).split()
        rows.append(
            {
                "app_id": str(100000 + i),
                "Name": f"{main} Game {i}",
                "About the game": " ".join(rng.choice(words, size=12)),
                "Tags": ",".join(genres),
                "genre": ",".join(genres),
                "Categories": "Single-player",
                "Developers": f"Studio {i % 7}",
                "Publishers": f"Publisher {i % 3}",
                "Price": float(rng.choice([0, 4.99, 9.99, 19.99, 29.99, 59.99])),
            }
        )
    return pd.DataFrame(rows)


def _history(user_id: str, favourites: list[str], games: pd.DataFrame, n: int, rng) -> list[dict]:
    fav = games["genre"].str.split(",").map(lambda gs: bool(set(gs) & set(favourites))).to_numpy()
    probs = np.where(fav, 6.0, 1.0)
    picks = rng.choice(len(games), size=min(n, len(games)), replace=False, p=probs / probs.sum())
    rows = []
    for i in picks:
        liked = fav[i]
        rows.append(
            {
                "user_id": user_id,
                "app_id": games.at[i, "app_id"],
                "hours": round(float(rng.lognormal(3.0 if liked else 1.0, 0.8)), 1),
                "is_recommended": bool(rng.random() < (0.9 if liked else 0.3)),
            }
        )
    return rows


def make_synthetic(
    out_dir: str | Path, n_users: int = 80, n_games: int = 96, seed: int = 42
) -> Path:
    """Write CSVs and a matching config to ``out_dir``; return the config path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    games = make_games(n_games, rng)

    rows = []
    for u in range(n_users):
        favs = rng.choice(list(GENRES), size=rng.integers(1, 3), replace=False).tolist()
        rows += _history(str(u), favs, games, int(rng.integers(15, 30)), rng)
    interactions = pd.DataFrame(rows)
    interactions.to_csv(out / "interactions.csv", index=False)
    # The real project's metadata file repeats game rows per interaction; mimic that shape.
    interactions[["user_id", "app_id"]].merge(games, on="app_id").to_csv(
        out / "games.csv", index=False
    )

    real_paths = []
    for uid, favs in REAL_USERS.items():
        path = out / f"user_{favs[0].lower()}.csv"
        pd.DataFrame(_history(uid, favs, games, 12, rng)).to_csv(path, index=False)
        real_paths.append(path.name)

    cfg = {
        "seed": seed,
        "data": {
            "interactions": "interactions.csv",
            "games": "games.csv",
            "real_users": real_paths,
            "real_user_encoding": "utf-8",
        },
        "content": {
            "tfidf_max_features": 2000,
            "tfidf_text": "combined",
            "encoder": "lsa",
            "bert_weights": {"content": 0.7, "genre": 0.2, "price": 0.1},
        },
        "knn": {"k": 20, "sim_name": "msd", "user_based": True},
        "svd": {"n_factors": 20, "reg_all": 0.02, "fold_in_reg": 1.0},
        "pipeline": {"stages": [["tfidf", 60], ["bert", 40], ["knn", 20], ["svd", 10]]},
        "weighted": {
            "candidate_model": "svd",
            "n_candidates": 60,
            "weights": {"knn": 0.5, "tfidf": 0.25, "bert": 0.25},
            "top_k": 10,
        },
        "adaptive": {"content_models": ["tfidf", "bert"], "cf_model": "svd", "shrinkage": 10},
        "evaluation": {
            "test_size": 0.2,
            "k": 10,
            "n_users": 50,
            "cold_start_users": 20,
            "cold_start_sizes": [1, 3, 5, 10],
        },
    }
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path
