"""Loading and preparing the Steam interaction log and game catalogue."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

INTERACTION_COLUMNS = ["user_id", "app_id", "hours", "is_recommended"]
GAME_TEXT_COLUMNS = ["About the game", "Tags", "genre", "Categories", "Developers", "Publishers"]
GAME_COLUMNS = ["app_id", "Name", *GAME_TEXT_COLUMNS, "Price"]  # Name is optional, used for display only


def normalise_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the core columns with consistent types: string ids, float hours, 0/1 recommendation."""
    missing = set(INTERACTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"interaction data is missing columns: {sorted(missing)}")
    out = df[INTERACTION_COLUMNS].copy()
    out["user_id"] = out["user_id"].astype(str).str.strip()
    out["app_id"] = out["app_id"].astype(str).str.strip()
    out["hours"] = pd.to_numeric(out["hours"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["is_recommended"] = _to_binary(out["is_recommended"])
    return out.reset_index(drop=True)


def _to_binary(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        s = s.astype(str).str.strip().str.lower().map(
            {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
        )
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def load_interactions(path: str | Path, encoding: str | None = None) -> pd.DataFrame:
    return normalise_interactions(pd.read_csv(path, encoding=encoding))


def load_real_users(paths: Iterable[str | Path], encoding: str | None = None) -> pd.DataFrame:
    frames = [load_interactions(p, encoding=encoding) for p in paths]
    if not frames:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_games(path: str | Path, chunksize: int = 200_000) -> pd.DataFrame:
    """Load one metadata row per game.

    The original ``merged_filtered_games.csv`` repeats each game's metadata on every
    interaction row (1.9 GB), so it is read in chunks and de-duplicated on the fly.
    """
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in GAME_COLUMNS if c in header]
    if "app_id" not in usecols:
        raise ValueError(f"{path} has no app_id column")
    seen: set[str] = set()
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype={"app_id": str}):
        chunk = chunk.drop_duplicates("app_id")
        chunk = chunk[~chunk["app_id"].isin(seen)]
        seen.update(chunk["app_id"])
        parts.append(chunk)
    return normalise_games(pd.concat(parts, ignore_index=True))


def normalise_games(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates("app_id").copy()
    out["app_id"] = out["app_id"].astype(str).str.strip()
    for col in GAME_TEXT_COLUMNS:
        out[col] = out[col].fillna("").astype(str) if col in out else ""
    price = out["Price"] if "Price" in out else pd.Series(0.0, index=out.index)
    out["Price"] = pd.to_numeric(price, errors="coerce").fillna(0.0)
    return out.reset_index(drop=True)


def filter_active(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Keep users with >= min_user interactions and games with >= min_item interactions.

    Both thresholds are counted on the unfiltered log, matching how filtered_games.csv was built.
    """
    users = df["user_id"].value_counts()
    items = df["app_id"].value_counts()
    keep = df["user_id"].isin(users[users >= min_user].index) & df["app_id"].isin(items[items >= min_item].index)
    return df[keep].reset_index(drop=True)


def assign_rating(df: pd.DataFrame) -> pd.Series:
    """Map (is_recommended, hours) to an explicit 1-5 rating for SVD.

    5: recommended and >= 25 h   4: recommended and >= 10 h   3: recommended
    2: not recommended but >= 5 h                              1: otherwise
    """
    rec = df["is_recommended"].astype(bool).to_numpy()
    hours = df["hours"].to_numpy()
    rating = np.select(
        [rec & (hours >= 25), rec & (hours >= 10), rec, hours >= 5],
        [5, 4, 3, 2],
        default=1,
    )
    return pd.Series(rating.astype(float), index=df.index, name="rating")


def split_per_user(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``test_size`` of every user's interactions so each user appears in both sets."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(df), dtype=bool)
    for idx in df.groupby("user_id").indices.values():
        n_test = int(round(len(idx) * test_size))
        if len(idx) < 2 or n_test == 0:
            continue
        test_mask[rng.choice(idx, size=min(n_test, len(idx) - 1), replace=False)] = True
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (interactions including real users, game catalogue) as described by the config."""
    data = cfg["data"]
    interactions = load_interactions(data["interactions"])
    real = [p for p in data.get("real_users", []) if Path(p).exists()]
    if real:
        interactions = pd.concat(
            [interactions, load_real_users(real, encoding=data.get("real_user_encoding"))],
            ignore_index=True,
        )
    return interactions, load_games(data["games"])
