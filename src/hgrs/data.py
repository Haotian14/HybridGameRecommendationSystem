"""Loading and preparing the Steam interaction log and game catalogue."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from hgrs.config import Config

INTERACTION_COLUMNS = ["user_id", "app_id", "hours", "is_recommended"]
GAME_TEXT_COLUMNS = ["About the game", "Tags", "genre", "Categories", "Developers", "Publishers"]
GAME_COLUMNS = ["app_id", "Name", *GAME_TEXT_COLUMNS, "Price"]  # Name is optional, display only

_TRUE_FALSE = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}


# --------------------------------------------------------------------------- user history


@dataclass(frozen=True)
class UserHistory:
    """What one user played: aligned arrays of game ids, hours and recommendations.

    Every model scores users through this one type, whether the user came from the
    training data or is a brand-new visitor who just picked a few games.
    """

    app_ids: np.ndarray
    hours: np.ndarray
    recommended: np.ndarray

    def __len__(self) -> int:
        return len(self.app_ids)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> UserHistory:
        return cls(
            app_ids=df["app_id"].to_numpy(dtype=object),
            hours=df["hours"].to_numpy(dtype=float),
            recommended=df["is_recommended"].to_numpy(dtype=float),
        )

    @classmethod
    def from_liked(cls, app_ids: Sequence[str], hours: float = 20.0) -> UserHistory:
        """A new user who just says "I like these games"."""
        n = len(app_ids)
        return cls(
            app_ids=np.array([str(a) for a in app_ids], dtype=object),
            hours=np.full(n, hours),
            recommended=np.ones(n),
        )

    def head(self, n: int) -> UserHistory:
        return UserHistory(self.app_ids[:n], self.hours[:n], self.recommended[:n])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"app_id": self.app_ids, "hours": self.hours, "is_recommended": self.recommended}
        )


def histories_by_user(interactions: pd.DataFrame) -> dict[str, UserHistory]:
    app_ids = interactions["app_id"].to_numpy(dtype=object)
    hours = interactions["hours"].to_numpy(dtype=float)
    recommended = interactions["is_recommended"].to_numpy(dtype=float)
    return {
        str(user): UserHistory(app_ids[rows], hours[rows], recommended[rows])
        for user, rows in interactions.groupby("user_id").indices.items()
    }


# --------------------------------------------------------------------------- loading


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
        s = s.astype(str).str.strip().str.lower().map(_TRUE_FALSE)
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


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (interactions including the test profiles, game catalogue)."""
    interactions = load_interactions(cfg.data.interactions)
    real = [p for p in cfg.data.real_users if Path(p).exists()]
    if real:
        profiles = load_real_users(real, encoding=cfg.data.real_user_encoding)
        interactions = pd.concat([interactions, profiles], ignore_index=True)
    return interactions, load_games(cfg.data.games)


# --------------------------------------------------------------------------- preparation


def filter_active(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Keep users with >= min_user interactions and games with >= min_item interactions.

    Both thresholds are counted on the unfiltered log, matching how filtered_games.csv was built.
    """
    users = df["user_id"].value_counts()
    items = df["app_id"].value_counts()
    active_user = df["user_id"].isin(users[users >= min_user].index)
    popular_item = df["app_id"].isin(items[items >= min_item].index)
    return df[active_user & popular_item].reset_index(drop=True)


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


# --------------------------------------------------------------------------- splits


def split_per_user(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out ``test_size`` of every user's interactions so each user appears in both sets."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(df), dtype=bool)
    for rows in df.groupby("user_id").indices.values():
        n_test = round(len(rows) * test_size)
        if len(rows) < 2 or n_test == 0:
            continue
        test_mask[rng.choice(rows, size=min(n_test, len(rows) - 1), replace=False)] = True
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def split_cold_start_users(
    df: pd.DataFrame, n_users: int, min_interactions: int, seed: int
) -> tuple[pd.DataFrame, dict[str, UserHistory]]:
    """Remove ``n_users`` users from the training data entirely.

    Returns the training interactions and, for each removed user, their full history in
    random order. The cold-start experiment then reveals the first n games of that history
    and asks the models to predict the rest, as if a new user had just signed up.
    """
    counts = df["user_id"].value_counts()
    eligible = sorted(counts[counts >= min_interactions].index)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))

    cold = df[df["user_id"].isin(chosen)]
    histories = {}
    for user, rows in cold.groupby("user_id"):
        shuffled = rows.sample(frac=1.0, random_state=int(rng.integers(2**31)))
        histories[str(user)] = UserHistory.from_frame(shuffled)
    train = df[~df["user_id"].isin(chosen)].reset_index(drop=True)
    return train, histories
