"""The game catalogue and the features shared by every content model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from hgrs.data import GAME_TEXT_COLUMNS, normalise_games


def split_tags(value: str) -> list[str]:
    """``"Action, RPG;Indie"`` -> ``["Action", "RPG", "Indie"]``."""
    return [tag.strip() for tag in str(value).replace(";", ",").split(",") if tag.strip()]


class GameCatalog:
    """One row per game, plus a genre matrix and normalised prices.

    Positions in the catalogue are the row indices used by every feature matrix.
    """

    def __init__(self, games: pd.DataFrame):
        self.games = normalise_games(games)
        self.app_ids = self.games["app_id"].to_numpy(dtype=object)
        self.position = {app_id: i for i, app_id in enumerate(self.app_ids)}

        genres = self.games["genre"].map(split_tags)
        self.genre_matrix = MultiLabelBinarizer().fit_transform(genres).astype(float)
        self.price_norm = _min_max(self.games["Price"].to_numpy(dtype=float))

    def __len__(self) -> int:
        return len(self.app_ids)

    def __contains__(self, app_id: object) -> bool:
        return app_id in self.position

    def positions(self, app_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Catalogue positions of the known ``app_ids`` and a mask of which ones were known."""
        known = np.array([a in self.position for a in app_ids], dtype=bool)
        found = [self.position[a] for a, ok in zip(app_ids, known, strict=True) if ok]
        return np.array(found, dtype=int), known

    def names(self) -> pd.Series | None:
        if "Name" not in self.games:
            return None
        return self.games.set_index("app_id")["Name"]

    def text(self, kind: str = "combined") -> list[str]:
        """The text the content models read: the description only, or every text field."""
        if kind == "about":
            return self.games["About the game"].tolist()
        if kind == "combined":
            return self.games[GAME_TEXT_COLUMNS].agg(" ".join, axis=1).tolist()
        raise ValueError(f"unknown text kind {kind!r}; use 'combined' or 'about'")


def _min_max(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or values.max() == values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())
