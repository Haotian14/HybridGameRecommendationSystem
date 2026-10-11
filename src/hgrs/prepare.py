"""Rebuild the project's data files from the two public source datasets.

Sources (see ``data/README.md`` for licences and details):

* **Game Recommendations on Steam** (Kaggle, CC0), version 28: ``recommendations.csv``,
  41M (user, game, hours, recommended) reviews.
* **Steam Games Dataset** (Hugging Face mirror of the Kaggle dataset), revision
  ``6ba5504`` of 2025-04-26, the version the project used: ``games.csv``, game metadata.

Pipeline, as in the original project:

1. attach each game's genres to every review, dropping games with no genre;
2. keep users with >= 100 reviews and games with >= 1,000 reviews, both counted
   *before* filtering;
3. write ``filtered_games.csv`` (interactions) and ``games.csv`` (one row per kept game).
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hgrs.data import filter_active

STEAM_GAMES_REVISION = "6ba55046b084aba5cbd62c64ce38fa7182058e42"


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    sha256: str | None
    how_to_get: str


STEAM_GAMES = SourceFile(
    name="steam_games.csv",
    url=(
        "https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/"
        f"{STEAM_GAMES_REVISION}/games.csv"
    ),
    sha256="c621746e5c6d60bd193d7748ed4b32d06c6e25d1bad81fd4615d24f74b7a2ce5",
    how_to_get="downloaded automatically from Hugging Face (public, no login)",
)
RECOMMENDATIONS = SourceFile(
    name="recommendations.csv",
    url="https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/versions/28",
    sha256="18731b9e253b63ad8642dde48c212620621c33080d7609fa092a308caaffbb34",
    how_to_get=(
        "download recommendations.csv from Kaggle (free account needed) and put it in the "
        "source directory"
    ),
)

# What the original project's files contained. ``hgrs prepare`` checks its output against these.
# The fingerprint is a sha256 of the interactions in a canonical order and format, so it does
# not depend on row order or on how a pandas version writes floats.
EXPECTED = {
    "interactions": 950_355,
    "users": 8_225,
    "games": 3_654,
    "fingerprint": "c2201fd42a787206f51f34c1efab4fe052a1edd40162733a93e1d39284bb874b",
}

GAME_COLUMNS = {
    "AppID": "app_id",
    "Name": "Name",
    "Genres": "genre",
    "About the game": "About the game",
    "Tags": "Tags",
    "Categories": "Categories",
    "Developers": "Developers",
    "Publishers": "Publishers",
    "Price": "Price",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source: SourceFile, directory: Path) -> Path:
    """Return the local copy of ``source``, downloading it when that is possible."""
    path = directory / source.name
    if not path.exists():
        if "huggingface.co" not in source.url:
            raise FileNotFoundError(
                f"{path} is missing: {source.how_to_get}.\n  Source: {source.url}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {source.url}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(source.url, partial)
        partial.rename(path)
    if source.sha256 and sha256_of(path) != source.sha256:
        raise ValueError(f"{path} does not match the expected checksum; re-download it")
    return path


def read_steam_games(path: Path) -> pd.DataFrame:
    """Read the Steam Games CSV, repairing its header.

    The header fuses two column names into ``DiscountDLC count``, so every name after
    ``Price`` is shifted one column to the left of its data. The original project fixed
    it the same way (its ``fixed_file.csv``).
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline().replace("DiscountDLC count", "Discount,DLC count")
        text = header + f.read()
    games = pd.read_csv(io.StringIO(text), usecols=list(GAME_COLUMNS), dtype={"AppID": str})
    return games.rename(columns=GAME_COLUMNS).drop_duplicates("app_id")


def read_recommendations(path: Path) -> pd.DataFrame:
    # 41M rows: integer ids keep this around 1 GB in memory, strings would need several.
    return pd.read_csv(
        path,
        usecols=["app_id", "is_recommended", "hours", "user_id"],
        dtype={"app_id": "int64", "user_id": "int64", "hours": "float64", "is_recommended": bool},
    )


def build(
    recommendations: pd.DataFrame,
    games: pd.DataFrame,
    min_user_interactions: int = 100,
    min_game_interactions: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (interactions, catalogue) exactly as the project used them."""
    genres = games.loc[games["genre"].fillna("").str.strip() != "", ["app_id", "genre"]]
    if pd.api.types.is_integer_dtype(recommendations["app_id"]):
        genres = genres.assign(app_id=pd.to_numeric(genres["app_id"], errors="coerce"))
        genres = genres.dropna(subset=["app_id"]).astype({"app_id": "int64"})
    with_genre = recommendations.merge(genres, on="app_id", how="inner")
    interactions = filter_active(with_genre, min_user_interactions, min_game_interactions)
    interactions = interactions[["app_id", "is_recommended", "hours", "user_id", "genre"]].astype(
        {"app_id": str, "user_id": str, "is_recommended": float}
    )
    catalogue = games[games["app_id"].isin(set(interactions["app_id"]))].reset_index(drop=True)
    return interactions.reset_index(drop=True), catalogue


def fingerprint(interactions: pd.DataFrame) -> str:
    columns = ["user_id", "app_id", "is_recommended", "hours", "genre"]
    canonical = interactions[columns].astype({"user_id": str, "app_id": str})
    canonical = canonical.sort_values(columns).reset_index(drop=True)
    text = canonical.to_csv(index=False, float_format="%.4f", lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarise(interactions: pd.DataFrame, catalogue: pd.DataFrame) -> dict[str, int | str]:
    return {
        "interactions": len(interactions),
        "users": interactions["user_id"].nunique(),
        "games": len(catalogue),
        "fingerprint": fingerprint(interactions),
    }


def prepare(source_dir: Path, out_dir: Path) -> dict[str, int]:
    steam_games = fetch(STEAM_GAMES, source_dir)
    recommendations = fetch(RECOMMENDATIONS, source_dir)

    print("Reading sources ...")
    games = read_steam_games(steam_games)
    reviews = read_recommendations(recommendations)
    print(f"  {len(reviews):,} reviews, {len(games):,} games with metadata")

    interactions, catalogue = build(reviews, games)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(out_dir / "filtered_games.csv", index=False)
    catalogue.to_csv(out_dir / "games.csv", index=False)

    counts = summarise(interactions, catalogue)
    for key, expected in EXPECTED.items():
        value = counts[key]
        shown = f"{value:>9,}" if isinstance(value, int) else f"{value[:16]}…"
        status = "ok" if value == expected else "differs from the original project"
        print(f"  {key:<12} {shown}  {status}")
    return counts
