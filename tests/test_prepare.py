import pandas as pd
import pytest

from hgrs.prepare import RECOMMENDATIONS, build, fetch, read_steam_games

# The real file's header fuses "Discount" and "DLC count", so names after Price are shifted.
STEAM_GAMES_CSV = """\
AppID,Name,Release date,Estimated owners,Peak CCU,Required age,Price,DiscountDLC count,\
About the game,Supported languages,Full audio languages,Reviews,Header image,Website,\
Support url,Support email,Windows,Mac,Linux,Metacritic score,Metacritic url,User score,\
Positive,Negative,Score rank,Achievements,Recommendations,Notes,Average playtime forever,\
Average playtime two weeks,Median playtime forever,Median playtime two weeks,Developers,\
Publishers,Categories,Genres,Tags,Screenshots,Movies
"""


def steam_row(app_id, name, genres):
    fields = [app_id, name, "Jan 1, 2020", "0 - 20000", 0, 0, 9.99, 0, 0, f"About {name}",
              "[]", "[]", "", "", "", "", "", True, False, False, 0, "", 0, 10, 1, "", 0, 0,
              "", 0, 0, 0, 0, "Dev", "Pub", "Single-player", genres, "Indie", "", ""]  # fmt: skip
    return ",".join(f'"{f}"' if isinstance(f, str) else str(f) for f in fields)


@pytest.fixture
def steam_games_file(tmp_path):
    rows = [steam_row("1", "Alpha", "Action,RPG"), steam_row("2", "Beta", "Sports"),
            steam_row("3", "No Genre", "")]  # fmt: skip
    path = tmp_path / "steam_games.csv"
    path.write_text(STEAM_GAMES_CSV + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_steam_games_header_is_repaired(steam_games_file):
    games = read_steam_games(steam_games_file)
    alpha = games.set_index("app_id").loc["1"]
    assert alpha["About the game"] == "About Alpha"  # not the shifted "DLC count" value
    assert alpha["genre"] == "Action,RPG"
    assert alpha["Developers"] == "Dev" and alpha["Price"] == pytest.approx(9.99)


def test_build_drops_games_without_genre_and_counts_before_filtering(steam_games_file):
    games = read_steam_games(steam_games_file)
    reviews = pd.DataFrame(
        {
            "app_id": ["1", "1", "1", "2", "2", "3", "3", "3"],
            "user_id": ["a", "b", "c", "a", "b", "a", "b", "c"],
            "hours": [1.0] * 8,
            "is_recommended": [True, False, True, True, True, True, True, True],
        }
    )
    # user "a" has 2 reviews of games with a genre, game "1" has 3, game "2" has 2
    interactions, catalogue = build(
        reviews, games, min_user_interactions=2, min_game_interactions=3
    )
    assert set(interactions["app_id"]) == {"1"}  # game 2 has too few reviews, game 3 no genre
    assert set(interactions["user_id"]) == {"a", "b"}  # user c reviewed only one game with a genre
    assert list(interactions.columns) == ["app_id", "is_recommended", "hours", "user_id", "genre"]
    assert interactions["is_recommended"].tolist() == [1.0, 0.0]
    assert catalogue["app_id"].tolist() == ["1"]


def test_missing_kaggle_file_explains_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        fetch(RECOMMENDATIONS, tmp_path)


def test_fingerprint_ignores_row_order_and_float_formatting():
    from hgrs.prepare import fingerprint

    a = pd.DataFrame(
        {
            "user_id": ["1", "2"],
            "app_id": ["10", "20"],
            "is_recommended": [1.0, 0.0],
            "hours": [3.5, 0.10000000001],
            "genre": ["Action", "RPG"],
        }
    )
    b = a.iloc[::-1].assign(hours=[0.1, 3.5])
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(a.assign(genre=["Action", "Sports"]))
