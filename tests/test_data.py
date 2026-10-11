import pandas as pd
import pytest

from hgrs.data import assign_rating, filter_active, normalise_interactions, split_per_user


def test_assign_rating_matches_original_rules():
    df = pd.DataFrame(
        {
            "is_recommended": [1, 1, 1, 0, 0],
            "hours": [30, 12, 2, 6, 1],
        }
    )
    assert assign_rating(df).tolist() == [5, 4, 3, 2, 1]


def test_normalise_interactions_coerces_types():
    raw = pd.DataFrame(
        {"user_id": [1, 2], "app_id": [10, 20], "hours": ["3.5", None], "is_recommended": ["True", "False"], "x": [0, 0]}
    )
    out = normalise_interactions(raw)
    assert list(out.columns) == ["user_id", "app_id", "hours", "is_recommended"]
    assert out["user_id"].tolist() == ["1", "2"]
    assert out["hours"].tolist() == [3.5, 0.0]
    assert out["is_recommended"].tolist() == [1.0, 0.0]


def test_normalise_interactions_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        normalise_interactions(pd.DataFrame({"user_id": [1]}))


def test_filter_active_uses_unfiltered_counts():
    df = pd.DataFrame({"user_id": ["a", "a", "a", "b"], "app_id": ["x", "x", "y", "x"]})
    out = filter_active(df, min_user=2, min_item=2)
    assert out.to_dict("records") == [{"user_id": "a", "app_id": "x"}, {"user_id": "a", "app_id": "x"}]


def test_split_per_user_keeps_every_user_in_train(dataset):
    interactions, _ = dataset
    train, test = split_per_user(interactions, test_size=0.2, seed=0)
    assert len(train) + len(test) == len(interactions)
    assert set(train["user_id"]) == set(interactions["user_id"])
    # deterministic
    again, _ = split_per_user(interactions, test_size=0.2, seed=0)
    pd.testing.assert_frame_equal(train, again)


def test_load_games_deduplicates(dataset):
    _, games = dataset
    assert games["app_id"].is_unique
    assert {"Name", "About the game", "genre", "Price"} <= set(games.columns)
