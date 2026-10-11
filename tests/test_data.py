import pandas as pd
import pytest

from hgrs.config import Config, load_config
from hgrs.data import (
    UserHistory,
    assign_rating,
    filter_active,
    histories_by_user,
    normalise_interactions,
    split_cold_start_users,
    split_per_user,
)


def test_assign_rating_matches_original_rules():
    df = pd.DataFrame({"is_recommended": [1, 1, 1, 0, 0], "hours": [30, 12, 2, 6, 1]})
    assert assign_rating(df).tolist() == [5, 4, 3, 2, 1]


def test_normalise_interactions_coerces_types():
    raw = pd.DataFrame(
        {
            "user_id": [1, 2],
            "app_id": [10, 20],
            "hours": ["3.5", None],
            "is_recommended": ["True", "False"],
            "unused": [0, 0],
        }
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
    assert out.to_dict("records") == [{"user_id": "a", "app_id": "x"}] * 2


def test_user_history_round_trip(dataset):
    interactions, _ = dataset
    histories = histories_by_user(interactions)
    assert sum(len(h) for h in histories.values()) == len(interactions)
    liked = UserHistory.from_liked(["1", "2"])
    assert liked.recommended.tolist() == [1.0, 1.0]
    assert len(liked.head(1)) == 1
    assert list(liked.to_frame().columns) == ["app_id", "hours", "is_recommended"]


def test_split_per_user_keeps_every_user_in_train(dataset):
    interactions, _ = dataset
    train, test = split_per_user(interactions, test_size=0.2, seed=0)
    assert len(train) + len(test) == len(interactions)
    assert set(train["user_id"]) == set(interactions["user_id"])
    again, _ = split_per_user(interactions, test_size=0.2, seed=0)
    pd.testing.assert_frame_equal(train, again)


def test_split_cold_start_users_removes_them_from_training(dataset):
    interactions, _ = dataset
    train, histories = split_cold_start_users(interactions, n_users=5, min_interactions=15, seed=0)
    assert len(histories) == 5
    assert not set(histories) & set(train["user_id"])
    assert len(train) + sum(len(h) for h in histories.values()) == len(interactions)


def test_load_games_deduplicates(dataset):
    _, games = dataset
    assert games["app_id"].is_unique
    assert {"Name", "About the game", "genre", "Price"} <= set(games.columns)


def test_config_resolves_paths_and_rejects_typos(synthetic_root):
    cfg = load_config(synthetic_root / "config.yaml", root=synthetic_root)
    assert cfg.data.interactions == synthetic_root / "interactions.csv"
    assert cfg.pipeline.stages[-1] == ("svd", 10)
    with pytest.raises(TypeError):
        Config.from_dict({"data": {"interactions": "a", "games": "b"}, "svd": {"n_factor": 5}})
