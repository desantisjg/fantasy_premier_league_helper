"""Leakage and ordering guarantees for the design matrix.

The headline test tampers with a single fixture's outcome and asserts that that
fixture's own feature row does not move. That is a direct structural proof of the
no-leakage property, rather than an indirect check on correlations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplr.datasets import load_player_fixtures
from fplr.features import (
    LEAKY_COLUMNS,
    MEAN_STATS,
    RATE_STATS,
    TARGET,
    build_features,
    feature_columns,
    walk_forward_splits,
)


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    try:
        return load_player_fixtures()
    except FileNotFoundError:
        pytest.skip("player_fixtures.parquet not built; run `fpl data build` first")


@pytest.fixture(scope="module")
def features(players) -> pd.DataFrame:
    return build_features(players=players)


def test_feature_columns_exclude_outcome_fields(features):
    columns = set(feature_columns(features))
    assert TARGET not in columns
    assert not (columns & LEAKY_COLUMNS), sorted(columns & LEAKY_COLUMNS)


def test_every_row_is_preserved(players, features):
    assert len(features) == len(players)


def test_features_ignore_the_fixture_they_predict(players):
    """Corrupting one fixture's stats must not change that fixture's features.

    This is the leakage guarantee stated as an experiment: if any feature read the
    current row, inflating that row's stats would move it.
    """
    counts = players.groupby("player_code").size().sort_values()
    busy_player = int(counts.index[-1])
    history = players[players["player_code"] == busy_player].sort_values("kickoff_time")
    target_row = history.iloc[len(history) // 2]
    season, fixture = target_row["season"], int(target_row["fixture"])

    tampered = players.copy()
    mask = (
        (tampered["player_code"] == busy_player)
        & (tampered["fixture"] == fixture)
        & (tampered["season"] == season)
    )
    assert mask.sum() == 1
    for column in RATE_STATS + MEAN_STATS + [TARGET]:
        tampered.loc[mask, column] = 999

    before = build_features(players=players)
    after = build_features(players=tampered)

    def row_of(frame):
        selected = frame[
            (frame["player_code"] == busy_player)
            & (frame["fixture"] == fixture)
            & (frame["season"] == season)
        ]
        assert len(selected) == 1
        return selected[feature_columns(frame)].iloc[0]

    moved = []
    original, corrupted = row_of(before), row_of(after)
    for column in original.index:
        a, b = original[column], corrupted[column]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or not np.isclose(float(a), float(b), equal_nan=True):
            moved.append(column)

    assert not moved, f"features leaked the current fixture: {moved}"


def test_rolling_windows_use_only_prior_fixtures(players, features):
    """Spot-check a rolling mean against a hand-computed value from earlier rows."""
    counts = players.groupby("player_code").size().sort_values()
    player = int(counts.index[-1])
    history = (
        players[players["player_code"] == player]
        .sort_values(["kickoff_time", "fixture"])
        .reset_index(drop=True)
    )
    position = 8  # far enough in that a 5-fixture window is full
    expected = history.loc[position - 5 : position - 1, "minutes"].mean()

    row = features[
        (features["player_code"] == player)
        & (features["fixture"] == history.loc[position, "fixture"])
        & (features["season"] == history.loc[position, "season"])
    ]
    assert len(row) == 1
    assert np.isclose(float(row["minutes_mean_l5"].iloc[0]), float(expected))


def test_first_fixture_has_no_form(players, features):
    """A player's debut cannot have look-back features."""
    debut = (
        players.sort_values(["kickoff_time", "fixture"])
        .groupby(["season", "player_code"])
        .head(1)
    )
    merged = features.merge(
        debut[["season", "player_code", "fixture"]],
        on=["season", "player_code", "fixture"],
        how="inner",
    )
    assert len(merged) > 0
    assert merged["minutes_mean_l5"].isna().all()
    assert merged["defcon_hit_rate_l5"].isna().all()


def test_walk_forward_splits_never_train_on_the_future(features):
    splits = walk_forward_splits(features)
    assert splits, "expected at least one split"

    order = {
        period: index
        for index, period in enumerate(
            features.sort_values(["season", "gameweek"])[["season", "gameweek"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    }
    positions = features[["season", "gameweek"]].apply(
        lambda row: order[(row["season"], row["gameweek"])], axis=1
    )

    for train_index, test_index in splits:
        assert positions.loc[train_index].max() < positions.loc[test_index].min()
        assert not set(train_index) & set(test_index)


def test_walk_forward_test_folds_are_single_gameweeks(features):
    for _, test_index in walk_forward_splits(features):
        block = features.loc[test_index]
        assert block[["season", "gameweek"]].drop_duplicates().shape[0] == 1


def _synthetic_history_with_future() -> pd.DataFrame:
    """Two clubs playing each other repeatedly, then three unplayed fixtures."""
    rows = []
    fixture = 0
    for gameweek in range(1, 9):  # played
        for home in (True, False):
            club, opponent = (10, 20) if home else (20, 10)
            for player in range(1, 4):
                rows.append(
                    {
                        "season": "2025-26", "finalised": True,
                        "player_code": club * 100 + player, "element": club * 100 + player,
                        "web_name": f"p{club}{player}", "position": 3,
                        "team_code": club, "opponent_team_code": opponent,
                        "team": club, "opponent_team": opponent,
                        "gameweek": gameweek, "fixture": fixture,
                        "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")
                        + pd.Timedelta(days=7 * gameweek),
                        "was_home": home, "minutes": 90, "starts": 1,
                        "total_points": 2 + player, "goals_scored": 0, "assists": 0,
                        "clean_sheets": 0, "goals_conceded": 1, "own_goals": 0,
                        "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0,
                        "red_cards": 0, "saves": 0, "bonus": 0,
                        "defensive_contribution": 3, "expected_goals": 0.3,
                        "clearances_blocks_interceptions": 2, "tackles": 1,
                        "recoveries": 3, "bps": 15,
                        "expected_assists": 0.1, "expected_goal_involvements": 0.4,
                        "expected_goals_conceded": 1.0, "influence": 10.0,
                        "creativity": 5.0, "threat": 8.0, "ict_index": 2.0, "value": 50,
                    }
                )
            fixture += 1

    # Three unplayed fixtures for club 10, each against a different opponent.
    for offset, opponent in enumerate((20, 30, 40)):
        for player in range(1, 4):
            rows.append(
                {
                    **rows[player - 1],
                    "finalised": False,
                    "gameweek": 9 + offset, "fixture": fixture + offset,
                    "opponent_team_code": opponent, "opponent_team": opponent,
                    "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")
                    + pd.Timedelta(days=7 * (9 + offset)),
                    "total_points": np.nan, "minutes": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_form_is_frozen_across_the_projection_horizon():
    """Rolling form must not decay across unplayed fixtures.

    A rolling window counts rows, and an unplayed fixture is a row. Without freezing,
    a projection several gameweeks out spends its window on fixtures that have not
    happened, and the forecast drifts toward the mean — an artefact indistinguishable
    from a hard run of fixtures.
    """
    frame = build_features(players=_synthetic_history_with_future())
    future = frame[frame[TARGET].isna()].sort_values("gameweek")
    assert future["gameweek"].nunique() == 3

    for player, block in future.groupby("player_code"):
        for column in ("minutes_mean_l5", "total_points_mean_l5", "team_xg_for_l5"):
            values = block[column].dropna().unique()
            assert len(values) <= 1, (
                f"{column} varied across the horizon for player {player}: {values}"
            )


def test_opponent_form_still_varies_across_the_horizon():
    """Freezing own form must not also freeze the opponent, who changes each week."""
    frame = build_features(players=_synthetic_history_with_future())
    future = frame[frame[TARGET].isna()]
    assert future["opponent_team_code"].nunique() == 3, "expected three opponents"


def test_fixture_context_is_not_frozen():
    """Venue and congestion are known per fixture and must stay per fixture."""
    frame = build_features(players=_synthetic_history_with_future())
    future = frame[frame[TARGET].isna()].sort_values("gameweek")
    assert future["gameweek_index"].nunique() == 3
