"""Schema and integrity checks for the normalised per-fixture table.

The headline test here replays an entire archived season through `fplr.scoring`.
The gameweek-level replay in `test_scoring.py` can only cover what happened in the
gameweeks synced so far; a full season additionally exercises the rare events --
penalty saves, red cards, forwards reaching the defensive contribution threshold --
that a couple of gameweeks will not contain.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplr.datasets import (
    CANONICAL_COLUMNS,
    STAT_COLUMNS,
    _archive_cache_path,
    load_archive_season,
)
from fplr.scoring import DEFENSIVE_CONTRIBUTION_THRESHOLD, Position, StatLine, score

ARCHIVE_SEASON = "2025-26"


@pytest.fixture(scope="module")
def archive() -> pd.DataFrame:
    if not _archive_cache_path(ARCHIVE_SEASON, "gws/merged_gw.csv").exists():
        pytest.skip("archive not cached; run `fpl sync --backfill` first")
    return load_archive_season(ARCHIVE_SEASON)


def test_schema_is_canonical(archive):
    assert list(archive.columns) == CANONICAL_COLUMNS


def test_identity_columns_are_complete(archive):
    identity = [
        "player_code", "element", "position", "team_code", "team",
        "opponent_team", "opponent_team_code", "gameweek", "fixture", "web_name",
    ]
    missing = archive[identity].isna().sum()
    assert not missing.any(), f"nulls in identity columns:\n{missing[missing > 0]}"


def test_grain_is_one_row_per_player_fixture(archive):
    key = ["season", "player_code", "gameweek", "fixture"]
    assert not archive.duplicated(key).any()


def test_positions_are_valid(archive):
    assert set(archive["position"].dropna().unique()) <= {1, 2, 3, 4}


def test_replay_reproduces_every_fixture_in_the_season(archive):
    """Our scoring must reproduce FPL's published points for every row."""
    fields = list(StatLine.__dataclass_fields__)
    frame = archive.dropna(subset=["position"])

    computed = [
        score(
            StatLine(**{f: int(row[f]) for f in fields}),
            Position(int(row["position"])),
        )
        for _, row in frame[fields + ["position"]].iterrows()
    ]
    frame = frame.assign(computed=computed)
    wrong = frame[frame["computed"] != frame["total_points"]]

    assert len(frame) > 20_000, "expected a full season of rows"
    assert wrong.empty, (
        f"{len(wrong)} of {len(frame)} fixtures mis-scored; examples:\n"
        + wrong[["gameweek", "web_name", "position", "total_points", "computed"]]
        .head(10)
        .to_string(index=False)
    )


def test_season_exercises_the_rare_scoring_events(archive):
    """Guard against the replay passing only because rare events never occur."""
    assert (archive["penalties_saved"] > 0).any(), "no penalty saves in the season"
    assert (archive["red_cards"] > 0).any(), "no red cards in the season"

    for position in (Position.DEF, Position.MID, Position.FWD):
        threshold = DEFENSIVE_CONTRIBUTION_THRESHOLD[position]
        rows = archive[archive["position"] == int(position)]
        assert (rows["defensive_contribution"] >= threshold).any(), (
            f"no {position.name} reached the DEFCON threshold of {threshold}"
        )


def test_goalkeepers_have_no_defensive_contribution(archive):
    """FPL does not record the metric for keepers, matching our ineligible marker."""
    keepers = archive[archive["position"] == int(Position.GK)]
    assert keepers["defensive_contribution"].max() == 0
    assert DEFENSIVE_CONTRIBUTION_THRESHOLD[Position.GK] is None


def test_double_gameweeks_are_kept_as_separate_rows(archive):
    """Per-fixture grain is what makes the rounding-down rules correct."""
    per_gameweek = archive.groupby(["player_code", "gameweek"]).size()
    assert (per_gameweek > 1).any(), "expected double gameweeks in a full season"


def test_stat_columns_are_numeric(archive):
    for column in STAT_COLUMNS:
        assert pd.api.types.is_numeric_dtype(archive[column]), column


def test_archive_rows_are_all_final(archive):
    """A completed season has nothing provisional left in it."""
    assert archive["finalised"].all()


def test_loader_excludes_provisional_rows_by_default():
    """Training data must never contain a gameweek FPL is still revising."""
    from fplr.datasets import load_player_fixtures

    try:
        trainable = load_player_fixtures()
        everything = load_player_fixtures(finalised_only=False)
    except FileNotFoundError:
        pytest.skip("player_fixtures.parquet not built; run `fpl data build` first")

    assert trainable["finalised"].all()
    assert len(trainable) <= len(everything)
    # The excluded rows are exactly the provisional ones, not an arbitrary subset.
    assert len(everything) - len(trainable) == int(
        (~everything["finalised"].fillna(False)).sum()
    )


def test_each_fixture_has_exactly_two_clubs(archive):
    """Club must come from the fixture, not from a player snapshot.

    `players_raw.csv` records a player's club at the end of the season, so deriving
    club from it stamps a January transfer onto the previous August's fixtures and
    silently corrupts every team-level rolling feature. The scoring replay cannot
    catch this, because club does not affect points.
    """
    per_fixture = archive.groupby("fixture")["team_code"].nunique()
    assert set(per_fixture.unique()) == {2}, (
        f"fixtures with the wrong number of clubs: "
        f"{per_fixture[per_fixture != 2].head().to_dict()}"
    )


def test_opponents_are_mutually_consistent(archive):
    """Each side of a fixture must name the other as its opponent."""
    sides = (
        archive.groupby(["fixture", "team_code"])["opponent_team_code"]
        .first()
        .reset_index()
    )
    paired = sides.merge(sides, on="fixture", suffixes=("", "_other"))
    paired = paired[paired["team_code"] != paired["team_code_other"]]
    mismatched = paired[paired["opponent_team_code"] != paired["team_code_other"]]
    assert mismatched.empty, f"{len(mismatched)} fixtures with inconsistent opponents"


def test_home_and_away_sides_are_distinct(archive):
    """Exactly one club per fixture is at home."""
    home_counts = archive.groupby(["fixture", "team_code"])["was_home"].first()
    per_fixture = home_counts.groupby("fixture").sum()
    assert set(per_fixture.unique()) == {1}
