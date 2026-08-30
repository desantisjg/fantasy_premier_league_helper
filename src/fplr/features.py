"""Building the model's design matrix.

Every feature describes what was knowable *before* a fixture kicked off, and each
row's target is that same fixture's actual points. Framing it this way -- rather
than as "features now, points next gameweek" -- keeps the leakage rule trivial to
state and to test: within a row, nothing may come from the fixture itself.

Three sources of leakage are guarded against explicitly:

* **Same-row statistics.** Every rolling window is shifted by one fixture before
  aggregating, so a player's own performance in the match being predicted is never
  visible.
* **Bootstrap convenience fields.** `form`, `points_per_game` and `total_points` on
  the players endpoint are season-to-date figures recomputed after every gameweek.
  Read during a sync that happens after the match, they already contain the answer.
  They are excluded from the canonical schema and must never be added.
* **End-of-season snapshots.** Club and team-strength ratings from season-final
  files encode how the season turned out. Team strength is therefore derived from
  rolling match results instead of taken from FPL's static ratings, which are also
  unusable in practice -- the current bootstrap reports every attack and defence
  rating as zero.

Rates are computed as summed-stat over summed-minutes, not as the mean of per-match
rates, so a substitute's ten-minute cameo cannot dominate a window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import load_fixtures, load_player_fixtures

#: Look-back windows, in prior fixtures.
ROLLING_WINDOWS = (3, 5, 10)

#: Counting stats converted to a per-90 rate over each window.
RATE_STATS = [
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
    "saves",
    "bps",
    "influence",
    "creativity",
    "threat",
]

#: Stats averaged per fixture rather than per 90 minutes.
MEAN_STATS = ["minutes", "total_points", "bonus", "starts"]

#: Team-level match outcomes, averaged per fixture over each window.
TEAM_STATS = ["goals_for", "goals_against", "xg_for", "xg_against"]

TARGET = "total_points"

#: Fixture outcomes carried alongside the features. The decomposed model fits a
#: component against each of these, so they must be present in the frame -- but they
#: are the fixture's own result, so `feature_columns` must never return them. They
#: are all members of LEAKY_COLUMNS, which is what enforces that.
OUTCOME_COLUMNS = [
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "defensive_contribution",
    "starts",
]

#: Never allowed into the design matrix -- these are the fixture's own outcome.
LEAKY_COLUMNS = frozenset(
    RATE_STATS
    + MEAN_STATS
    + [
        "total_points",
        "form",
        "points_per_game",
        "goals_conceded",
        "clean_sheets",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "ict_index",
        "expected_goals_conceded",
    ]
)


def _sorted_for_rolling(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Chronological order within each group; kickoff time breaks double-gameweeks."""
    return frame.sort_values(keys + ["kickoff_time", "fixture"]).reset_index(drop=True)


def _lagged_rolling(
    frame: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    window: int,
    how: str,
) -> pd.DataFrame:
    """Rolling aggregate over the `window` fixtures *preceding* each row.

    The shift happens inside the group and before the window, so the current
    fixture is excluded by construction rather than by a later correction.
    """
    grouped = frame.groupby(keys, sort=False)[columns]
    shifted = grouped.shift(1)
    rolled = (
        shifted.groupby([frame[key] for key in keys], sort=False)
        .rolling(window, min_periods=1)
        .agg(how)
    )
    return rolled.reset_index(level=list(range(len(keys))), drop=True).sort_index()


def build_team_fixtures(players: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """One row per club per match, with the result and shot quality on both sides."""
    grouped = (
        players.groupby(["season", "fixture", "team_code"], as_index=False)
        .agg(
            gameweek=("gameweek", "first"),
            kickoff_time=("kickoff_time", "first"),
            was_home=("was_home", "first"),
            opponent_team_code=("opponent_team_code", "first"),
            xg_for=("expected_goals", "sum"),
        )
    )

    fixtures = pd.concat([load_fixtures(season) for season in seasons], ignore_index=True)
    scores = fixtures.set_index(["season", "fixture"])[["team_h_score", "team_a_score"]]
    joined = grouped.join(scores, on=["season", "fixture"])

    home = joined["was_home"].astype("boolean").fillna(False)
    joined["goals_for"] = joined["team_h_score"].where(home, joined["team_a_score"])
    joined["goals_against"] = joined["team_a_score"].where(home, joined["team_h_score"])

    # A team's xG conceded is its opponent's xG created in the same match.
    opponent_xg = joined.set_index(["season", "fixture", "team_code"])["xg_for"]
    joined["xg_against"] = (
        pd.MultiIndex.from_arrays(
            [joined["season"], joined["fixture"], joined["opponent_team_code"]]
        )
        .map(opponent_xg)
        .to_numpy()
    )

    for column in ("goals_for", "goals_against"):
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    return joined.drop(columns=["team_h_score", "team_a_score"])


def build_team_form(team_fixtures: pd.DataFrame) -> pd.DataFrame:
    """Lagged rolling form for every club, used for both own-team and opponent."""
    frame = _sorted_for_rolling(team_fixtures, ["season", "team_code"])
    out = frame[["season", "fixture", "team_code", "gameweek"]].copy()

    for window in ROLLING_WINDOWS:
        rolled = _lagged_rolling(
            frame, ["season", "team_code"], TEAM_STATS, window, "mean"
        )
        for stat in TEAM_STATS:
            out[f"team_{stat}_l{window}"] = rolled[stat].to_numpy()

    played = _lagged_rolling(frame, ["season", "team_code"], ["goals_for"], 38, "count")
    out["team_matches_played"] = played["goals_for"].to_numpy()
    return out


def build_player_form(players: pd.DataFrame) -> pd.DataFrame:
    """Lagged rolling player form: per-90 rates plus per-fixture averages."""
    frame = _sorted_for_rolling(players, ["season", "player_code"])
    keys = ["season", "player_code"]
    out = frame[["season", "player_code", "fixture"]].copy()

    for window in ROLLING_WINDOWS:
        minutes = _lagged_rolling(frame, keys, ["minutes"], window, "sum")["minutes"]
        totals = _lagged_rolling(frame, keys, RATE_STATS, window, "sum")
        means = _lagged_rolling(frame, keys, MEAN_STATS, window, "mean")

        # Per-90 is undefined with no minutes played; leave it missing rather than
        # imputing a zero that would read as "played a lot, did nothing".
        safe_minutes = minutes.replace(0, np.nan).to_numpy()
        for stat in RATE_STATS:
            out[f"{stat}_p90_l{window}"] = totals[stat].to_numpy() / safe_minutes * 90.0
        for stat in MEAN_STATS:
            out[f"{stat}_mean_l{window}"] = means[stat].to_numpy()

        out[f"minutes_total_l{window}"] = minutes.to_numpy()

    # Threshold-hit rate matters more than the raw count: defensive contribution
    # pays a flat 2 points at the cutoff and nothing for exceeding it.
    from .scoring import DEFENSIVE_CONTRIBUTION_THRESHOLD, Position

    # Ineligible positions get an unreachable threshold rather than a missing one:
    # a goalkeeper's hit rate is genuinely zero, not unknown. Mapping to None would
    # make the whole column NaN for keepers and read as absent information.
    # `fplr.model.defcon_hit` uses the same convention.
    thresholds = frame["position"].map(
        {int(p): (DEFENSIVE_CONTRIBUTION_THRESHOLD[p] or np.inf) for p in Position}
    )
    # `starts` is already 0/1 per fixture, so its rolling mean in MEAN_STATS is
    # exactly a start rate -- no separate feature is needed, and adding one only
    # feeds the collinearity.
    frame = frame.assign(
        _defcon_hit=(frame["defensive_contribution"] >= thresholds).astype(float),
        _played_60=(frame["minutes"].fillna(0) >= 60).astype(float),
    )
    for window in ROLLING_WINDOWS:
        rates = _lagged_rolling(
            frame, keys, ["_defcon_hit", "_played_60"], window, "mean"
        )
        out[f"defcon_hit_rate_l{window}"] = rates["_defcon_hit"].to_numpy()
        out[f"played_60_rate_l{window}"] = rates["_played_60"].to_numpy()

    out["career_fixtures"] = (
        _lagged_rolling(frame, keys, ["minutes"], 380, "count")["minutes"].to_numpy()
    )
    return out


def build_fixture_context(players: pd.DataFrame) -> pd.DataFrame:
    """Circumstances of the fixture that are known before kickoff."""
    frame = _sorted_for_rolling(players, ["season", "player_code"])
    out = frame[["season", "player_code", "fixture"]].copy()

    out["is_home"] = frame["was_home"].astype("boolean").astype(float)
    out["gameweek_index"] = frame["gameweek"].astype(float)

    # Price is set before the deadline, so it is legitimately known in advance.
    out["price"] = frame["value"].astype(float) / 10.0

    previous_kickoff = frame.groupby(["season", "player_code"], sort=False)[
        "kickoff_time"
    ].shift(1)
    out["days_rest"] = (
        (frame["kickoff_time"] - previous_kickoff).dt.total_seconds() / 86400.0
    )

    # Double gameweeks are the weeks with the largest decisions attached, and a
    # player's fixture count is known as soon as the calendar is published.
    counts = frame.groupby(["season", "player_code", "gameweek"])["fixture"].transform("size")
    out["fixtures_in_gameweek"] = counts.astype(float)

    for position in (1, 2, 3, 4):
        out[f"is_position_{position}"] = (frame["position"] == position).astype(float)

    return out


def _freeze_form_across_horizon(
    frame: pd.DataFrame, future: pd.Series, columns: list[str], keys: list[str]
) -> pd.DataFrame:
    """Hold rolling form constant across every unplayed fixture.

    Rolling windows are a fixed number of *rows*, and an unplayed fixture is still a
    row. Left alone, a projection four gameweeks out spends most of its window on
    fixtures that have not happened, so the effective history shrinks and the
    projection decays toward the mean the further ahead you look -- an artefact that
    looks exactly like a hard run of fixtures.

    The first unplayed fixture is the only one whose window is entirely real history,
    so its values are the correct "form carried into the horizon" and are broadcast
    across the rest. Fixture context -- venue, opponent, congestion -- is genuinely
    known per fixture and is deliberately not frozen.
    """
    if not future.any():
        return frame

    ordered = frame[future].sort_values(["gameweek", "kickoff_time", "fixture"])
    carried = ordered.groupby(keys, sort=False)[columns].transform("first")
    frame.loc[carried.index, columns] = carried
    return frame


def build_features(
    *,
    finalised_only: bool = True,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the full design matrix, one row per player-fixture.

    Returns identity columns, the target, and every feature. Rows for a player's
    very first fixture in the data survive with missing form features; the caller
    decides whether to require a minimum history.
    """
    if players is None:
        players = load_player_fixtures(finalised_only=finalised_only)
    players = _sorted_for_rolling(players, ["season", "player_code"])

    seasons = sorted(players["season"].dropna().unique().tolist())
    team_fixtures = build_team_fixtures(players, seasons)
    team_form = build_team_form(team_fixtures)

    player_form = build_player_form(players)
    context = build_fixture_context(players)

    key = ["season", "player_code", "fixture"]
    identity = [
        "season", "player_code", "element", "web_name", "position",
        "team_code", "opponent_team_code", "gameweek", "fixture",
        "kickoff_time", "finalised", "minutes",
    ]
    frame = players[identity + OUTCOME_COLUMNS + [TARGET]].copy()
    frame = frame.merge(player_form, on=key, how="left")
    frame = frame.merge(context, on=key, how="left")

    # Own-team form.
    own = team_form.drop(columns=["gameweek"]).rename(
        columns={c: c for c in team_form.columns}
    )
    frame = frame.merge(
        own, on=["season", "fixture", "team_code"], how="left"
    )

    # Opponent form, same columns re-joined through the opposing club.
    opponent = team_form.drop(columns=["gameweek"]).rename(
        columns={"team_code": "opponent_team_code"}
    )
    opponent = opponent.rename(
        columns={
            c: c.replace("team_", "opp_", 1)
            for c in opponent.columns
            if c.startswith("team_")
        }
    )
    frame = frame.merge(
        opponent, on=["season", "fixture", "opponent_team_code"], how="left"
    )

    # Unplayed fixtures carry the form the player and club take into the horizon.
    future = frame[TARGET].isna()
    player_form_columns = [c for c in player_form.columns if c not in key]
    _freeze_form_across_horizon(
        frame, future, player_form_columns, ["season", "player_code"]
    )
    # Own-club form is carried by the player's club; opponent form is carried by the
    # opponent, which is a different club every gameweek. Freezing both against the
    # player's own club would hold the opponent's strength constant across a horizon
    # whose whole point is that the opponent changes.
    own_club = [c for c in frame.columns if c.startswith("team_") and c != "team_code"]
    _freeze_form_across_horizon(frame, future, own_club, ["season", "team_code"])

    opponent_club = [c for c in frame.columns if c.startswith("opp_")]
    _freeze_form_across_horizon(
        frame, future, opponent_club, ["season", "opponent_team_code"]
    )

    return frame.sort_values(["season", "gameweek", "player_code"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs: everything that is neither identity nor outcome."""
    identity = {
        "season", "player_code", "element", "web_name", "position",
        "team_code", "opponent_team_code", "gameweek", "fixture",
        "kickoff_time", "finalised", "minutes",
    }
    return [
        column
        for column in frame.columns
        if column not in identity
        and column != TARGET
        and column not in LEAKY_COLUMNS
    ]


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    min_train_gameweeks: int = 6,
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-window splits: train on everything before a gameweek, test on it.

    Seasons are ordered, so a split late in 2025/26 trains on that season only,
    while a 2026/27 split trains on all of 2025/26 plus the current season to date.
    A random split would leak the future and flatter the model badly.
    """
    ordered = frame.sort_values(["season", "gameweek"])
    periods = ordered[["season", "gameweek"]].drop_duplicates().to_numpy().tolist()

    splits = []
    for index in range(min_train_gameweeks, len(periods)):
        season, gameweek = periods[index]
        is_test = (frame["season"] == season) & (frame["gameweek"] == gameweek)
        earlier_seasons = frame["season"] < season
        earlier_gameweeks = (frame["season"] == season) & (frame["gameweek"] < gameweek)
        is_train = earlier_seasons | earlier_gameweeks
        if is_train.sum() and is_test.sum():
            splits.append((frame.index[is_train], frame.index[is_test]))
    return splits
