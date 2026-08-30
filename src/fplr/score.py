"""Scoring upcoming fixtures with the promoted model.

Future fixtures are scored by appending them to the historical per-fixture table as
rows with no outcome, then running the ordinary feature pipeline over the whole
thing. Every rolling window is backward-looking, so a future row automatically picks
up the form a player carries into it, and no separate "prediction-time" feature path
exists to drift out of sync with the training one.

**Form is held flat across the horizon.** A player's rolling statistics are identical
for every upcoming fixture, because nothing is known about how they will play in the
intervening weeks -- only the fixture context changes (opponent, venue, congestion).
That is the honest representation: a five-gameweek projection is a statement about
fixtures, not about form five weeks from now.

**Availability comes from the live bootstrap**, not from the model. A player flagged
injured or suspended has their projection zeroed regardless of what their form says,
because the model has no way to know about a Thursday scan result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CURRENT_SEASON
from .datasets import load_fixtures, load_player_fixtures
from .features import build_features, feature_columns
from .ingest import latest_snapshot, next_gameweek
from .scoring import Position

#: FPL's own availability flags. Anything other than available carries doubt.
AVAILABLE_STATUS = "a"


def upcoming_fixtures(horizon: int = 5, *, snapshot=None) -> pd.DataFrame:
    """The next `horizon` gameweeks of fixtures, from the live snapshot."""
    snapshot = snapshot or latest_snapshot()
    if snapshot is None:
        raise FileNotFoundError("no snapshot on disk; run `fpl sync` first")

    bootstrap = snapshot.read("bootstrap")
    upcoming = next_gameweek(bootstrap)
    if upcoming is None:
        raise RuntimeError("the season has no next gameweek")

    start = int(upcoming["id"])
    fixtures = load_fixtures(CURRENT_SEASON)
    window = fixtures[
        fixtures["gameweek"].between(start, start + horizon - 1)
    ].copy()
    return window.sort_values(["gameweek", "kickoff_time", "fixture"])


def player_availability(snapshot=None) -> pd.DataFrame:
    """Current squad list with price and availability from the live bootstrap."""
    snapshot = snapshot or latest_snapshot()
    bootstrap = snapshot.read("bootstrap")
    team_codes = {team["id"]: team["code"] for team in bootstrap["teams"]}
    team_names = {team["id"]: team["short_name"] for team in bootstrap["teams"]}

    rows = []
    for element in bootstrap["elements"]:
        chance = element.get("chance_of_playing_next_round")
        rows.append(
            {
                "element": element["id"],
                "player_code": element["code"],
                "web_name": element["web_name"],
                "position": element["element_type"],
                "team": element["team"],
                "team_code": team_codes.get(element["team"]),
                "team_name": team_names.get(element["team"]),
                "price": element["now_cost"] / 10.0,
                "value": element["now_cost"],
                "status": element.get("status"),
                "selected_by_percent": float(element.get("selected_by_percent") or 0),
                # FPL reports None when there is no doubt at all.
                "chance_of_playing": 100.0 if chance is None else float(chance),
            }
        )
    frame = pd.DataFrame(rows)
    frame["is_available"] = (frame["status"] == AVAILABLE_STATUS)
    return frame


def build_future_rows(horizon: int = 5, *, snapshot=None) -> pd.DataFrame:
    """One row per player per upcoming fixture, with no outcome recorded."""
    snapshot = snapshot or latest_snapshot()
    fixtures = upcoming_fixtures(horizon, snapshot=snapshot)
    squad = player_availability(snapshot)

    home = fixtures.rename(columns={"team_h": "team", "team_a": "opponent_team",
                                    "team_h_code": "team_code",
                                    "team_a_code": "opponent_team_code"})
    home["was_home"] = True
    away = fixtures.rename(columns={"team_a": "team", "team_h": "opponent_team",
                                    "team_a_code": "team_code",
                                    "team_h_code": "opponent_team_code"})
    away["was_home"] = False
    sides = pd.concat(
        [
            home[["season", "fixture", "gameweek", "kickoff_time", "team", "team_code",
                  "opponent_team", "opponent_team_code", "was_home"]],
            away[["season", "fixture", "gameweek", "kickoff_time", "team", "team_code",
                  "opponent_team", "opponent_team_code", "was_home"]],
        ],
        ignore_index=True,
    )

    rows = squad.merge(sides, on=["team", "team_code"], how="inner")
    rows["finalised"] = False
    rows["total_points"] = np.nan
    return rows


def score_upcoming(
    model,
    *,
    horizon: int = 5,
    haul_model=None,
    components_model=None,
    snapshot=None,
) -> pd.DataFrame:
    """Rank every player for the coming gameweeks.

    Returns one row per player per upcoming fixture, plus the availability-adjusted
    projection the agent should actually reason about.
    """
    snapshot = snapshot or latest_snapshot()
    history = load_player_fixtures(finalised_only=False)
    future = build_future_rows(horizon, snapshot=snapshot)

    combined = pd.concat([history, future], ignore_index=True)
    features = build_features(players=combined)

    is_future = features["total_points"].isna() & (features["gameweek"] >= future["gameweek"].min())
    upcoming_rows = features[is_future].copy()

    upcoming_rows["expected_points"] = model.predict(upcoming_rows)
    if haul_model is not None:
        upcoming_rows["p_haul"] = haul_model.predict(upcoming_rows)
    if components_model is not None:
        parts = components_model.predict_components(upcoming_rows)
        for column in parts.columns:
            if column != "expected_points":
                upcoming_rows[f"component_{column}"] = parts[column].to_numpy()

    availability = player_availability(snapshot)[
        ["player_code", "team_name", "status", "chance_of_playing",
         "is_available", "selected_by_percent", "price"]
    ]
    upcoming_rows = upcoming_rows.drop(columns=["price"], errors="ignore").merge(
        availability, on="player_code", how="left"
    )

    # The model cannot know about injuries; FPL's own flag overrides it.
    weight = (upcoming_rows["chance_of_playing"].fillna(100.0) / 100.0).clip(0.0, 1.0)
    upcoming_rows["availability_weight"] = weight
    upcoming_rows["projected_points"] = upcoming_rows["expected_points"] * weight
    if "p_haul" in upcoming_rows:
        upcoming_rows["p_haul_adjusted"] = upcoming_rows["p_haul"] * weight

    upcoming_rows["position_name"] = upcoming_rows["position"].map(
        {int(p): p.name for p in Position}
    )
    return upcoming_rows.sort_values(
        ["gameweek", "projected_points"], ascending=[True, False]
    ).reset_index(drop=True)


def rank_for_gameweek(scored: pd.DataFrame, gameweek: int | None = None) -> pd.DataFrame:
    """Collapse to one row per player for a single gameweek, summing double fixtures."""
    if gameweek is None:
        gameweek = int(scored["gameweek"].min())
    block = scored[scored["gameweek"] == gameweek]

    aggregations = {
        "web_name": "first",
        "position_name": "first",
        "team_name": "first",
        "price": "first",
        "selected_by_percent": "first",
        "status": "first",
        "chance_of_playing": "first",
        "projected_points": "sum",  # a double gameweek is two chances to score
        "expected_points": "sum",
        "fixture": "count",
    }
    if "p_haul_adjusted" in block.columns:
        # Probability of hauling in at least one of the fixtures.
        aggregations["p_haul_adjusted"] = lambda s: 1.0 - float(np.prod(1.0 - s))

    grouped = (
        block.groupby("player_code").agg(**{
            key: (key if not callable(value) else key, value)
            for key, value in aggregations.items()
        }).rename(columns={"fixture": "fixtures"})
    )
    return grouped.sort_values("projected_points", ascending=False).reset_index()
