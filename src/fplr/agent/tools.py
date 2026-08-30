"""Tools the agent can call.

Each one wraps the same function the CLI calls, so there is exactly one
implementation of "score the upcoming gameweek" and the agent cannot drift away
from what `fpl score` reports.

Scoring the horizon takes a few seconds and several tools need the same frame, so
it is computed once per process and reused. The cache is keyed on the horizon and
cleared whenever a new snapshot is synced.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from anthropic import beta_tool

from ..config import FPL_ENTRY_ID
import requests

from ..ingest import FPLClient, latest_snapshot, next_gameweek


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


@lru_cache(maxsize=4)
def _scored(horizon: int):
    from ..score import score_upcoming
    from ..train import load_current_model

    bundle = load_current_model()
    frame = score_upcoming(
        bundle["model"],
        horizon=horizon,
        haul_model=bundle["haul"],
        components_model=bundle["components"],
    )
    return frame, bundle["name"]


def clear_cache() -> None:
    _scored.cache_clear()


@beta_tool
def score_players(
    gameweek: int = 0,
    position: str = "",
    max_price: float = 0.0,
    limit: int = 20,
) -> str:
    """Rank players by projected FPL points for an upcoming gameweek.

    Projections are already adjusted for FPL's published availability flags, so an
    injured player is down-weighted automatically. Double gameweeks are summed.

    Args:
        gameweek: Which gameweek to rank. 0 means the next one.
        position: Filter to GK, DEF, MID or FWD. Empty means all positions.
        max_price: Only players at or below this price in millions. 0 means no cap.
        limit: How many players to return.
    """
    from ..score import rank_for_gameweek

    frame, _ = _scored(5)
    ranked = rank_for_gameweek(frame, gameweek or None)
    if position:
        ranked = ranked[ranked["position_name"] == position.upper()]
    if max_price:
        ranked = ranked[ranked["price"] <= max_price]

    columns = [
        "web_name", "position_name", "team_name", "price", "fixtures",
        "projected_points", "p_haul_adjusted", "chance_of_playing",
        "selected_by_percent",
    ]
    columns = [c for c in columns if c in ranked.columns]
    return _json(ranked.head(limit)[columns].round(3).to_dict(orient="records"))


@beta_tool
def captaincy_candidates(limit: int = 10) -> str:
    """Rank players by the probability of a double-digit haul in the next gameweek.

    Use this for the armband, not `score_players`. Captaincy doubles one player's
    return, so it is a question about the upper tail, not about the average --
    the highest projected scorer is often not the likeliest to haul.

    Args:
        limit: How many candidates to return.
    """
    from ..score import rank_for_gameweek

    frame, _ = _scored(1)
    ranked = rank_for_gameweek(frame)
    if "p_haul_adjusted" not in ranked.columns:
        return _json({"error": "no haul model in the current bundle; run `fpl train`"})
    ranked = ranked.sort_values("p_haul_adjusted", ascending=False)
    columns = ["web_name", "position_name", "team_name", "price",
               "p_haul_adjusted", "projected_points", "selected_by_percent"]
    return _json(ranked.head(limit)[columns].round(3).to_dict(orient="records"))


@beta_tool
def explain_player(name: str) -> str:
    """Break a player's projection into its scoring components and recent form.

    Returns the appearance, attacking, clean-sheet, defensive-contribution and bonus
    parts of the projection, so a recommendation can cite why the number is what it
    is rather than asserting it.

    Args:
        name: The player's FPL display name, or part of it.
    """
    frame, _ = _scored(5)

    # Substring matching alone silently resolves "White" to "Gibbs-White". An exact
    # name wins outright; otherwise ambiguity is reported rather than guessed, since
    # quietly returning the wrong player's numbers is far worse than asking.
    exact = frame[frame["web_name"].str.lower() == name.strip().lower()]
    if not exact.empty:
        matches = exact
    else:
        matches = frame[frame["web_name"].str.contains(name, case=False, na=False, regex=False)]
        if matches.empty:
            return _json({"error": f"no player matching {name!r}"})

        candidates = matches.drop_duplicates("player_code")
        if len(candidates) > 1:
            return _json(
                {
                    "ambiguous": f"{len(candidates)} players match {name!r}",
                    "candidates": candidates[
                        ["web_name", "position_name", "team_name", "price"]
                    ].to_dict(orient="records"),
                    "hint": "Call explain_player again with the exact web_name.",
                }
            )

    player = matches["player_code"].iloc[0]
    rows = frame[frame["player_code"] == player].sort_values("gameweek")

    components = [c for c in rows.columns if c.startswith("component_")]
    form = [
        "minutes_mean_l5", "total_points_mean_l5", "expected_goals_p90_l5",
        "expected_assists_p90_l5", "defcon_hit_rate_l5", "bps_p90_l5",
        "team_xg_for_l5", "team_xg_against_l5",
    ]
    first = rows.iloc[0]
    return _json(
        {
            "player": first["web_name"],
            "position": first["position_name"],
            "team": first["team_name"],
            "price": float(first["price"]),
            "status": first["status"],
            "chance_of_playing": float(first["chance_of_playing"]),
            "recent_form": {c: (None if first.get(c) is None else float(first[c]))
                            for c in form if c in rows.columns},
            "projection_components_next_gameweek": {
                c.removeprefix("component_"): round(float(first[c]), 3)
                for c in components
            },
            "by_gameweek": rows[
                ["gameweek", "is_home", "projected_points", "p_haul"]
            ].round(3).to_dict(orient="records")
            if "p_haul" in rows.columns
            else rows[["gameweek", "is_home", "projected_points"]].round(3).to_dict(orient="records"),
        }
    )


@beta_tool
def get_fixtures(horizon: int = 5) -> str:
    """List upcoming fixtures with each club's recent attacking and defensive form.

    Args:
        horizon: How many gameweeks ahead to list.
    """
    frame, _ = _scored(max(horizon, 1))
    columns = ["gameweek", "team_name", "is_home", "opp_xg_for_l5",
               "opp_xg_against_l5", "opp_goals_against_l5"]
    columns = [c for c in columns if c in frame.columns]
    fixtures = (
        frame[frame["gameweek"] <= frame["gameweek"].min() + horizon - 1]
        .groupby(["gameweek", "team_name"], as_index=False)
        .first()[columns]
    )
    return _json(fixtures.round(3).to_dict(orient="records"))


@beta_tool
def get_model_metrics() -> str:
    """Report how accurate the current model actually is, out of sample.

    Call this before making confident claims. The model ranks better than recent
    form but explains only a small share of the variance in a single gameweek, and
    every recommendation should be framed with that in mind.
    """
    from ..train import load_current_metrics

    metrics = load_current_metrics()
    if metrics is None:
        return _json({"error": "no promoted model; run `fpl train`"})

    model = metrics["primary_model"]
    return _json(
        {
            "model": model,
            "trained": metrics["generated_at"],
            "rows": metrics["n_rows"],
            "features": metrics["n_features"],
            "backtest_folds": metrics["folds"],
            "starters": metrics["performance_starters"][model],
            "best_naive_baseline": {
                "name": metrics["best_baseline"],
                "starters": metrics["performance_starters"][metrics["best_baseline"]],
            },
            "interpretation": (
                "Rank quality is what matters: spearman_weekly is the average "
                "within-gameweek rank correlation among players with real recent "
                "minutes. R^2 near 0.06 is normal for single-gameweek FPL points "
                "and does not mean the model is broken -- most of the variance is "
                "irreducible. Treat projections as a ranking, not as forecasts of "
                "individual scores."
            ),
        }
    )


@beta_tool
def get_my_team(entry_id: int = 0) -> str:
    """Fetch the manager's current squad, bank, free transfers and chips.

    Args:
        entry_id: FPL entry id. 0 uses the FPL_ENTRY_ID environment variable.
    """
    entry = entry_id or (int(FPL_ENTRY_ID) if FPL_ENTRY_ID else 0)
    if not entry:
        return _json(
            {"error": "no entry id; set FPL_ENTRY_ID or pass entry_id"}
        )

    snapshot = latest_snapshot()
    bootstrap = snapshot.read("bootstrap")
    upcoming = next_gameweek(bootstrap)
    names = {e["id"]: e["web_name"] for e in bootstrap["elements"]}
    positions = {e["id"]: e["element_type"] for e in bootstrap["elements"]}
    costs = {e["id"]: e["now_cost"] / 10.0 for e in bootstrap["elements"]}

    client = FPLClient()
    try:
        profile = client.entry(entry)
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status == 404:
            return _json(
                {
                    "error": (
                        f"FPL has no entry {entry}. Find the right id by logging in "
                        "at fantasy.premierleague.com, opening the Points tab, and "
                        "reading the number in the URL: "
                        "fantasy.premierleague.com/entry/<ID>/event/<GW>"
                    )
                }
            )
        return _json({"error": f"could not reach FPL for entry {entry}: {error}"})

    last_finalised = (upcoming["id"] - 1) if upcoming else None
    picks = {}
    if last_finalised:
        try:
            picks = client.entry_picks(entry, last_finalised)
        except requests.HTTPError:
            # A brand-new entry has no picks for a gameweek it did not play.
            picks = {}

    from ..scoring import Position

    squad = [
        {
            "name": names.get(p["element"]),
            "position": Position(positions.get(p["element"], 1)).name,
            "price": costs.get(p["element"]),
            "slot": p["position"],
            "is_captain": p.get("is_captain", False),
            "is_vice_captain": p.get("is_vice_captain", False),
        }
        for p in picks.get("picks", [])
    ]
    return _json(
        {
            "entry": entry,
            "manager": f"{profile.get('player_first_name')} {profile.get('player_last_name')}",
            "overall_rank": profile.get("summary_overall_rank"),
            "total_points": profile.get("summary_overall_points"),
            "bank": (profile.get("last_deadline_bank") or 0) / 10.0,
            "squad_value": (profile.get("last_deadline_value") or 0) / 10.0,
            "squad": squad,
            "entry_history": picks.get("entry_history", {}),
        }
    )


#: Every tool the weekly agent is given.
AGENT_TOOLS = [
    score_players,
    captaincy_candidates,
    explain_player,
    get_fixtures,
    get_model_metrics,
    get_my_team,
]
