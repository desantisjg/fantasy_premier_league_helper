"""Agent tool contracts.

These run offline against the built dataset and promoted model. They check the
shape and honesty of what the agent is handed -- not the model's quality, which is
`test_model.py`'s job.
"""

from __future__ import annotations

import json

import pytest

from fplr.agent import tools


@pytest.fixture(scope="module", autouse=True)
def require_model():
    try:
        tools._scored(1)
    except (FileNotFoundError, RuntimeError) as error:
        pytest.skip(f"model or data unavailable: {error}")


def _parse(payload: str) -> object:
    parsed = json.loads(payload)
    assert not (isinstance(parsed, dict) and "error" in parsed), parsed
    return parsed


def test_every_tool_has_a_description_and_schema():
    """The docstring is the model's only guide to when a tool applies."""
    for tool in tools.AGENT_TOOLS:
        assert tool.name
        assert tool.description and len(tool.description) > 40, tool.name
        assert tool.input_schema["type"] == "object"


def test_score_players_returns_a_ranked_table():
    rows = _parse(tools.score_players(limit=10))
    assert 0 < len(rows) <= 10
    points = [row["projected_points"] for row in rows]
    assert points == sorted(points, reverse=True), "results must be ranked"
    for row in rows:
        assert {"web_name", "position_name", "team_name", "price"} <= row.keys()


def test_score_players_respects_filters():
    rows = _parse(tools.score_players(position="DEF", max_price=5.0, limit=20))
    assert rows, "expected at least one cheap defender"
    assert all(row["position_name"] == "DEF" for row in rows)
    assert all(row["price"] <= 5.0 for row in rows)


def test_captaincy_ranks_by_haul_probability_not_points():
    """The armband is a tail question and must not just mirror expected points."""
    haul = _parse(tools.captaincy_candidates(limit=15))
    probabilities = [row["p_haul_adjusted"] for row in haul]
    assert probabilities == sorted(probabilities, reverse=True)

    by_points = _parse(tools.score_players(limit=15))
    assert [r["web_name"] for r in haul] != [r["web_name"] for r in by_points], (
        "haul ranking is identical to the points ranking; the tail model adds nothing"
    )


def test_explain_player_breaks_the_projection_into_components():
    top = _parse(tools.score_players(limit=1))[0]
    detail = _parse(tools.explain_player(top["web_name"]))

    components = detail["projection_components_next_gameweek"]
    assert {"appearance", "attack", "clean_sheet", "defensive_contribution"} <= components.keys()
    assert detail["recent_form"], "a projection must be explainable by form"


def test_explain_player_reports_unknown_names_rather_than_guessing():
    result = json.loads(tools.explain_player("Nonexistent Player XYZ"))
    assert "error" in result


def test_model_metrics_expose_the_baseline_comparison():
    """The agent must be able to see what the model is worth, not just its score."""
    metrics = _parse(tools.get_model_metrics())
    assert "starters" in metrics
    assert "best_naive_baseline" in metrics
    assert metrics["starters"]["spearman_weekly"] > (
        metrics["best_naive_baseline"]["starters"]["spearman_weekly"]
    )
    assert "irreducible" in metrics["interpretation"]


def test_get_my_team_fails_clearly_without_an_entry_id(monkeypatch):
    monkeypatch.setattr(tools, "FPL_ENTRY_ID", None)
    result = json.loads(tools.get_my_team(entry_id=0))
    assert "error" in result and "entry id" in result["error"]


def test_exact_name_wins_over_a_longer_substring_match():
    """"White" must not resolve to "Gibbs-White".

    Found by the agent on its first live run: substring matching alone returned the
    wrong player's numbers under the right player's name, which is the most
    dangerous kind of tool bug because the output looks entirely plausible.
    """
    frame, _ = tools._scored(1)
    names = set(frame["web_name"])
    if not {"White", "Gibbs-White"} <= names:
        pytest.skip("this season's squad has no White/Gibbs-White pair")

    assert _parse(tools.explain_player("White"))["player"] == "White"
    assert _parse(tools.explain_player("Gibbs-White"))["player"] == "Gibbs-White"


def test_ambiguous_names_are_reported_rather_than_guessed():
    """A prefix matching several players must ask, not pick one silently."""
    frame, _ = tools._scored(1)
    counts = frame.drop_duplicates("player_code")["web_name"].str[:3].value_counts()
    shared = counts[counts > 1]
    if shared.empty:
        pytest.skip("no ambiguous name prefix in this squad")

    result = json.loads(tools.explain_player(shared.index[0]))
    assert "ambiguous" in result
    assert len(result["candidates"]) > 1
    assert "exact web_name" in result["hint"]


def test_unique_partial_names_still_resolve():
    frame, _ = tools._scored(1)
    if "Haaland" not in set(frame["web_name"]):
        pytest.skip("Haaland not in this squad")
    assert _parse(tools.explain_player("Haal"))["player"] == "Haaland"
