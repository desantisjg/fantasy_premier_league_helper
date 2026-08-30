"""Replay real gameweeks through `fplr.scoring` and assert we reproduce FPL exactly.

This is the gate for the rest of the project. If we cannot turn a raw stat line into
FPL's own point attribution, we do not understand the scoring rules well enough to
model them, and every downstream number is suspect.

The replay checks *per component*, not just the total, so a wrong clean-sheet rule
cannot hide behind a compensating error somewhere else.

Scoring is applied per fixture rather than per gameweek because two of the rules
round down: a player who concedes one goal in each leg of a double gameweek loses
nothing, while the gameweek aggregate of two conceded would wrongly cost a point.
The same trap applies to saves.
"""

from __future__ import annotations

import pytest

from fplr.scoring import (
    DEFENSIVE_CONTRIBUTION_THRESHOLD,
    Position,
    StatLine,
    appearance_points,
    score,
    score_breakdown,
)

# --- Rule arithmetic, no network -------------------------------------------


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(0, 0), (1, 1), (59, 1), (60, 2), (90, 2)],
)
def test_appearance_points_boundary(minutes, expected):
    assert appearance_points(minutes) == expected


@pytest.mark.parametrize(
    ("conceded", "expected"), [(0, 0), (1, 0), (2, -1), (3, -1), (4, -2), (5, -2)]
)
def test_goals_conceded_rounds_down(conceded, expected):
    line = StatLine(minutes=90, goals_conceded=conceded)
    assert score_breakdown(line, Position.DEF).get("goals_conceded", 0) == expected


@pytest.mark.parametrize(("saves", "expected"), [(0, 0), (2, 0), (3, 1), (5, 1), (6, 2)])
def test_saves_round_down_and_are_keeper_only(saves, expected):
    line = StatLine(minutes=90, saves=saves)
    assert score_breakdown(line, Position.GK).get("saves", 0) == expected
    assert "saves" not in score_breakdown(line, Position.DEF)


def test_clean_sheet_requires_full_appearance():
    assert "clean_sheets" not in score_breakdown(
        StatLine(minutes=59, clean_sheets=1), Position.DEF
    )
    assert score_breakdown(StatLine(minutes=60, clean_sheets=1), Position.DEF)[
        "clean_sheets"
    ] == 4


def test_defensive_contribution_is_a_capped_threshold():
    below = StatLine(minutes=90, defensive_contribution=9)
    at = StatLine(minutes=90, defensive_contribution=10)
    far_above = StatLine(minutes=90, defensive_contribution=30)
    assert "defensive_contribution" not in score_breakdown(below, Position.DEF)
    assert score_breakdown(at, Position.DEF)["defensive_contribution"] == 2
    assert score_breakdown(far_above, Position.DEF)["defensive_contribution"] == 2


# --- Replay against real data ----------------------------------------------


def _fixture_lines(live, positions):
    """Yield (player_id, position, per-fixture StatLine, expected point map).

    The `explain` block reports, per fixture, each scoring identifier with the stat
    value that produced it and the points awarded. Identifiers that scored nothing
    are omitted -- which is lossless for our purposes, since reconstructing them as
    zero yields zero points either way.
    """
    for element in live["elements"]:
        player_id = element["id"]
        position = Position(positions[player_id])
        for fixture in element["explain"]:
            values, expected = {}, {}
            modification = 0
            for stat in fixture["stats"]:
                identifier = stat["identifier"]
                if identifier in StatLine.__dataclass_fields__:
                    values[identifier] = stat["value"]
                if stat["points"]:
                    expected[identifier] = stat["points"]
                modification += stat.get("points_modification") or 0
            yield player_id, position, StatLine(**values), expected, modification


def test_replay_reproduces_fpl_attribution(live_gameweeks, positions):
    """Every scored component of every fixture must match FPL's own attribution."""
    mismatches = []
    fixtures_checked = 0

    for gameweek, live in sorted(live_gameweeks.items()):
        for player_id, position, line, expected, _ in _fixture_lines(live, positions):
            fixtures_checked += 1
            actual = score_breakdown(line, position)
            if actual != expected:
                mismatches.append(
                    f"GW{gameweek} player {player_id} ({position.name}): "
                    f"ours={actual} fpl={expected} from {line}"
                )

    assert fixtures_checked > 0, "replay found no fixtures to check"
    assert not mismatches, (
        f"{len(mismatches)} of {fixtures_checked} fixtures mis-scored:\n  "
        + "\n  ".join(mismatches[:15])
    )


def test_replay_reproduces_gameweek_totals(live_gameweeks, positions):
    """Summing our per-fixture scores must reproduce each player's gameweek total."""
    mismatches = []
    for gameweek, live in sorted(live_gameweeks.items()):
        totals: dict[int, int] = {}
        modifications: dict[int, int] = {}
        for player_id, position, line, _, modification in _fixture_lines(live, positions):
            totals[player_id] = totals.get(player_id, 0) + score(line, position)
            modifications[player_id] = modifications.get(player_id, 0) + modification

        for element in live["elements"]:
            player_id = element["id"]
            ours = totals.get(player_id, 0) + modifications.get(player_id, 0)
            reported = element["stats"]["total_points"]
            if ours != reported:
                mismatches.append(
                    f"GW{gameweek} player {player_id}: ours={ours} fpl={reported}"
                )

    assert not mismatches, (
        f"{len(mismatches)} players mis-totalled:\n  " + "\n  ".join(mismatches[:15])
    )


def test_defensive_contribution_thresholds_are_empirically_correct(
    live_gameweeks, positions
):
    """Locate each position's DEFCON cutoff in the data and check our table.

    Only single-fixture players are used, so the gameweek-level
    `defensive_contribution` stat is unambiguously that one fixture's count.
    """
    awarded: dict[Position, list[int]] = {}
    withheld: dict[Position, list[int]] = {}

    for live in live_gameweeks.values():
        for element in live["elements"]:
            if len(element["explain"]) != 1:
                continue
            position = Position(positions[element["id"]])
            count = element["stats"]["defensive_contribution"]
            earned = any(
                stat["identifier"] == "defensive_contribution" and stat["points"]
                for stat in element["explain"][0]["stats"]
            )
            (awarded if earned else withheld).setdefault(position, []).append(count)

    for position, threshold in DEFENSIVE_CONTRIBUTION_THRESHOLD.items():
        got_points = awarded.get(position, [])
        no_points = withheld.get(position, [])

        if threshold is None:
            assert not got_points, (
                f"{position.name} was awarded defensive contribution points "
                f"(counts {sorted(got_points)}) but our table marks it ineligible"
            )
            continue

        if not got_points:
            pytest.skip(f"no {position.name} reached the DEFCON threshold in this data")

        assert min(got_points) >= threshold, (
            f"{position.name} scored DEFCON at {min(got_points)}, "
            f"below our threshold of {threshold}"
        )
        if no_points:
            assert max(no_points) < threshold, (
                f"{position.name} was denied DEFCON at {max(no_points)}, "
                f"at or above our threshold of {threshold}"
            )
