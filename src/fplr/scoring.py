"""FPL scoring rules for the 2026/27 season.

This module is the project's ground truth for how a match stat line becomes Fantasy
points. Everything downstream -- features, model targets, the point decomposition the
agent reasons about -- assumes these rules are correct, so `tests/test_scoring.py`
replays real gameweeks through this module and asserts we reproduce FPL's own
per-component attribution exactly.

The rules that are easy to get wrong, and are therefore stated explicitly here:

* Clean sheets require 60+ minutes. FPL's own `clean_sheets` stat already encodes
  that, but we re-apply the minutes gate so the rule survives if the upstream
  field ever changes meaning.
* Goals conceded are penalised per *two* conceded, rounded down, and only for
  goalkeepers and defenders.
* Saves score per *three* saves, rounded down, goalkeepers only.
* Defensive contribution is a threshold, not a rate: it pays a flat 2 points at
  the cutoff and nothing extra beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping


class Position(IntEnum):
    """FPL `element_type`."""

    GK = 1
    DEF = 2
    MID = 3
    FWD = 4


# --- Rule tables -------------------------------------------------------------
# Keyed by position so the asymmetries stay visible rather than buried in `if`s.

GOAL_POINTS: dict[Position, int] = {
    Position.GK: 6,
    Position.DEF: 6,
    Position.MID: 5,
    Position.FWD: 4,
}

CLEAN_SHEET_POINTS: dict[Position, int] = {
    Position.GK: 4,
    Position.DEF: 4,
    Position.MID: 1,
    Position.FWD: 0,
}

ASSIST_POINTS = 3
OWN_GOAL_POINTS = -2
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3

#: Minutes needed for the 2-point appearance award and for a clean sheet to count.
FULL_APPEARANCE_MINUTES = 60

#: Positions penalised for goals conceded, one point per this many goals.
CONCESSION_POSITIONS = frozenset({Position.GK, Position.DEF})
GOALS_CONCEDED_PER_POINT = 2

#: Saves per point, goalkeepers only.
SAVES_PER_POINT = 3

#: Defensive contribution: flat 2 points at the position's threshold, capped there.
#: Defenders are measured on CBIT (clearances, blocks, interceptions, tackles);
#: midfielders and forwards on CBIRT, which adds ball recoveries. The FPL API
#: pre-computes the position-appropriate total in `defensive_contribution`.
DEFENSIVE_CONTRIBUTION_POINTS = 2
DEFENSIVE_CONTRIBUTION_THRESHOLD: dict[Position, int | None] = {
    Position.GK: None,  # not eligible; asserted by the replay test
    Position.DEF: 10,
    Position.MID: 12,
    Position.FWD: 12,
}


@dataclass(frozen=True)
class StatLine:
    """One player's stats for one fixture, as returned by the FPL API.

    Field names match the API's `stats` object exactly so a raw response can be
    passed straight through `from_api`.
    """

    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    defensive_contribution: int = 0

    @classmethod
    def from_api(cls, stats: Mapping[str, object]) -> "StatLine":
        """Build from an FPL `stats` mapping, ignoring fields we do not score."""
        known = {f: stats.get(f, 0) for f in cls.__dataclass_fields__}
        return cls(**{k: int(v or 0) for k, v in known.items()})


def appearance_points(minutes: int) -> int:
    if minutes <= 0:
        return 0
    return 2 if minutes >= FULL_APPEARANCE_MINUTES else 1


def score_breakdown(stats: StatLine, position: Position) -> dict[str, int]:
    """Return points per FPL `explain` identifier.

    Only non-zero components are included, matching how the API reports them, so
    the result can be diffed directly against an `explain` block.
    """
    pos = Position(position)
    out: dict[str, int] = {}

    def add(identifier: str, points: int) -> None:
        if points:
            out[identifier] = points

    add("minutes", appearance_points(stats.minutes))
    add("goals_scored", stats.goals_scored * GOAL_POINTS[pos])
    add("assists", stats.assists * ASSIST_POINTS)

    # Clean sheets are gated on a full appearance.
    if stats.minutes >= FULL_APPEARANCE_MINUTES:
        add("clean_sheets", stats.clean_sheets * CLEAN_SHEET_POINTS[pos])

    if pos in CONCESSION_POSITIONS:
        add("goals_conceded", -(stats.goals_conceded // GOALS_CONCEDED_PER_POINT))

    if pos is Position.GK:
        add("saves", stats.saves // SAVES_PER_POINT)

    add("own_goals", stats.own_goals * OWN_GOAL_POINTS)
    add("penalties_saved", stats.penalties_saved * PENALTY_SAVE_POINTS)
    add("penalties_missed", stats.penalties_missed * PENALTY_MISS_POINTS)
    add("yellow_cards", stats.yellow_cards * YELLOW_CARD_POINTS)
    add("red_cards", stats.red_cards * RED_CARD_POINTS)

    threshold = DEFENSIVE_CONTRIBUTION_THRESHOLD[pos]
    if threshold is not None and stats.defensive_contribution >= threshold:
        add("defensive_contribution", DEFENSIVE_CONTRIBUTION_POINTS)

    add("bonus", stats.bonus)
    return out


def score(stats: StatLine, position: Position) -> int:
    """Total FPL points for one player in one fixture."""
    return sum(score_breakdown(stats, position).values())
