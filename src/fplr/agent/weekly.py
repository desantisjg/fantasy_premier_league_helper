"""The pre-deadline advisory run.

The agent's job is to turn the model's numbers into a decision and to say why. It
is explicitly *not* allowed to overrule the model on a hunch: its edge over the
regression is news the regression cannot see -- a Friday press conference, a late
fitness test, a rotation hint before a European tie -- not a better opinion about
who is in form.

Two implementation details matter:

**Prompt caching.** The system prompt, the scoring rules and the tool definitions
are byte-identical every week, and only the gameweek context changes. A cache
breakpoint after the stable prefix means the weekly run re-reads almost none of it.

**`pause_turn` must be handled explicitly.** Web search is a server-side tool, and a
long search turn can stop with `stop_reason: "pause_turn"`. The Python tool runner
only continues after a *client* tool produces a result, so a paused turn silently
ends the loop and returns a truncated answer with no error. The loop below mirrors
the conversation and restarts the runner on a pause, which is the documented remedy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPORTS_DIR
from ..ingest import latest_snapshot, next_gameweek
from .runner import (
    MODEL,
    WEB_SEARCH_TOOL,
    AgentError,
    build_client,
    describe_api_error,
    stream_agent,
)
from .tools import AGENT_TOOLS

#: Kept as an alias so callers that predate the shared runner still work.
BriefError = AgentError

SYSTEM_PROMPT = """You are an analytical Fantasy Premier League assistant. You advise on one gameweek \
at a time, grounded in a fitted statistical model, and you show your reasoning in \
terms of that model's numbers.

## How FPL scores (2026/27)

Appearance: 1 point under 60 minutes, 2 at 60 or more.
Goals: 6 for a goalkeeper or defender, 5 for a midfielder, 4 for a forward.
Assists: 3. Clean sheet: 4 for a goalkeeper or defender, 1 for a midfielder, \
and only with a full 60-minute appearance.
Goalkeepers earn 1 point per 3 saves and 5 for a penalty save.
Goalkeepers and defenders lose 1 point per 2 goals conceded.
Cards: -1 yellow, -3 red. Own goal: -2. Penalty miss: -2.
Bonus: 3/2/1 to the top three by the Bonus Points System.
Defensive contribution: a flat 2 points at 10 combined clearances, blocks, \
interceptions and tackles for defenders, or 12 of those plus ball recoveries for \
midfielders and forwards. Goalkeepers are not eligible. It is capped at 2 -- \
exceeding the threshold earns nothing extra.

Squad rules: 15 players (2 GK, 5 DEF, 5 MID, 3 FWD), 100.0m budget, at most 3 \
players from any one club, and a starting XI of 11 with at least 1 GK, 3 DEF, \
2 MID and 1 FWD.

## What the model is, and is not

`score_players` returns projected points from a ridge regression over rolling \
form, fixture context and opponent strength, already adjusted for FPL's published \
availability flags. Call `get_model_metrics` and respect what it says: the model \
ranks meaningfully better than recent form, but explains only a small share of \
single-gameweek variance. Most of that variance is irreducible. Never present a \
projection as a forecast of someone's actual score.

Expected points and captaincy are different questions. Points are a conditional \
mean; the armband doubles one return and is therefore a bet on the upper tail. Use \
`captaincy_candidates` for the captain, not the top of the projections table.

## Your edge, and its limits

The model cannot read the news. Your genuine contribution is information it has no \
access to: press conferences, fitness tests, rotation risk before or after European \
fixtures, managers signalling changes. Search for that, and weigh it.

You may not overrule the model's ranking on taste. If you depart from it, name the \
specific piece of information that justifies the departure. "He looks due" is not a \
reason; "the manager confirmed on Friday that he is rested for the cup" is.

## How to answer

Lead with the recommendation. Every claim cites a number from a tool and, where it \
matters, the uncertainty around it. Use `explain_player` when a projection needs \
justifying — a component breakdown is far more persuasive than a total. Be concise \
and specific. Say plainly when the model is close to indifferent between options, \
rather than manufacturing a distinction."""


@dataclass
class BriefResult:
    gameweek: int
    deadline: str
    text: str
    path: Path | None
    usage: dict


def gameweek_context() -> tuple[int, str]:
    """The gameweek we are about to pick a team for, and its deadline."""
    snapshot = latest_snapshot()
    if snapshot is None:
        raise FileNotFoundError("no snapshot on disk; run `fpl sync` first")
    upcoming = next_gameweek(snapshot.read("bootstrap"))
    if upcoming is None:
        raise RuntimeError("the season has no next gameweek")
    return int(upcoming["id"]), upcoming["deadline_time"]


def brief_task(gameweek: int, deadline: str) -> str:
    """The standard weekly request."""
    return (
        f"Write my pre-deadline brief for gameweek {gameweek} (deadline {deadline}).\n\n"
        "Cover, in this order:\n"
        "1. Captain and vice-captain, argued from haul probability.\n"
        "2. The three or four players most worth transferring in, with their "
        "projected points and what is driving them.\n"
        "3. Anyone in my squad I should be worried about — injury, rotation, "
        "or a projection that has fallen away. If you cannot see my squad, say so "
        "and give general watch-outs instead.\n"
        "4. Any chip worth considering this week, or explicitly none.\n\n"
        "Check the news for late injury and rotation information before you "
        "conclude. Keep it under 600 words."
    )


def run_brief(
    *,
    question: str | None = None,
    write: bool = True,
    reports_dir: Path = REPORTS_DIR,
    verbose: bool = True,
) -> BriefResult:
    """Produce the pre-deadline brief for the upcoming gameweek."""
    client = build_client()
    gameweek, deadline = gameweek_context()
    task = question or brief_task(gameweek, deadline)

    # The stable prefix is cached; only the task text below it changes week to week.
    system = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    messages: list[dict] = [{"role": "user", "content": task}]

    text, usage = "", {}
    for event in stream_agent(
        messages=messages,
        system=system,
        tools=list(AGENT_TOOLS) + [WEB_SEARCH_TOOL],
        client=client,
    ):
        if event["type"] == "tools" and verbose:
            print(f"  → {', '.join(event['names'])}", flush=True)
        elif event["type"] == "paused" and verbose:
            print(f"  (paused mid-turn, resuming — restart {event['restart']})", flush=True)
        elif event["type"] == "text":
            text = event["text"]
        elif event["type"] == "done":
            usage = event["usage"]

    path = None
    if write and text:
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"gw{gameweek:02d}_brief.md"
        header = (
            f"# Gameweek {gameweek} brief\n\n"
            f"*Deadline {deadline} · generated "
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `{MODEL}`*\n\n"
        )
        path.write_text(header + text + "\n", encoding="utf-8")

    return BriefResult(
        gameweek=gameweek, deadline=deadline, text=text, path=path, usage=usage
    )
