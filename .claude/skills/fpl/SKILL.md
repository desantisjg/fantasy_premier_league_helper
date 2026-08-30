---
name: fpl
description: Manage a Fantasy Premier League season from a fitted statistical model — sync data, retrain monthly, score upcoming gameweeks, pick a captain, and produce a pre-deadline brief. Use whenever the user asks about FPL players, transfers, captaincy, chips, their squad, or model performance.
---

# Fantasy Premier League assistant

This project fits a regression model to FPL points and exposes it through a CLI.
Run everything with the project virtualenv: `.venv/bin/fpl <command>`.

## The one thing to get right

**Projections are a ranking, not a forecast.** Out of sample the model reaches
R² ≈ 0.06 on players with real minutes. That is *normal* for single-gameweek FPL
points — most of the variance is irreducible — and it still ranks meaningfully
better than recent form, which is what the decision needs. Never present a
projected total as a prediction of what someone will actually score. Run
`fpl score --output json` and quote the ranking; if you are about to say "he will
score 6", say "he ranks 2nd this week at 6.1 projected" instead.

## Commands

| Command | Use it when |
|---|---|
| `fpl sync` | Before anything else if data may be stale. Fetches the latest snapshot and refreshes per-fixture histories. `--backfill` also downloads archived seasons. |
| `fpl data build` | Rebuild the per-fixture table after a sync. |
| `fpl data features` | Rebuild the design matrix. |
| `fpl data info` | What is on disk: latest finalised gameweek, next deadline, whether the dataset is built. |
| `fpl train` | Monthly. Backtests, refits, writes `models/<YYYY-MM>/report.html` and promotes only if it is not worse than the incumbent. |
| `fpl score` | Rank players for the coming gameweeks. `--position`, `--max-price`, `--top`, `--output json`. |
| `fpl haul` | Rank by probability of a double-digit return. **This is the captaincy tool.** |
| `fpl brief` | Full agent run: reads the model, searches for team news, writes `reports/gwNN_brief.md`. Needs Anthropic credentials. |

## Routine order

Weekly, before a deadline: `fpl sync` → `fpl data build` → `fpl score`.
Monthly: add `fpl train` and read the report.

A gameweek only becomes trainable once FPL sets `data_checked`, which happens after
the Opta review completes the morning after the last match. Before that, bonus
points and defensive contributions are still provisional. `fpl sync` handles this;
do not work around it.

## Captaincy is a different question

Use `fpl haul`, not the top of `fpl score`. Expected points is a conditional mean;
the armband doubles one return, so it is a bet on the upper tail. The two rankings
genuinely differ — residual skew is 2.8 with kurtosis 16, which is why the tail has
its own model.

## Reading the output

- `projected_points` — already adjusted for FPL's availability flag.
- `p_haul_adjusted` — probability of 10+ points, availability-adjusted.
- `chance_of_playing` — FPL's own figure; below 100 means a published doubt.
- `fixtures` — 2 means a double gameweek, and the projection is the sum.

## Interpreting model quality

`models/current/metrics.json` holds the backtest. Compare `performance_starters`
against `best_baseline` — the model's value is the *lift* over naive form, not the
absolute R². If a monthly retrain is not promoted, the report says why; do not
force it.

## What the model cannot know

It has no access to news. Press conferences, fitness tests and rotation hints are
exactly where a human or a web search adds value the regression cannot. That is the
only legitimate reason to depart from its ranking — and say which piece of news it
was.
