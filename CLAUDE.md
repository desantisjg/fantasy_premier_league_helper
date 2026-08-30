# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

An analytical Fantasy Premier League assistant. A linear model is fitted to FPL
points, exposed as CLI commands, and driven by an LLM agent that queries those
commands — including the model's own accuracy statistics — before recommending
captaincy, transfers and chips. There is also a local dark-mode chat UI.

The governing principle: **no recommendation is asserted**. Every number the agent
reports is computed by the model and traceable to it.

## Environment

Use the project virtualenv for everything. There is no `uv` here.

```bash
.venv/bin/python -m pytest -q      # 74 passed, 1 skipped
.venv/bin/fpl <command>
.venv/bin/python -m pip install -e ".[dev]"
```

`.env` holds `ANTHROPIC_API_KEY` and `FPL_ENTRY_ID`. It is gitignored and loaded by
`fplr/config.py` on import — a real environment variable always wins over the file.
**Never commit it, and never print its contents.**

## Command surface

| Command | Notes |
|---|---|
| `fpl sync` | `--backfill` adds archived seasons. Cold run is ~12 min (~700 player histories). |
| `fpl data build` / `features` / `info` | Rebuild after every sync. |
| `fpl train` | Backtest + refit + report + promotion gate. Minutes. |
| `fpl score` / `fpl haul` | Ranking, and the captaincy tail model. |
| `fpl brief` | Live agent run. **Costs ~$0.63.** Ask before running. |
| `fpl serve` | Local chat UI on 127.0.0.1:8000. Chat turns ~$0.02. |

Anything touching the Anthropic API spends the user's money. Confirm first.

## Architecture

```
scoring.py    2026/27 rules — the oracle everything is validated against
ingest.py     API client, dated snapshots, per-fixture history cache
datasets.py   normalise live API + community archive into one table
features.py   design matrix, leakage guards, walk-forward splitter
model.py      pooled ridge, decomposed components, haul model
evaluate.py   walk-forward backtest and metrics
report.py     inferential + predictive ML report (statsmodels)
train.py      versioning, promotion gate
score.py      score upcoming fixtures
agent/runner.py   the shared agent loop  ← both brief and chat use this
agent/tools.py    six @beta_tool wrappers over the CLI's own functions
agent/weekly.py   the weekly brief
web/server.py     FastAPI app + SSE streaming
```

Two thin adapters over one implementation: `cli.py` for humans, `agent/tools.py`
for the model. **They must never diverge** — the agent should be incapable of
reporting a number `fpl score` would not show.

---

## Invariants — breaking these silently corrupts results

### Data

**`scoring.py` is the project's gate.** It must reproduce FPL's published points for
every player-fixture of a full season, checked *per component* via the `explain`
block, not just per total. `test_replay_reproduces_every_fixture_in_the_season` is
the canary. If it fails, nothing downstream is trustworthy — fix it before anything
else.

**Scoring is per fixture, never per gameweek.** `goals_conceded // 2` and
`saves // 3` round down, so conceding 1+1 across a double gameweek costs nothing in
reality but −1 on the aggregate. This is why the canonical grain is one row per
player per fixture, and why `/event/{gw}/live/` (gameweek-aggregated) is *not* the
model's row source — it is the scoring oracle only. Row source is
`element-summary/{id}/history` plus the archive, which share a field schema.

**Only `data_checked` gameweeks are trainable.** FPL revises bonus and defensive
contribution during the Opta review. Provisional rows are *marked*, not dropped —
useless as training targets, but wanted as features when scoring the next gameweek.
`load_player_fixtures()` excludes them by default; opting in is explicit.

**Cross-season identity uses `player_code` and `team_code`, never `element`/`team`.**
FPL reassigns those every season.

**A player's club comes from the fixture, never from `players_raw.csv`.** That file
records their club at *season end*, so it stamps a January transfer onto August
fixtures. This bug passed the 29,747-row scoring replay for a long time, because
club does not affect points. Fixture integrity has its own tests.

### Features

**Nothing in a row may come from its own fixture.** `test_features_ignore_the_
fixture_they_predict` corrupts one fixture's stats to 999 and asserts its own
features do not move. Run it after touching `features.py`.

**Never use `form`, `points_per_game` or `total_points` from the bootstrap as
features.** They are season-to-date figures recomputed after each gameweek and
already contain the answer. They live in `LEAKY_COLUMNS`.

**Form is frozen across the projection horizon.** A rolling window counts *rows*,
and an unplayed fixture is a row — so a projection four gameweeks out otherwise
spends its window on fixtures that have not happened and decays toward the mean,
looking exactly like a hard run of fixtures. Own-club form freezes against the
player's club; **opponent form freezes against the opponent**, which changes every
gameweek.

**FPL's static team-strength ratings are unusable.** The live bootstrap reports
every `strength_attack_*`/`strength_defence_*` as `0`, and the archive's
`teams.csv` is an end-of-season snapshot that leaks the season's outcome. Team
strength is derived from lagged rolling goals and xG. Do not "fix" this by
reintroducing them.

### Modelling

**Validation is walk-forward by gameweek, never a random split.** Form is strongly
autocorrelated; a random split leaks the future and flatters the model badly.

**Report two populations.** Across all players, naive form out-ranks the model
(ρ 0.73 vs 0.69) because that population is dominated by who plays at all. On
*starters* — real recent minutes — the model wins (0.299 vs 0.205) and is the only
predictor with positive out-of-sample R². Quoting only the first number is
flattering and useless.

**Rank metrics are computed within a gameweek then averaged; error metrics are
pooled.** "Top 10 of the season" is not a decision anyone makes.

**OLS needs a full-rank basis; ridge does not.** The imputer emits a missingness
indicator per feature and many are exact duplicates. Left in, the design matrix is
singular, the condition number goes infinite, and F and every standard error become
meaningless (this shipped once as F = 0.049 alongside R² = 0.32). `report.py`
reduces to a basis first and reports what it dropped.

**Standard errors are clustered by player.** A player contributes up to 38
correlated rows per season.

**Captaincy is a separate model.** Residual skew 2.78, kurtosis 16.2 — a conditional
mean cannot represent that tail, and the armband doubles one return. Use `fpl haul`.

**The decomposed model loses to pooled ridge and that is the documented outcome.**
It is kept only because its component breakdown lets the agent explain a
projection. Do not quietly promote it.

**Promotion is gated.** A retrain reaches `models/current` only if it ranks starters
no worse. If it is not promoted, the report says why — do not force it.

### Agent

**`agent/runner.py` holds the one agent loop.** Both the brief and the chat use it.
It carries `pause_turn` handling: web search is server-side, and the SDK's tool
runner only continues after a *client* tool returns, so a paused turn otherwise ends
the loop silently and returns a truncated answer with no error. Do not duplicate
this loop.

**Server tools appear as `server_tool_use`, not `tool_use`.** Matching only the
latter hides web search entirely from logs and the UI.

**The stable prefix is cached.** System prompt, rules and tool definitions are
byte-identical each run and sit behind a `cache_control` breakpoint. Do not
interpolate volatile values (timestamps, gameweek numbers) into `SYSTEM_PROMPT` —
it silently destroys the cache.

**Tools must report ambiguity, not guess.** `explain_player("White")` once returned
*Gibbs-White* — real model output, correctly computed, for the wrong player. Exact
match wins; multiple matches return a candidate list. This class of bug is the most
dangerous, because the output is internally consistent and looks entirely plausible.

**The agent may not overrule the model on taste.** Its legitimate edge is news the
regression cannot see. Departures must name the specific information.

### Web UI

**It is a server, not a standalone HTML file, so the API key stays server-side.**
Do not "simplify" it into a page that calls Anthropic directly. Binds to localhost.

**The markdown renderer is hand-rolled so the app works offline.** Do not swap in a
CDN library. It escapes HTML; keep it that way.

---

## Testing

```bash
.venv/bin/python -m pytest -q
```

| File | Covers |
|---|---|
| `test_scoring.py` | Replay against FPL's own per-component attribution |
| `test_datasets.py` | Full-season replay, schema, fixture integrity |
| `test_features.py` | Leakage tamper test, horizon freezing, split ordering |
| `test_model.py` | Component arithmetic, calibration, evaluation harness |
| `test_agent.py` | Tool contracts, ambiguity handling, baseline visibility |
| `test_web.py` | Endpoints, brief ordering, credential failure path |

Tests skip rather than fail when data or a model is missing. A skipped suite is not
a passing one — check what skipped before claiming green.

Prefer tests that assert a *property* over tests that pin a number: the tamper test
proves no leakage; a hardcoded coefficient just breaks on retrain.

## Conventions

- Comments explain *why*, especially where the code looks wrong but is not (the
  rounding rules, the freezing, the basis reduction). Do not strip them.
- New numbers in the README and PLAN must be verified against artifacts, not
  transcribed from memory.
- `data/`, `models/`, `reports/` and `.env` are gitignored and regenerable.
- Branch names use hyphens, not spaces.
- Report outcomes honestly, including negative results. The decomposed model losing
  is documented, not hidden — that is the standard here.

## Current state

- `main` — model, tools, agent, tests.
- `locally-hosted-ui` — adds `fpl serve` and the chat UI. Not yet merged.
- Season 2026/27, and the dataset covers 2025/26 onward only (DEFCON began then, so
  earlier seasons have a different target distribution). ~1.3 seasons of data is the
  binding constraint on model complexity and the reason a regularised linear model
  is the right call.
