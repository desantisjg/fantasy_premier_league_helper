# FPL Analytics Agent — Project Plan

**Goal:** an LLM agent that manages a Fantasy Premier League season from a fitted statistical
model, not from vibes. Linear regression predicts per-gameweek FPL points; the model retrains
monthly; the agent runs weekly before each deadline and produces a written recommendation
grounded in the model's numbers.

**Status at time of writing:** 2026/27 season, GW2 complete, GW3 deadline Fri 4 Sep 17:30 UTC.

**Decisions locked:**
- Training data: 2025/26 onward only (the first season with DEFCON scoring).
- Tool surface: Python CLI + a Claude Code skill.
- Agent runtime: Anthropic API (`claude-opus-5`), weekly loop.
- Deliverable from training: a full ML report with standard inferential + predictive statistics.

---

## 0. Scoring rules (the thing we are predicting)

### Base scoring, 2026/27

| Action | GK | DEF | MID | FWD |
|---|---|---|---|---|
| Playing 1–59 min | 1 | 1 | 1 | 1 |
| Playing 60+ min | 2 | 2 | 2 | 2 |
| Goal | 6 | 6 | 5 | 4 |
| Assist | 3 | 3 | 3 | 3 |
| Clean sheet (60+ min) | 4 | 4 | 1 | — |
| Every 3 saves | 1 | — | — | — |
| Penalty save | 5 | — | — | — |
| Every 2 goals conceded | −1 | −1 | — | — |
| Penalty miss | −2 | −2 | −2 | −2 |
| Own goal | −2 | −2 | −2 | −2 |
| Yellow card | −1 | −1 | −1 | −1 |
| Red card | −3 | −3 | −3 | −3 |
| Bonus (top 3 BPS in match) | 3/2/1 | 3/2/1 | 3/2/1 | 3/2/1 |

### Defensive Contribution (DEFCON)

Introduced 2025/26, unchanged for 2026/27. **A threshold event, not a linear one** — this drives
the model structure.

- **DEF:** 10+ combined Clearances, Blocks, Interceptions, Tackles (CBIT) → **2 pts**
- **MID/FWD:** 12+ CBIT **+ Ball Recoveries** (CBIRT) → **2 pts**
- Capped at 2 pts per match regardless of how far past the threshold.

### 2026/27 changes that affect the pipeline

1. **BPS was reworked** to reduce DEFCON overlap and improve GK / full-back / attacker bonus
   prospects. → *2025/26 bonus and BPS data is not fully comparable to this season.* The bonus
   component of the model is our weakest link until ~GW10; weight it down and re-check monthly.
2. **Scores finalize at 09:00 UK the morning after the last match** (previously 1hr after final
   whistle), to allow Opta review. → *Implemented better than planned:* the API exposes
   `data_checked` per gameweek, which flips true when the review completes. `finalised_gameweeks()`
   requires `finished AND data_checked`, so we never need to reason about the clock.
3. Two full chip sets (Wildcard, Free Hit, Triple Captain, Bench Boost); the first set expires at
   the GW19 deadline (2 Jan). → chip-timing advice needs a first-half/second-half deadline model.
4. Up to 5 free transfers may be banked. No extra December transfers this season (no AFCON).

### Squad constraints (for later optimization)

15 players (2 GK / 5 DEF / 5 MID / 3 FWD), £100.0m budget, max 3 per club, starting XI of 11 with
min 1 GK / 3 DEF / 2 MID / 1 FWD.

**Implementation note:** encode the scoring table once in `src/fplr/scoring.py` as a pure function
`points(stat_line, position) -> int`. Unit-test it by replaying actual 2025/26 gameweeks and
asserting we reproduce FPL's published `total_points` exactly. If we cannot reproduce their number,
we do not understand the rules well enough to model them, and everything downstream is suspect.

---

## 1. Data layer

### Sources

| Source | Endpoint / path | Gives us |
|---|---|---|
| Bootstrap | `fantasy.premierleague.com/api/bootstrap-static/` | Players, teams (incl. `strength_attack_home/away`, `strength_defence_home/away`), positions, gameweek calendar |
| Fixtures | `/api/fixtures/` | `team_h`, `team_a`, difficulty ratings, kickoff times, `finished` |
| Live GW | `/api/event/{gw}/live/` | Per-player, per-GW stat lines — the training target and most features |
| Player detail | `/api/element-summary/{id}/` | Per-fixture history for the current season |
| My team | `/api/entry/{id}/` + `/api/entry/{id}/event/{gw}/picks/` | Current squad, bank, free transfers, chips used |
| 2025/26 backfill | `vaastav/Fantasy-Premier-League` → `data/2025-26/gws/merged_gw.csv` | The prior season |

No API key needed. Rate-limit to ~1 req/sec and cache aggressively.

### Storage

Every sync writes an immutable dated snapshot to `data/raw/YYYY-MM-DD/` (gzipped JSON), then
normalizes into Parquet in `data/processed/`. Training always reads from a named snapshot, so any
model version can be rebuilt byte-for-byte later.

### Verified availability (checked 2026-08-30)

All four FPL endpoints and both archive seasons return 200. The per-gameweek stat line from
`/event/{gw}/live/` carries exactly these fields:

```
minutes  goals_scored  assists  clean_sheets  goals_conceded  own_goals
penalties_saved  penalties_missed  yellow_cards  red_cards  saves  bonus  bps
influence  creativity  threat  ict_index
clearances_blocks_interceptions  recoveries  tackles  defensive_contribution
starts  expected_goals  expected_assists  expected_goal_involvements
expected_goals_conceded  total_points  in_dreamteam  played
```

**DEFCON components are available.** The raw inputs are separate fields
(`clearances_blocks_interceptions`, `tackles`, `recoveries`), so we can model the threshold from its
components rather than only its outcome. `defensive_contribution` is the **count**, not the points,
and FPL pre-computes it position-appropriately — verified against GW2 data:

| Player | Pos | `defensive_contribution` | CBI+T | CBI+T+rec |
|---|---|---|---|---|
| Muharemović | DEF | 19 | **19** | 22 |
| Ajer | DEF | 17 | **17** | 19 |
| Stach | MID | 16 | 12 | **16** |
| Mainoo | MID | 14 | 5 | **14** |

So `defensive_contribution == CBIT` for defenders and `== CBIRT` for midfielders and forwards. The
2025/26 archive carries the same four columns, so the feature is consistent across both seasons.

**The `explain` block is the replay test's oracle.** Each live element carries a per-fixture point
attribution keyed by identifier:

```json
{"identifier": "defensive_contribution", "points": 2, "value": 19, "points_modification": 0}
```

This means `scoring.py` can be validated component-by-component, not just against the total — if our
clean-sheet logic is wrong but our bonus logic compensates, the test still catches it.

### Remaining Phase-1 risk

- **Sample size.** The 2025/26 archive is 29,757 player-gameweek rows. After filtering to players
  with meaningful minutes that is closer to 8–10k usable rows, split four ways by position → roughly
  2–3k rows per position model. **This is the binding constraint on model complexity** and is the
  main reason a regularized linear model is the right call rather than something deeper.

---

## 2. Model

### Why not a single OLS on total points

A single regression on raw points will fit badly, for four reasons that are all fixable:

1. **Zero inflation.** Most rows are 0–2 points (benched, injured, subbed on late). The minutes
   process dominates total variance and is a fundamentally different question from "how well did
   they play".
2. **Position-dependent coefficients.** A goal is worth 6 to a defender and 4 to a forward; a clean
   sheet is worth 4, 1, and 0. One pooled slope cannot represent this.
3. **DEFCON is a step function.** 9 CBIT scores 0; 10 scores 2; 25 still scores 2. Linear in CBIT is
   the wrong functional form.
4. **Points are right-skewed.** OLS predicts conditional means; FPL decisions (especially captaincy)
   are about the right tail.

### Structure: decomposed linear model

Keep every component linear and interpretable, but model the components separately and recombine
them using the known scoring rules.

```
E[points] =  appearance
           + goal_pts + assist_pts
           + CS_value(pos) × P(clean sheet)
           + 2 × P(DEFCON threshold hit)
           + E[bonus]
           + E[saves pts]          (GK)
           − E[card & concession penalties]
```

| Component | Form | Notes |
|---|---|---|
| `M_appear` | Logistic → P(plays), P(60+ \| plays) | Fitted first; feeds every other component as an exposure term |
| `M_attack` | Ridge, per position, target = attacking points **per 90** | Features: rolling xG90, xA90, shots, key passes, npxG |
| `M_cs` | Logistic, team-level → P(clean sheet) | Team defensive strength × opponent attack strength × home/away |
| `M_defcon` | Logistic, per position → P(≥ threshold) | From rolling CBIT90 / CBIRT90 and expected minutes |
| `M_bonus` | Ridge → E[bonus] | Weakest component; 2026/27 BPS rework makes prior-season data partly stale |
| `M_saves` | Ridge, GK only | Rolling saves per 90 × opponent shot volume |

**Also fit a plain pooled OLS on total points as a documented baseline.** The report must show the
decomposed model beating it — if it does not, the added structure is not earning its keep and we
simplify.

### Features (~40)

- **Form / rolling (3, 5, 10 GW windows):** minutes, starts, minutes share, xG90, xA90, npxG90,
  shots90, key passes90, CBIT90, CBIRT90, DEFCON hit rate, BPS90, bonus, saves90
- **Team context:** team xG and xGA over last 5, team goals for/against
- **Opponent:** opponent xGA and xG over last 5, `strength_attack` / `strength_defence` from the API
- **Fixture:** home/away, FDR, days since last match, **number of fixtures this GW**
- **Availability:** `chance_of_playing_next_round`, `status`
- **Static:** price, position dummies, season gameweek index

### Two things that must not be got wrong

- **Double and blank gameweeks.** A player can have 0 or 2 fixtures in a gameweek. Predict
  **per fixture** and sum to the gameweek. Getting this wrong silently halves DGW predictions,
  which are exactly the weeks where the biggest decisions get made.
- **Target leakage.** The bootstrap fields `form`, `points_per_game`, and `total_points` are
  computed *including* the gameweek we are predicting whenever we sync after the fact. Never use
  them as features. Every rolling feature must be reconstructed strictly from gameweeks ≤ n.

### Validation

**Expanding-window walk-forward, by gameweek.** Train on GW1..n, predict GW n+1, roll forward.
Six gameweeks of burn-in for the rolling features. A random train/test split leaks the future and
will make the model look far better than it is — never use one here.

---

## 3. The ML report

Generated on every training run to `models/{tag}/report.html`, with a machine-readable
`metrics.json` beside it for the agent to read.

### Inferential statistics (statsmodels OLS, per position)

- Full coefficient table: estimate, std. error, t-statistic, P>|t|, 95% CI
- R², **Adjusted R²**, F-statistic and its p-value, log-likelihood, AIC, BIC
- **Cluster-robust standard errors, clustered by player.** Residuals are correlated within a
  player across the season; naive OLS standard errors will be too small and will make features look
  significant when they are not. This is the single most important correction in the report.
- **VIF per feature** + condition number. Rolling windows of the same statistic are highly
  collinear; expect this to flag, and use it to prune the feature set.
- **Breusch–Pagan** test for heteroskedasticity (expect it to reject — points variance grows with
  expected points; report HC3 errors alongside)
- **Durbin–Watson** for residual autocorrelation
- **Omnibus / Jarque–Bera** normality, plus skew and kurtosis of residuals

> Practical consequence: statsmodels drives the *inferential* report; scikit-learn pipelines drive
> *production scoring*. Both are fitted on the same design matrix so the report describes the model
> actually being used.

### Predictive statistics (walk-forward, out of sample)

- MAE, RMSE, out-of-sample R², overall and per position
- **Baselines to beat** — a model is only as good as what it improves on:
  predict the mean · predict last gameweek's points · predict season points-per-game ·
  FPL's own `form` field
- **Rank quality:** Spearman and Kendall correlation within position, plus Precision@10 and
  Precision@20 (of the 10 players we ranked highest, how many actually returned?). *This matters
  more than MAE* — every real decision is a ranking, not a point estimate.
- **Calibration:** predicted-vs-actual by predicted decile
- **Residual diagnostics:** residuals vs fitted, QQ plot, residuals vs minutes played
- **Learning curve** by training-set size — tells us whether waiting for more data will help
- **Coefficient stability across monthly refits.** Track each coefficient over time; sign flips or
  large swings between months are a red flag that the model is fitting noise.
- Standardized coefficients + permutation importance for feature ranking

### Expectations, stated honestly up front

Next-gameweek FPL points are mostly irreducible variance. An out-of-sample R² in the **0.2–0.3**
range is a good result for this problem, not a failure. The value of the model is in *ranking*
players and in being consistent, not in nailing individual scores. The report should always print
the baseline comparison next to the headline number so this stays in view.

---

## 4. Tools (CLI)

Typer CLI in `src/fplr/cli.py`. Every command is a thin wrapper over a plain Python function, so the
agent tools and the CLI share one implementation.

| Command | Does |
|---|---|
| `fpl sync [--backfill]` | Pull the API, detect the newest *finalized* gameweek (respecting the 09:00 rule), write a dated snapshot |
| `fpl train [--as-of-gw N] [--tag YYYY-MM]` | Rebuild features, walk-forward fit, write versioned artifacts + `metrics.json` |
| `fpl report [--tag ...] [--open]` | Render the full ML report |
| `fpl score [--gw N] [--horizon 5]` | Score every player for the next N gameweeks → ranked table (JSON + markdown) |
| `fpl team [--entry-id ...]` | Fetch my squad, bank, free transfers, chips remaining |

**Model promotion gate:** a new monthly model is written to `models/{tag}/` but only symlinked to
`models/current` if its walk-forward MAE is at least as good as the incumbent's on the shared
holdout. Otherwise it is written and flagged. This stops a bad retrain from silently degrading
advice, which is the most likely way this project quietly fails.

---

## 5. Claude Code skill

`.claude/skills/fpl/SKILL.md` — teaches the interactive assistant when to sync, train, score, and
report, how to read `metrics.json`, and how to interpret the ranked output (including the caveat
that the model predicts means and under-predicts hauls).

---

## 6. Weekly agent (Anthropic API)

A Python program in `src/fplr/agent/weekly.py`.

- **Model:** `claude-opus-5` with `thinking: {"type": "adaptive"}` and
  `output_config: {"effort": "high"}`.
- **Loop:** `client.beta.messages.tool_runner` with `@beta_tool`-decorated functions — the SDK
  drives the tool-call loop, we just write the tools. Same functions the CLI calls.
- **Tools exposed:** `sync_data`, `get_model_metrics`, `score_players`, `get_my_team`,
  `get_fixtures`, `simulate_transfer` (expected point delta over the next N gameweeks for a
  proposed swap, subject to budget / 3-per-club / formation).
- **Plus the server-side `web_search_20260209` tool.** The statistical model cannot know about a
  Friday press conference, a late injury, or a rotation hint. This is where the LLM adds something
  the regression genuinely cannot.
- **Prompt caching:** the rules reference, tool definitions, and system prompt are identical every
  week — put a `cache_control: {"type": "ephemeral"}` breakpoint after them and keep the volatile
  gameweek context after it.
- **Output:** `reports/gw{N}_brief.md` — captain and vice, transfer recommendations with expected
  point deltas, chip advice, bench order. **Every recommendation must cite the model's number and
  the model's error bar**; the agent's job is to explain and contextualize the model, not to
  overrule it on a hunch.
- **Trigger:** cron the day before each deadline (or the `/loop` skill).
- **Cost:** roughly 50k input / 5k output tokens per run at $5/$25 per MTok ≈ **$0.38 per week**,
  before caching. Not a consideration.

### One modeling addition specifically for the agent

Captaincy is a **tail** decision, not a mean decision — you captain the player most likely to haul,
which is not always the highest expected score. Fit a separate logistic model for
**P(points ≥ 10)** and expose it as its own tool output. The agent should use expected points for
transfers and haul probability for the armband.

---

## 7. Monthly retrain

Cron on the 1st: `fpl sync && fpl train && fpl report`. The agent then reads the new and old
`metrics.json`, reports whether the model improved, and flags any coefficient that changed sign
since last month.


### M2 outcome

**102 features across 30,357 player-fixtures**, built as `fplr.features`. Framing: each row's
features describe what was knowable *before* that fixture, and the target is that same fixture's
points. This makes the leakage rule a single sentence — nothing in a row may come from its own
fixture — and makes it directly testable.

**The leakage guarantee is proved by experiment, not by inspection.** `test_features_ignore_the_
fixture_they_predict` corrupts one fixture's stats to 999 and asserts that fixture's own 102
features do not move. If any window forgot to shift, the test fails loudly.

**FPL's static team-strength ratings turned out to be unusable, twice over.** The current bootstrap
reports every `strength_attack_*` and `strength_defence_*` as **0**, and the archive's `teams.csv`
is an end-of-season snapshot, so using it for mid-season rows leaks how the season finished. Team
strength is therefore *derived*: lagged rolling goals for/against and xG for/against per club, with
the opponent's identical block joined through the opposing club. This is better-founded than FPL's
subjective ratings anyway.

**A club-attribution bug was caught and fixed.** The first normaliser took each player's club from
`players_raw.csv` — which records their club at *season end* — so anyone transferred in January had
that club stamped onto their August fixtures. It surfaced as fixtures with four clubs instead of
two. Club is now derived from the fixture itself (`team_h` if `was_home`, else `team_a`), which is
exact and transfer-proof. Note that the scoring replay could never have caught this, since club does
not affect points; three new integrity tests cover it instead.

**Design notes.** Rates are summed-stat over summed-minutes rather than the mean of per-match rates,
so a ten-minute cameo cannot dominate a window. Defensive contribution is featurised as a *hit rate*
against the position threshold, not a raw count, matching the flat-2-points-at-the-cutoff rule. Per-90
features are left missing rather than zero-imputed when a player has no minutes in the window (45%
of rows for the 3-fixture window) — M3 must impute explicitly and add a missingness indicator, since
zero would read as "played a lot, did nothing".

### M3 outcome

**The pooled ridge won, and the decomposed model did not earn its keep.** The plan committed in
advance to dropping the extra structure if it failed to beat the simple model, and it did fail.
On starters, walk-forward over 31 gameweek folds:

| model | MAE | RMSE | R² | Spearman ρ (weekly) | P@10 | P@20 |
|---|---|---|---|---|---|---|
| **pooled ridge** | **2.295** | **3.041** | **0.064** | **0.299** | **0.120** | **0.190** |
| decomposed | 2.356 | 3.140 | 0.002 | 0.246 | 0.100 | 0.160 |
| form (last 3) | 2.591 | 3.460 | −0.212 | 0.205 | 0.097 | 0.137 |
| form (last 10) | 2.425 | 3.201 | −0.038 | 0.185 | 0.113 | 0.153 |
| predict the mean | 2.298 | 3.599 | −0.311 | — | 0.050 | 0.103 |

The model beats the best naive baseline by **+0.095 Spearman** and is the only predictor with a
positive out-of-sample R². Blending the two models was tested and is monotonically worse than pooled
alone. The decomposed model is retained anyway — not for accuracy but because its component
breakdown (appearance / attack / clean sheet / DEFCON / bonus) is what lets the agent *explain* a
recommendation. It is fitted and stored, just not the one that ranks.

**Two populations, two different stories.** Across all players, naive form out-ranks the model
(ρ 0.73 vs 0.69) — because that population is dominated by the easy question of who plays at all.
Restricted to players with real recent minutes, which is where transfers actually come from, the
model wins decisively and every baseline goes negative on R². Reporting only the first number would
have been flattering and useless.

**Diagnostics confirm the modelling choices, and one refutes a plan assumption.**
Durbin–Watson 2.005 — no residual autocorrelation. Breusch–Pagan p ≈ 0 — strong heteroskedasticity,
as expected, which is why robust clustered errors are mandatory. Jarque–Bera p ≈ 0 with **skew 2.78
and kurtosis 16.2**: residuals have a very heavy right tail. This is the quantitative form of the
captaincy argument — a mean-predicting model systematically under-serves the haul decision, so the
separate P(points ≥ 10) model is not optional.

**A bug that would have invalidated the entire inferential half.** The first report gave an F
statistic of 0.049 (p = 0.999) alongside R² = 0.32, and an *infinite* condition number. The imputer
emits one missingness indicator per feature, and many are exact duplicates — every per-90 rate over
the same window is missing on identical rows — leaving the design matrix singular. Ridge tolerates
that; OLS does not, and every standard error and the F-test were meaningless. The inferential fit
now reduces to a full-rank basis first (89 aliased columns dropped, reported in the run), after
which F = 111.4, p < 0.001, condition number 9,135. R² is unchanged at 0.318, as it must be.

**Read the coefficients with the VIF table open.** `minutes_mean_l3` carries +1.75 points per
standard deviation while `starts_mean_l3` carries −0.80 — they are near-duplicates with a max VIF of
1,509, so the pair is splitting one effect between them and the individual signs are not
interpretable. This is exactly why production scoring is ridge-regularised and why the report ships
VIF beside the coefficients. `price` (+0.26, t = 7.5) and `is_home` (+0.076, t = 6.9) are clean.

**Expectations, revised down honestly.** The plan guessed out-of-sample R² of 0.2–0.3. That holds
across all players (0.32 in-sample, ~0.31 walk-forward) but is **0.064 on starters**, which is the
number that matters. Next-gameweek points are overwhelmingly irreducible variance. The model earns
its place by ranking better than form, not by predicting scores.

### M4–M7 outcome

**Scoring works by appending unplayed fixtures to the historical table** and running the ordinary
feature pipeline over the whole thing. Every window is backward-looking, so a future row picks up
the form a player carries into it — and crucially there is no separate prediction-time feature path
that could drift out of sync with the training one.

**Two bugs found in projection, both caught by checking a documented claim rather than by a test
failing.** First, form was *not* flat across the horizon as the design asserted: a rolling window
counts rows, and an unplayed fixture is a row, so a projection four gameweeks out spent most of its
window on fixtures that had not happened. Haaland decayed 6.60 → 3.89 across GW3–GW7, an artefact
indistinguishable from a hard run of fixtures. Form is now frozen at the first unplayed fixture, the
only one whose window is entirely real history. Second, the initial fix over-corrected and froze the
*opponent's* form too — but the opponent changes every gameweek, which is the entire point of a
horizon. Own-club form is now carried by the player's club and opponent form by the opponent.
Three regression tests cover both directions.

**The haul model earns its place empirically.** `test_captaincy_ranks_by_haul_probability_not_points`
asserts the two rankings differ; they do. Cherki outranks Gibbs-White on haul probability despite a
lower projection, and Palmer (4.90 projected) falls out of the captaincy top eight entirely at
P(haul) = 0.067.

**Agent design.** `claude-opus-5` with adaptive thinking and `effort: high`, driven by the SDK's
`tool_runner` over six tools that wrap the same functions the CLI calls — so the agent cannot report
something `fpl score` would not. Plus the server-side `web_search` tool, which is the agent's only
genuine edge over the regression: press conferences, fitness tests, rotation hints. The system
prompt forbids overruling the model's ranking on taste and requires naming the specific piece of
news behind any departure.

Two implementation details that would otherwise bite:
- **Prompt caching.** The system prompt, scoring rules and tool definitions are byte-identical every
  week; a cache breakpoint after that stable prefix means the weekly run re-reads almost none of it.
- **`pause_turn` must be handled explicitly.** Web search is server-side, and a long search turn can
  stop with `stop_reason: "pause_turn"`. The Python tool runner only continues after a *client* tool
  returns, so a paused turn silently ends the loop and returns a truncated answer with no error. The
  loop mirrors the conversation and restarts on a pause, capped at five restarts.

**Verified live.** The first real run completed in 5 turns and 23 tool calls, producing a coherent
GW3 brief. Cost **≈ $0.63** — above the $0.30–0.40 originally estimated, because the agent called
`explain_player` fifteen times to justify individual recommendations. Prompt caching worked as
designed: **161,439 cached tokens against 76,034 fresh input**, so the stable prefix is being reused
rather than re-read. No `pause_turn` occurred, so that path remains exercised only by construction.

**The live run found a tool bug that the offline tests could not.** `explain_player("White")`
returned *Gibbs-White*: substring matching resolved a shorter name to a longer one and reported the
wrong player's numbers under the right player's name. This is the most dangerous shape of tool bug,
because the output is entirely plausible — every field is internally consistent and nothing errors.
The agent caught it and said so in the brief rather than reporting the numbers as fact, which is the
behaviour the system prompt asks for, but the tool was still wrong. Now: an exact name wins
outright, and genuine ambiguity returns the candidate list instead of guessing. Three regression
tests cover it.

Also fixed: the run's progress log matched only `tool_use` blocks, so the server-side web search —
which arrives as `server_tool_use` — was invisible. Searching *had* happened (the brief cited an ACL
rupture and a three-match ban), but the log implied otherwise.

**Scheduling.** `scripts/weekly.sh` (sync → build → features → score → brief) and
`scripts/monthly.sh` (adds `train`). The weekly script skips the brief rather than failing the run
when credentials are absent. Cron lines are in the README.
---

## Sequencing

| Milestone | Contents | Rough effort |
|---|---|---|
| **M1 — Data** | ✅ **Done.** Repo scaffold, API clients, snapshot store, per-fixture history cache, dataset normaliser, `scoring.py` + replay tests | — |
| **M2 — Features** | ✅ **Done.** 102 features, leakage guards, DGW handling, walk-forward splitter | — |
| **M3 — Model + report** | ✅ **Done.** Pooled ridge + decomposed model, walk-forward backtest, full ML report, promotion gate | — |
| **M4 — CLI** | ✅ **Done.** `sync` / `data` / `train` / `score` / `haul` / `brief`, promotion gate |  — |
| **M5 — Skill** | ✅ **Done.** `.claude/skills/fpl/SKILL.md` | — |
| **M6 — Agent** | ✅ **Done and verified live.** Tool runner, web search, caching, `pause_turn` handling | — |
| **M7 — Optimizer** *(deferred)* | ILP squad/transfer optimizer under FPL constraints | later |

### M1 outcome

**The replay gate passed.** `fplr.scoring` reproduces FPL's published points for **all 29,747
player-fixtures of 2025/26 with zero mismatches**, and reproduces the live API's per-component
`explain` attribution for every fixture of the synced current-season gameweeks. Rare events are
covered — 11 penalty saves, 44 red cards, and forwards reaching the DEFCON threshold 9 times — so
the result is not an artefact of those branches never executing.

Four things the build corrected or discovered:

1. **`data_checked` replaced the clock rule** for gameweek finalisation (above).
2. **Scoring must be per fixture, never per gameweek.** `goals_conceded // 2` and `saves // 3` both
   round down, so aggregating a double gameweek gives the wrong answer — conceding 1+1 costs
   nothing in reality but −1 on the aggregate. The canonical table's grain is therefore one row per
   player per fixture. 2025/26 contained 419 double-gameweek player-weeks.
3. **`element-summary/{id}/history` and the archive's `merged_gw.csv` share a field schema**, so the
   two seasons stack with renaming rather than reconciliation. The live `/event/{gw}/live/`
   endpoint is gameweek-aggregated and so is *not* suitable as the model's row source; it is kept
   for the scoring oracle only.
4. **Cross-season identity needs `code`, not `id`.** FPL reassigns `element` ids each season; every
   row is keyed on the stable `player_code`, with `team_code` likewise stable across
   promotion/relegation.

Known data quirks handled: the archive's gameweek file lives under `gws/`, and 2025/26 ships 10
byte-identical duplicate rows which are dropped on the identity key (keeping them would
double-count those players in every rolling feature).

---

## Open questions

- **FPL entry ID** for `fpl team` — needed before M6.
- **Optimizer scope.** M7 (a proper integer program over the 15-man squad) was one of the options
  and is currently deferred. Worth revisiting once the model is validated, since a good model with
  greedy transfer selection leaves points on the table.
- ~~**Archive availability.**~~ *Resolved 2026-08-30:* the 2025/26 archive has all four DEFCON
  columns (`clearances_blocks_interceptions`, `tackles`, `recoveries`, `defensive_contribution`)
  across 29,757 rows, matching the live API schema. No fallback needed.
