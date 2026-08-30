#!/usr/bin/env bash
# Pre-deadline run: refresh data, rescore, and write the brief.
#
# A gameweek is only trainable once FPL sets `data_checked`, which happens after
# the Opta review the morning after the last match. `fpl sync` enforces that, so
# this is safe to run at any hour.
set -euo pipefail
cd "$(dirname "$0")/.."

FPL="$PWD/.venv/bin/fpl"
LOG="$PWD/reports/weekly.log"
mkdir -p reports

{
  echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') weekly run ==="
  "$FPL" sync
  "$FPL" data build
  "$FPL" data features
  "$FPL" score --top 30
  # The brief needs Anthropic credentials; skip rather than fail the whole run.
  if [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -d "$HOME/.config/anthropic" ]; then
    "$FPL" brief
  else
    echo "skipping brief: no Anthropic credentials"
  fi
} 2>&1 | tee -a "$LOG"
