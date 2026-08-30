#!/usr/bin/env bash
# Monthly retrain. Writes a new versioned model and report, and promotes it only
# if it ranks starters at least as well as the incumbent.
set -euo pipefail
cd "$(dirname "$0")/.."

FPL="$PWD/.venv/bin/fpl"
LOG="$PWD/reports/monthly.log"
mkdir -p reports

{
  echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') monthly retrain ==="
  "$FPL" sync --backfill
  "$FPL" data build
  "$FPL" data features
  "$FPL" train
} 2>&1 | tee -a "$LOG"
