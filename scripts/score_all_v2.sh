#!/bin/zsh
# Score every v2 task for the given models (all scorers are cached and idempotent).
# Usage: scripts/score_all_v2.sh <model> [model ...]
set -eu
cd "${0:A:h}/.."
py=.venv/bin/python
$py scripts/score_v2_qa.py "$@"
$py scripts/metrics_v2_qa.py "$@"
$py tasks/t3_source_monitoring/score_v2.py "$@"
$py tasks/t4_strategy_switching/score_v2.py "$@"
$py scripts/score_v2_review.py "$@"
