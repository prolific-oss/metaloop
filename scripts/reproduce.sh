#!/usr/bin/env bash
# Reproduce the paper's numbers, tables and figures from the released per-item results.
#
#   scripts/reproduce.sh           no API calls: fetch the question bank if needed, recompute every
#                                  number, table and figure, and check them against the shipped values
#   scripts/reproduce.sh --quick   also run a 3-job pilot of six tasks (all but The-Loop) for one model
#                                  (API calls; set MODEL, default gpt-4.1). Pilot results go to
#                                  tasks/*/results_v2_pilot/, not to the released results.
#
# Needs .venv with requirements.txt installed (or PYTHON=...), and Hugging Face access to GPQA
# Diamond for the question bank (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-gpt-4.1}

if [ ! -f data/question_bank_v2.json ]; then
  echo "== fetching the question bank from the source datasets"
  "$PY" scripts/fetch_qbank.py
fi
# fetch_qbank.py skips a source it cannot load (e.g. GPQA Diamond without access): stop here if so
"$PY" - <<'PYEOF'
import json, sys
n = len(json.load(open("data/question_bank_v2.json")))
m = len(json.load(open("data/question_bank_v2_removed.json"))["items"])
if (n, m) != (454, 2):
    sys.exit(f"question bank incomplete ({n} of 454 items, {m} of 2 removed items): check GPQA Diamond access "
             "(README, 'The question bank'), delete data/question_bank_v2*.json and rerun")
print("question bank: 454 items + 2 removed items")
PYEOF

echo "== recomputing the paper's numbers and tables"
ref=$(mktemp -d)
cp results_v2_metrics/camera_ready_numbers.json "$ref/"
cp -R paper/tables "$ref/tables"
# the recompute overwrites the shipped files; put them back on failure so a rerun compares against them
restore() { cp "$ref/camera_ready_numbers.json" results_v2_metrics/; rm -rf paper/tables; cp -R "$ref/tables" paper/tables; }
trap 'restore; echo "shipped numbers and tables restored"' ERR
"$PY" paper/scripts/camera_ready_numbers.py > "$ref/numbers.log"
"$PY" - "$ref/camera_ready_numbers.json" results_v2_metrics/camera_ready_numbers.json <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
def diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            yield from diff(x.get(k), y.get(k), f"{path}.{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (p, q) in enumerate(zip(x, y)):
            yield from diff(p, q, f"{path}[{i}]")
    elif x != y:
        yield f"{path}: shipped {x!r}, recomputed {y!r}"
d = list(diff(a, b))
print("numbers: identical to the shipped values" if not d else f"numbers: {len(d)} differences")
for line in d[:20]:
    print("  ", line)
sys.exit(1 if d else 0)
EOF
diff -r "$ref/tables" paper/tables > /dev/null && echo "tables: identical to the shipped tables" \
  || { echo "tables: differ"; diff -r "$ref/tables" paper/tables | head -20; false; }
trap - ERR

echo "== figures"
"$PY" paper/scripts/figures_v2.py

echo "== Self-Review Control 1 (difficulty-adjusted confidence gap)"
"$PY" scripts/control1_v2.py | tail -2

echo "== human baselines"
if [ -d results/human_baselines ]; then
  "$PY" scripts/analyse_human.py > "$ref/human.log" && echo "human baselines: recomputed ($ref/human.log)"
else
  echo "human baselines: participant files not in this release; human numbers are the shipped summary"
fi

if [ "${1:-}" = "--quick" ]; then
  echo "== live pilot for $MODEL (API calls)"
  for t in t2_confidence_wagering t5_selective_deferral t7_consistency t6_error_detection \
           t3_source_monitoring t4_strategy_switching; do
    "$PY" "tasks/$t/run_v2.py" "$MODEL" --pilot --limit 3 --workers 3
  done
fi
echo "done"
