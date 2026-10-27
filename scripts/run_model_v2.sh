#!/bin/zsh
# Full v2 pipeline for one model: 6 independent tasks, then Self-Review scoring,
# Control 2, the Loop, the Deferral counterfactual, and one retry pass on API errors.
# Usage: scripts/run_model_v2.sh <model_key>   (logs in run_logs/<model>_<step>.log)
set -u
m=$1
cd "${0:A:h}/.."
py=.venv/bin/python
for t in t6_error_detection t5_selective_deferral t2_confidence_wagering t7_consistency t3_source_monitoring t4_strategy_switching; do
  $py tasks/$t/run_v2.py $m > run_logs/${m}_$t.log 2>&1
done
$py tasks/t6_error_detection/score_v2_phase1.py $m > run_logs/${m}_score.log 2>&1
$py tasks/t6_error_detection/run_v2_control2.py $m > run_logs/${m}_control2.log 2>&1
$py tasks/t8_loop/run_v3.py $m > run_logs/${m}_loop.log 2>&1
$py tasks/t5_selective_deferral/run_v2.py $m --counterfactual > run_logs/${m}_counterfactual.log 2>&1
for t in t6_error_detection t5_selective_deferral t2_confidence_wagering t7_consistency t3_source_monitoring t4_strategy_switching; do
  $py tasks/$t/run_v2.py $m >> run_logs/${m}_$t.log 2>&1   # resumable: reruns only "error" records
done
echo "DONE $m" > run_logs/${m}_DONE
