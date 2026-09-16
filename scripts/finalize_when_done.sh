#!/usr/bin/env bash
# Wait for the given runs to finish, then run the noise-robustness evaluation on each and
# regenerate docs/RESULTS.md.
# usage: scripts/finalize_when_done.sh <gpu> <run_name> [<run_name> ...]
set -u
cd "$(dirname "$0")/.."
gpu=$1; shift
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  still=()
  for r in "${pending[@]}"; do
    if [ -f "runs/$r/summary.json" ]; then
      echo "$(date '+%F %T') finished: $r"
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m kws.eval_noise "runs/$r" 2>&1 | grep -v Warn
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m kws.eval_protocols "runs/$r" 2>&1 | grep -v Warn
      .venv/bin/python scripts/collect_results.py > /dev/null
    elif [ -f "runs/$r/STOPPED" ] || { grep -q "launched $r on" runs/logs/queue.log && ! pgrep -f -- "--name $r\$" > /dev/null && sleep 30 && [ ! -f "runs/$r/summary.json" ]; }; then
      echo "$(date '+%F %T') not running (crashed or stopped): $r"
    else
      still+=("$r")
    fi
  done
  pending=("${still[@]+"${still[@]}"}")
  [ ${#pending[@]} -gt 0 ] && sleep 120
done
.venv/bin/python scripts/collect_results.py > /dev/null
echo "$(date '+%F %T') all done"
