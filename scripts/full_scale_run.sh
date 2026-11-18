#!/usr/bin/env bash
# Full-scale dress rehearsal on WSL2 + RAPIDS. Logs wall time, peak RAM (RSS) and
# peak GPU memory per step to reports/full/run_log.txt.
#   usage: bash scripts/full_scale_run.sh [step ...]   (default: all steps)
set -uo pipefail
source ~/.venvs/mixtape/bin/activate
cd "$(dirname "$0")/.."
CFG=configs/full.yaml
LOG=reports/full/run_log.txt
mkdir -p reports/full
STEPS=("${@:-synth ingest prep train-cooccur graph train-sasrec}")
read -ra STEPS <<< "${STEPS[*]}"

run_step() {
  local name=$1; shift
  local gpu_peak_file; gpu_peak_file=$(mktemp)
  echo 0 > "$gpu_peak_file"
  ( while true; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      p=$(cat "$gpu_peak_file"); [ "${m:-0}" -gt "$p" ] && echo "$m" > "$gpu_peak_file"
      sleep 1
    done ) & local mon=$!
  local t0=$SECONDS
  python - "$@" <<'PY'
import resource, runpy, sys
sys.argv = ["mixtape"] + sys.argv[1:]
try:
    runpy.run_module("mixtape", run_name="__main__")
finally:
    print(f"__PEAK_RSS_MB__ {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}", flush=True)
PY
  local rc=$?
  kill $mon 2>/dev/null; wait $mon 2>/dev/null
  printf '%-14s rc=%d  %6ds  gpu_peak=%sMiB\n' "$name" "$rc" "$((SECONDS - t0))" "$(cat "$gpu_peak_file")" | tee -a "$LOG"
  rm -f "$gpu_peak_file"
  return $rc
}

echo "=== $(date -Is) ===" >> "$LOG"
for s in "${STEPS[@]}"; do
  case $s in
    synth)         run_step synth synth --config $CFG ;;
    ingest)        run_step ingest ingest --config $CFG ;;
    prep)          run_step prep prep --config $CFG ;;
    train-cooccur) run_step cooccur train --config $CFG --models popularity cooccur ;;
    graph)         run_step graph graph --config $CFG ;;
    train-sasrec)  run_step sasrec train --config $CFG --models sasrec ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac || { echo "step $s failed"; exit 1; }
done
