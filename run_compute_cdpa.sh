#!/usr/bin/env bash
# run_cdpa.sh
# Sequential execution with nohup logging

set -euo pipefail

# ===== Configuration =====
# DATASETS=(trec-covid nfcorpus nq hotpotqa msmarco)
DATASETS=(trec-covid nfcorpus nq hotpotqa msmarco)
JUDGE_MODEL="llama"
RETRIES=1
LOG_INTERVAL=50

BASEPATH="/mnt/ssd/TSF/knot-main/output"
SCRIPT="src/compute_cdpa.py"

LOG_FILE="nohup_cdpa.log"

# ===== Utilities =====
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

log_header() {
  local msg="$1"
  {
    echo ""
    echo "======================================================================="
    echo "[$(timestamp)] $msg"
    echo "======================================================================="
    echo ""
  } >> "$LOG_FILE"
}

# ===== Execution =====
for dataset in "${DATASETS[@]}"; do
  CLEAN_JSON="$BASEPATH/main_task/$dataset/clean_main_task.json"
  WM_JSON="$BASEPATH/main_task/$dataset/wm_main_task.json"
  OUT_DIR="$BASEPATH/main_task/$dataset"

  log_header "Start CDPA for dataset=$dataset"

  nohup python "$SCRIPT" \
    --clean_json "$CLEAN_JSON" \
    --wm_json "$WM_JSON" \
    --out_dir "$OUT_DIR" \
    --judge_model "$JUDGE_MODEL" \
    --retries "$RETRIES" \
    --log_interval "$LOG_INTERVAL" \
    --log_file "$OUT_DIR/cdpa.log" \
    >> "$LOG_FILE" 2>&1

  log_header "Finished CDPA for dataset=$dataset"
done

echo "[$(timestamp)] All datasets finished." >> "$LOG_FILE"

# nohup ./run_compute_cdpa.sh &
