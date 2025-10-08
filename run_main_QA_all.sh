#!/usr/bin/env bash
# run_main_QA_all.sh
# Run main_QA.py for multiple datasets (nfcorpus, nq, hotpotqa, msmarco)
# Save logs with nohup

# Usage:
# nohup bash run_main_QA_all.sh > nohup.out 2>&1 &

set -euo pipefail

# DATASETS=(trec-covid nfcorpus nq hotpotqa msmarco)
DATASETS=(trec-covid nfcorpus nq hotpotqa msmarco)
EVAL_MODEL_CODE="contriever"
SCORE_FUNCTION="cosine"
MODEL_NAME_RLLM="llama"
MUTUAL_TIMES=10
TOP_K=10
QA_TOP_K=10

LOG_FILE="nohup_main_QA_all.txt"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

for ds in "${DATASETS[@]}"; do
  {
    echo "======================================================================="
    echo "[$(timestamp)] Start dataset: $ds"
    echo "======================================================================="

    # QA
    echo "[$(timestamp)] Step 2: ask_qa_wm for $ds"
    python src/main_QA.py \
      --eval_dataset "$ds" \
      --eval_model_code "$EVAL_MODEL_CODE" \
      --score_function "$SCORE_FUNCTION" \
      --model_name_rllm "$MODEL_NAME_RLLM" \
      --ask_qa_wm 1 \
      --top_k $TOP_K \
      --qa_top_k $QA_TOP_K

    echo "======================================================================="
    echo "[$(timestamp)] Finished dataset: $ds"
    echo "======================================================================="
    echo ""
  } >> "$LOG_FILE" 2>&1
done
