#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT=${DATA_ROOT:-./datasets}
GPU=${GPU:-0}
BERT_PATH=${BERT_PATH:-bert-base-uncased}
python run.py \
  --dataset MIntRec \
  --data_root "$DATA_ROOT" \
  --gpu "$GPU" \
  --bert_path "$BERT_PATH" \
  --seeds "${SEEDS:-0,1,2,3,4}" \
  "$@"
