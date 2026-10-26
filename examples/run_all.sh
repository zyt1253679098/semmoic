#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash examples/run_train_MIntRec.sh
bash examples/run_train_MIntRec2.0.sh
