#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train_model.sh [DATA_DIR] [MODEL_TYPE]
#
# Examples:
#   ./train_model.sh V17_Synthetic_IMU_Dataset lstm
#   ./train_model.sh /data/V17_Synthetic_IMU_Dataset mamba
#
# Optional environment variables:
#   EPOCHS=80
#   BATCH_SIZE=16
#   NUM_WORKERS=4
#   LR=1e-3
#
# Example:
#   BATCH_SIZE=32 EPOCHS=100 ./train_model.sh /data/V17_Synthetic_IMU_Dataset lstm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Works both when the script is placed in project root and in project_root/scripts/.
if [[ -f "${SCRIPT_DIR}/../model/train.py" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

cd "${PROJECT_ROOT}"

DATA_DIR="${1:-${DATA_DIR:-V17_Synthetic_IMU_Dataset}}"
MODEL_TYPE="${2:-${MODEL_TYPE:-lstm}}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"

export PYTHONUNBUFFERED=1

mkdir -p logs

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="logs/train_${MODEL_TYPE}_${TIMESTAMP}.log"

echo "============================================================"
echo "V17 predictor training"
echo "project root : ${PROJECT_ROOT}"
echo "data dir     : ${DATA_DIR}"
echo "model type   : ${MODEL_TYPE}"
echo "epochs       : ${EPOCHS}"
echo "batch size   : ${BATCH_SIZE}"
echo "num workers  : ${NUM_WORKERS}"
echo "learning rate: ${LR}"
echo "log file     : ${LOG_FILE}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total,memory.free \
        --format=csv,noheader || true
    echo "============================================================"
fi

python -m model.train \
    --data-dir "${DATA_DIR}" \
    --model-type "${MODEL_TYPE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo "Training finished."
echo "Best checkpoint should be under: model/checkpoints/best.pt"
echo "Log: ${LOG_FILE}"
echo "============================================================"
