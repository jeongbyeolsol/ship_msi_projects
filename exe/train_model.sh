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
#   HISTORY_SECONDS=30
#   PREDICTION_SECONDS=1
#   CHECKPOINT_DIR=model/checkpoints
#   CHECKPOINT_NAME=best_1s.pt
#   RUN_NAME=lstm_1s
#   PYTHON_BIN=python
#
# Example:
#   BATCH_SIZE=32 EPOCHS=100 ./train_model.sh /data/V17_Synthetic_IMU_Dataset lstm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATA_DIR="${1:-${DATA_DIR:-V17_Synthetic_IMU_Dataset}}"
MODEL_TYPE="${2:-${MODEL_TYPE:-lstm}}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
HISTORY_SECONDS="${HISTORY_SECONDS:-30}"
PREDICTION_SECONDS="${PREDICTION_SECONDS:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-model/checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_1s.pt}"
RUN_NAME="${RUN_NAME:-${MODEL_TYPE}_${PREDICTION_SECONDS}s}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[Error] data directory not found: ${DATA_DIR}" >&2
    exit 1
fi

if [[ "${MODEL_TYPE}" != "lstm" && "${MODEL_TYPE}" != "mamba" ]]; then
    echo "[Error] MODEL_TYPE must be 'lstm' or 'mamba': ${MODEL_TYPE}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[Error] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p logs

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="logs/train_${RUN_NAME}_${TIMESTAMP}.log"

echo "============================================================"
echo "V17 predictor training"
echo "project root : ${PROJECT_ROOT}"
echo "data dir     : ${DATA_DIR}"
echo "model type   : ${MODEL_TYPE}"
echo "epochs       : ${EPOCHS}"
echo "batch size   : ${BATCH_SIZE}"
echo "num workers  : ${NUM_WORKERS}"
echo "learning rate: ${LR}"
echo "history sec  : ${HISTORY_SECONDS}"
echo "horizon sec  : ${PREDICTION_SECONDS}"
echo "checkpoint   : ${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
echo "python       : ${PYTHON_BIN}"
echo "log file     : ${LOG_FILE}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total,memory.free \
        --format=csv,noheader || true
    echo "============================================================"
fi

"${PYTHON_BIN}" -m model.train \
    --data-dir "${DATA_DIR}" \
    --model-type "${MODEL_TYPE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --history-seconds "${HISTORY_SECONDS}" \
    --prediction-seconds "${PREDICTION_SECONDS}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-name "${CHECKPOINT_NAME}" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo "Training finished."
echo "Best checkpoint: ${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
echo "Log: ${LOG_FILE}"
echo "============================================================"
