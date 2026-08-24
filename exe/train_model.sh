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
    2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo "Training finished."
echo "Best checkpoint should be under: model/checkpoints/best.pt"
echo "Log: ${LOG_FILE}"
echo "============================================================"
