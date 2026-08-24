#!/usr/bin/env bash
set -euo pipefail

# Launch the complete real-time system.
#
# Usage:
#   ./run_system.sh [CHECKPOINT]
#
# Examples:
#   ./run_system.sh
#   ./run_system.sh model/checkpoints/best.pt
#
# Optional:
#   PYTHON_BIN=python
#
# main.py should read the checkpoint from:
#
#   MODEL_PATH = os.getenv(
#       "MODEL_PATH",
#       "model/checkpoints/best.pt",
#   )
#
# If main.py still has a hard-coded MODEL_PATH, either apply the one-line
# change above or use the default model/checkpoints/best.pt path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/../main.py" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

cd "${PROJECT_ROOT}"

CHECKPOINT="${1:-${MODEL_PATH:-model/checkpoints/best.pt}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[Error] checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "main.py" ]]; then
    echo "[Error] main.py not found under ${PROJECT_ROOT}" >&2
    exit 1
fi

export MODEL_PATH="${CHECKPOINT}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "V17 real-time prediction/control system"
echo "project root : ${PROJECT_ROOT}"
echo "checkpoint   : ${MODEL_PATH}"
echo "python       : ${PYTHON_BIN}"
echo "============================================================"
echo "Press Ctrl+C for safe shutdown."
echo "============================================================"

exec "${PYTHON_BIN}" main.py
