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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[Error] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export MODEL_PATH="${CHECKPOINT}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "============================================================"
echo "V17 real-time prediction/control system"
echo "project root : ${PROJECT_ROOT}"
echo "checkpoint   : ${MODEL_PATH}"
echo "python       : ${PYTHON_BIN}"
echo "============================================================"
echo "Press Ctrl+C for safe shutdown."
echo "============================================================"

exec "${PYTHON_BIN}" main.py
