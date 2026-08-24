#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./exe/compare_horizons.sh [SPLIT] CHECKPOINT...
#
# Example:
#   ./exe/compare_horizons.sh test \
#       model/checkpoints/best_1s.pt \
#       model/checkpoints/best_3s.pt \
#       model/checkpoints/best.pt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SPLIT="${1:-}"
if [[ "${SPLIT}" != "validation" && "${SPLIT}" != "test" ]]; then
    echo "Usage: $0 validation|test CHECKPOINT..." >&2
    exit 2
fi
shift

if [[ "$#" -lt 1 ]]; then
    echo "At least one checkpoint is required." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"

export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON_BIN}" -m model.evaluate_horizons \
    --split "${SPLIT}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    "$@"
