#!/usr/bin/env bash
set -euo pipefail

# Run the fast pytest regression suite.
# Any additional arguments are forwarded to pytest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[Error] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON_BIN}" -m pytest "$@"
