#!/usr/bin/env bash
set -euo pipefail

# Run one prediction from a trained checkpoint and a single IMU window.
#
# Usage:
#   ./run_inference.sh CHECKPOINT INPUT_FILE [OUTPUT_FILE]
#
# INPUT_FILE:
#   .npy -> array with exact shape (T, 6)
#   .csv -> exact T rows containing the checkpoint's six IMU columns
#
# Optional environment variables:
#   DEVICE=auto|cpu|cuda
#   PYTHON_BIN=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CHECKPOINT="${1:-}"
INPUT_FILE="${2:-}"
OUTPUT_FILE="${3:-prediction.npy}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${CHECKPOINT}" || -z "${INPUT_FILE}" ]]; then
    echo "Usage: $0 CHECKPOINT INPUT_FILE [OUTPUT_FILE]" >&2
    exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[Error] checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "[Error] input file not found: ${INPUT_FILE}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[Error] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export CHECKPOINT INPUT_FILE OUTPUT_FILE DEVICE
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "============================================================"
echo "V17 predictor inference"
echo "project root : ${PROJECT_ROOT}"
echo "checkpoint   : ${CHECKPOINT}"
echo "input        : ${INPUT_FILE}"
echo "output       : ${OUTPUT_FILE}"
echo "device       : ${DEVICE}"
echo "python       : ${PYTHON_BIN}"
echo "============================================================"

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd

from model.inference import ModelInference


checkpoint_path = Path(os.environ["CHECKPOINT"])
input_path = Path(os.environ["INPUT_FILE"])
output_path = Path(os.environ["OUTPUT_FILE"])
device = os.environ["DEVICE"]

inference = ModelInference(
    checkpoint_path,
    device=device,
)

if input_path.suffix.lower() == ".npy":
    imu_window = np.load(
        input_path,
        allow_pickle=False,
    )
elif input_path.suffix.lower() == ".csv":
    dataframe = pd.read_csv(input_path)
    missing = sorted(
        set(inference.data_config.input_columns)
        .difference(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Input CSV is missing IMU columns: "
            + ", ".join(missing)
        )

    imu_window = dataframe.loc[
        :,
        list(inference.data_config.input_columns),
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )
else:
    raise ValueError(
        "INPUT_FILE must have a .npy or .csv extension."
    )

expected_shape = inference.input_shape

if imu_window.shape != expected_shape:
    raise ValueError(
        f"Input window must have shape {expected_shape}, "
        f"got {imu_window.shape}."
    )

prediction = inference.predict(imu_window)
output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if output_path.suffix.lower() == ".npy":
    np.save(
        output_path,
        prediction,
        allow_pickle=False,
    )
elif output_path.suffix.lower() == ".csv":
    pd.DataFrame(
        {
            "step": np.arange(
                prediction.size
            ),
            "true_vertical_specific_force_mps2": (
                prediction
            ),
        }
    ).to_csv(
        output_path,
        index=False,
    )
else:
    raise ValueError(
        "OUTPUT_FILE must have a .npy or .csv extension."
    )

print(f"input shape : {imu_window.shape}")
print(f"output shape: {prediction.shape}")
print(f"model       : {inference.model_type}")
print(f"device      : {inference.device}")
print(f"saved       : {output_path}")
PY
