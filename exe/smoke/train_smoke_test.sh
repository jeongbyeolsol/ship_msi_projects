#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train_smoke_test.sh [DATA_DIR]
#
# Examples:
#   ./train_smoke_test.sh V17_Synthetic_IMU_Dataset
#   MODEL_TYPE=mamba ./train_smoke_test.sh /path/to/dataset
#
# Optional environment variables:
#   MODEL_TYPE=lstm|mamba   (default: lstm)
#   SMOKE_SCENARIOS=2       (default: 2)
#   SMOKE_ROWS=50000        (default: 50000, CSV read limit)
#   PYTHON_BIN=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

DATA_DIR="${1:-${DATA_DIR:-V17_Synthetic_IMU_Dataset}}"
MODEL_TYPE="${MODEL_TYPE:-lstm}"
SMOKE_SCENARIOS="${SMOKE_SCENARIOS:-2}"
SMOKE_ROWS="${SMOKE_ROWS:-50000}"
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

export DATA_DIR
export MODEL_TYPE
export SMOKE_SCENARIOS
export SMOKE_ROWS
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "============================================================"
echo "V17 predictor smoke test"
echo "project root     : ${PROJECT_ROOT}"
echo "data dir         : ${DATA_DIR}"
echo "model type       : ${MODEL_TYPE}"
echo "smoke scenarios  : ${SMOKE_SCENARIOS}"
echo "CSV row limit    : ${SMOKE_ROWS}"
echo "python           : ${PYTHON_BIN}"
echo "============================================================"

"${PYTHON_BIN}" - <<'PY'
import gc
import os
from dataclasses import replace

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import DataConfig, ModelConfig
from model.dataset import (
    IMUForecastDataset,
    fit_preprocessor_from_dataframe,
    resolve_split_path,
    validate_dataframe_columns,
)
from model.network import build_model


data_dir = os.environ["DATA_DIR"]
model_type = os.environ["MODEL_TYPE"]
num_scenarios = int(os.environ["SMOKE_SCENARIOS"])
max_rows = int(os.environ["SMOKE_ROWS"])

if num_scenarios <= 0:
    raise ValueError("SMOKE_SCENARIOS must be > 0.")

if max_rows <= 0:
    raise ValueError("SMOKE_ROWS must be > 0.")

data_cfg = DataConfig(data_dir=data_dir)
model_cfg = ModelConfig(model_type=model_type)

print("[1/6] Loading train split...")
split_path = resolve_split_path(
    data_cfg.data_dir,
    "train",
)

if split_path.suffix == ".csv":
    df = pd.read_csv(
        split_path,
        nrows=max_rows,
    )
else:
    df = pd.read_parquet(
        split_path
    ).head(max_rows)

print("[2/6] Checking required columns...")
validate_dataframe_columns(
    df,
    data_cfg,
)

scenario_ids = (
    df[data_cfg.scenario_column]
    .drop_duplicates()
    .head(num_scenarios)
    .tolist()
)

if not scenario_ids:
    raise RuntimeError("No scenarios found in train split.")

small_df = df[
    df[data_cfg.scenario_column].isin(scenario_ids)
].copy()

del df
gc.collect()

print(
    f"[3/6] Building smoke dataset from "
    f"{len(scenario_ids)} scenario(s)..."
)

preprocessor = fit_preprocessor_from_dataframe(
    small_df,
    data_cfg,
)

dataset = IMUForecastDataset(
    dataframe=small_df,
    config=data_cfg,
    preprocessor=preprocessor,
)

del small_df
gc.collect()

loader = DataLoader(
    dataset,
    batch_size=min(2, len(dataset)),
    shuffle=True,
    num_workers=0,
)

x, y = next(iter(loader))

expected_x = (
    x.shape[0],
    data_cfg.input_steps,
    data_cfg.num_input_channels,
)
expected_y = (
    y.shape[0],
    data_cfg.prediction_steps,
)

if tuple(x.shape) != expected_x:
    raise RuntimeError(
        f"Unexpected input shape: {tuple(x.shape)} "
        f"(expected {expected_x})"
    )

if tuple(y.shape) != expected_y:
    raise RuntimeError(
        f"Unexpected target shape: {tuple(y.shape)} "
        f"(expected {expected_y})"
    )

print(f"      x shape: {tuple(x.shape)}")
print(f"      y shape: {tuple(y.shape)}")

print("[4/6] Building model...")

model_cfg = replace(
    model_cfg,
    input_channels=data_cfg.num_input_channels,
    output_steps=data_cfg.prediction_steps,
)

model = build_model(model_cfg)

print("[5/6] Forward + backward pass...")

pred = model(x)

if tuple(pred.shape) != tuple(y.shape):
    raise RuntimeError(
        f"Model output shape {tuple(pred.shape)} "
        f"does not match target {tuple(y.shape)}"
    )

loss = F.mse_loss(pred, y)
loss.backward()

finite_grads = True
for p in model.parameters():
    if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
        finite_grads = False
        break

if not torch.isfinite(loss):
    raise RuntimeError("Smoke-test loss is NaN or Inf.")

if not finite_grads:
    raise RuntimeError("Model gradients contain NaN or Inf.")

param_count = sum(p.numel() for p in model.parameters())

print("[6/6] PASS")
print(f"      windows    : {len(dataset):,}")
print(f"      parameters : {param_count:,}")
print(f"      loss       : {loss.item():.6f}")
print(f"      output     : {tuple(pred.shape)}")
PY

echo "============================================================"
echo "Smoke test completed successfully."
echo "============================================================"
