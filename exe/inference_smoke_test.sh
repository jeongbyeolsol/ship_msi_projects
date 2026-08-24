#!/usr/bin/env bash
set -euo pipefail

# Inference integration smoke test.
#
# Usage:
#   ./inference_smoke_test.sh [DATA_DIR] [CHECKPOINT]
#
# Examples:
#   ./inference_smoke_test.sh V17_Synthetic_IMU_Dataset
#   ./inference_smoke_test.sh V17_Synthetic_IMU_Dataset model/checkpoints/best.pt
#
# If CHECKPOINT does not exist, a temporary random-weight checkpoint is
# generated from the train split. This tests the inference plumbing only,
# not prediction accuracy.
#
# Optional:
#   MODEL_TYPE=lstm|mamba
#   DEVICE=auto|cpu|cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/../model/inference.py" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

cd "${PROJECT_ROOT}"

DATA_DIR="${1:-${DATA_DIR:-V17_Synthetic_IMU_Dataset}}"
CHECKPOINT="${2:-${CHECKPOINT_PATH:-model/checkpoints/best.pt}}"
MODEL_TYPE="${MODEL_TYPE:-lstm}"
DEVICE="${DEVICE:-auto}"

export DATA_DIR
export CHECKPOINT
export MODEL_TYPE
export DEVICE
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Inference smoke test"
echo "project root : ${PROJECT_ROOT}"
echo "data dir     : ${DATA_DIR}"
echo "checkpoint   : ${CHECKPOINT}"
echo "model type   : ${MODEL_TYPE}"
echo "device       : ${DEVICE}"
echo "============================================================"

python - <<'PY'
import gc
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from model.config import DataConfig, ModelConfig
from model.dataset import (
    fit_preprocessor_from_dataframe,
    load_split_dataframe,
)
from model.inference import ModelInference
from model.network import build_model

# Test the public integration boundary too.
from predictor import Predictor


data_dir = os.environ["DATA_DIR"]
checkpoint_arg = Path(os.environ["CHECKPOINT"])
model_type = os.environ["MODEL_TYPE"]
device = os.environ["DEVICE"]

print("[1/6] Loading a raw IMU sample from train split...")
df = load_split_dataframe(data_dir, "train")

if df.empty:
    raise RuntimeError("Train split is empty.")

if checkpoint_arg.is_file():
    checkpoint_path = checkpoint_arg
    created_temp_checkpoint = False
    print(f"[2/6] Using existing checkpoint: {checkpoint_path}")

    # Read config from checkpoint only to know the expected runtime window.
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    data_cfg = DataConfig(
        **{
            k: v
            for k, v in ckpt["data_config"].items()
            if k in DataConfig.__dataclass_fields__
        }
    )

    if not isinstance(
        data_cfg.input_columns,
        tuple,
    ):
        data_cfg.input_columns = tuple(
            data_cfg.input_columns
        )

else:
    print(
        "[2/6] Requested checkpoint does not exist. "
        "Creating a temporary random-weight checkpoint..."
    )

    created_temp_checkpoint = True
    data_cfg = DataConfig(
        data_dir=data_dir
    )

    first_ids = (
        df[data_cfg.scenario_column]
        .drop_duplicates()
        .head(2)
        .tolist()
    )

    small_df = df[
        df[data_cfg.scenario_column].isin(first_ids)
    ].copy()

    preprocessor = (
        fit_preprocessor_from_dataframe(
            small_df,
            data_cfg,
        )
    )

    model_cfg = replace(
        ModelConfig(
            model_type=model_type
        ),
        input_channels=data_cfg.num_input_channels,
        output_steps=data_cfg.prediction_steps,
    )

    model = build_model(
        model_cfg
    )

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix="v17_inference_smoke_"
        )
    )

    checkpoint_path = (
        tmp_dir
        / "random_smoke.pt"
    )

    torch.save(
        {
            "epoch": 0,
            "val_loss": float("nan"),
            "val_mae_mps2": float("nan"),
            "model_state_dict": model.state_dict(),
            "data_config": data_cfg.to_dict(),
            "model_config": model_cfg.to_dict(),
            "train_config": {},
            "preprocessor_state": preprocessor.state_dict(),
        },
        checkpoint_path,
    )

    del small_df, model, preprocessor
    gc.collect()

    print(
        f"      temporary checkpoint: "
        f"{checkpoint_path}"
    )

print("[3/6] Selecting one complete raw runtime window...")

required_rows = data_cfg.input_steps

sample = None

for _, group in df.groupby(
    data_cfg.scenario_column,
    sort=False,
    observed=True,
):
    if len(group) >= required_rows:
        sample = (
            group.loc[
                :,
                list(
                    data_cfg.input_columns
                ),
            ]
            .iloc[:required_rows]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )
        break

del df
gc.collect()

if sample is None:
    raise RuntimeError(
        "Could not find a scenario long enough "
        f"for {required_rows} input steps."
    )

expected_input = (
    data_cfg.input_steps,
    data_cfg.num_input_channels,
)

if sample.shape != expected_input:
    raise RuntimeError(
        f"Raw sample shape {sample.shape} "
        f"!= expected {expected_input}"
    )

print(f"      raw input: {sample.shape}")

print("[4/6] Loading ModelInference...")
backend = ModelInference(
    checkpoint_path=checkpoint_path,
    device=device,
)

if backend.input_shape != expected_input:
    raise RuntimeError(
        f"Backend input contract "
        f"{backend.input_shape} "
        f"!= {expected_input}"
    )

print(
    f"      backend model : "
    f"{backend.model_type}"
)

print(
    f"      backend device: "
    f"{backend.device}"
)

print("[5/6] Direct inference...")
direct_pred = backend.predict(
    sample
)

expected_output = (
    data_cfg.prediction_steps,
)

if direct_pred.shape != expected_output:
    raise RuntimeError(
        f"Direct prediction shape "
        f"{direct_pred.shape} "
        f"!= expected "
        f"{expected_output}"
    )

if not np.all(
    np.isfinite(
        direct_pred
    )
):
    raise RuntimeError(
        "Direct prediction contains "
        "NaN or Inf."
    )

print(
    f"      output: "
    f"{direct_pred.shape}"
)

print("[6/6] Predictor -> ModelInference integration...")

# Predictor currently selects the backend's default device.
# This step verifies the public API used by main.py.
public_predictor = Predictor(
    checkpoint_path=str(
        checkpoint_path
    )
)

public_pred = public_predictor.predict(
    sample
)

if public_pred.shape != expected_output:
    raise RuntimeError(
        f"Predictor output shape "
        f"{public_pred.shape} "
        f"!= expected "
        f"{expected_output}"
    )

if not np.all(
    np.isfinite(
        public_pred
    )
):
    raise RuntimeError(
        "Predictor output contains "
        "NaN or Inf."
    )

print("============================================================")
print("PASS")
print(f"input       : {sample.shape}")
print(f"output      : {public_pred.shape}")
print(f"model       : {backend.model_type}")
print(f"checkpoint  : {checkpoint_path}")

if created_temp_checkpoint:
    print(
        "note        : random-weight checkpoint was used; "
        "this validates plumbing, not accuracy."
    )

print("============================================================")
PY
