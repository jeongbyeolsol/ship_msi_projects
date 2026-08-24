from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from model.config import (
    DataConfig,
    ModelConfig,
)
from model.inference import ModelInference
from model.network import build_model
from model.preprocessing import TrajectoryPreprocessor


@pytest.fixture
def tiny_data_config() -> DataConfig:
    return DataConfig(
        sample_rate_hz=2,
        history_seconds=1.0,
        prediction_seconds=1.0,
        window_stride_seconds=0.5,
    )


@pytest.fixture
def synthetic_dataframe(
    tiny_data_config: DataConfig,
) -> pd.DataFrame:
    config = tiny_data_config
    rows = []

    # Scenario가 섞여 있고 각 scenario의 시간도 뒤섞인 입력.
    for timestamp in (2, 0, 5, 1, 4, 3):
        for scenario_id, offset in (
            ("scenario-a", 0.0),
            ("scenario-b", 100.0),
        ):
            value = offset + timestamp
            row = {
                config.scenario_column: scenario_id,
                config.time_column: float(timestamp),
                config.target_column: value,
            }

            for channel_index, column in enumerate(
                config.input_columns
            ):
                row[column] = (
                    value
                    + channel_index * 0.01
                )

            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def fitted_preprocessor(
    synthetic_dataframe: pd.DataFrame,
    tiny_data_config: DataConfig,
) -> TrajectoryPreprocessor:
    config = tiny_data_config

    return TrajectoryPreprocessor().fit(
        synthetic_dataframe[
            list(config.input_columns)
        ].to_numpy(
            dtype=np.float32
        ),
        synthetic_dataframe[
            config.target_column
        ].to_numpy(
            dtype=np.float32
        ),
    )


@pytest.fixture
def tiny_inference(
    tmp_path,
    synthetic_dataframe,
    tiny_data_config,
    fitted_preprocessor,
):
    config = tiny_data_config
    model_config = replace(
        ModelConfig(
            conv_channels=(4,),
            conv_kernel_sizes=(3,),
            conv_strides=(1,),
            conv_dropout=0.0,
            lstm_hidden_size=4,
            lstm_num_layers=1,
            lstm_dropout=0.0,
            head_hidden_size=4,
            head_dropout=0.0,
        ),
        input_channels=config.num_input_channels,
        output_steps=config.prediction_steps,
    )
    model = build_model(model_config)
    checkpoint_path = tmp_path / "tiny.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "data_config": config.to_dict(),
            "model_config": model_config.to_dict(),
            "preprocessor_state": (
                fitted_preprocessor.state_dict()
            ),
        },
        checkpoint_path,
    )

    inference = ModelInference(
        checkpoint_path,
        device="cpu",
    )
    scenario = (
        synthetic_dataframe[
            synthetic_dataframe[
                config.scenario_column
            ]
            == "scenario-a"
        ]
        .sort_values(
            config.time_column,
            kind="stable",
        )
    )
    raw_window = scenario[
        list(config.input_columns)
    ].iloc[
        :config.input_steps
    ].to_numpy(
        dtype=np.float32
    )

    model.eval()
    normalized_window = (
        fitted_preprocessor.transform_inputs(
            raw_window
        )
    )

    with torch.inference_mode():
        expected_normalized = model(
            torch.from_numpy(
                normalized_window
            ).unsqueeze(0)
        )[0].numpy()

    expected_prediction = (
        fitted_preprocessor.inverse_targets(
            expected_normalized
        )
    )

    return (
        inference,
        raw_window,
        expected_prediction,
    )
