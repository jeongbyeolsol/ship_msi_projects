from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.config import DataConfig
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
