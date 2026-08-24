from pathlib import Path

import numpy as np
import pandas as pd

from model.config import DataConfig
from model.dataset import (
    load_split_dataframe,
    required_dataframe_columns,
)


def make_custom_config(data_dir):
    return DataConfig(
        data_dir=str(data_dir),
        scenario_column="voyage",
        time_column="timestamp",
        input_columns=(
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ),
        target_column="vertical_target",
        sample_rate_hz=2,
        history_seconds=1,
        prediction_seconds=1,
    )


def make_source_dataframe(config):
    dataframe = pd.DataFrame(
        {
            config.scenario_column: [10, 10, 10],
            config.time_column: [0.0, 0.5, 1.0],
            config.target_column: [1.0, 2.0, 3.0],
            "unused_wave_height": [9.0, 9.0, 9.0],
        }
    )

    for index, column in enumerate(
        config.input_columns
    ):
        dataframe[column] = np.asarray(
            [index, index + 1, index + 2],
            dtype=np.float64,
        )

    return dataframe


def assert_projected_dataframe(
    dataframe,
    config,
):
    assert tuple(dataframe.columns) == (
        required_dataframe_columns(config)
    )
    assert "unused_wave_height" not in dataframe
    assert (
        dataframe[config.time_column].dtype
        == np.dtype("float64")
    )

    for column in (
        *config.input_columns,
        config.target_column,
    ):
        assert (
            dataframe[column].dtype
            == np.dtype("float32")
        )


def test_csv_loads_only_configured_columns_and_dtypes(
    tmp_path,
):
    config = make_custom_config(tmp_path)
    source = make_source_dataframe(config)
    source.to_csv(
        tmp_path / "train.csv",
        index=False,
    )

    loaded = load_split_dataframe(
        tmp_path,
        "train",
        config=config,
    )

    assert_projected_dataframe(
        loaded,
        config,
    )
    assert loaded[config.scenario_column].tolist() == [
        10,
        10,
        10,
    ]


def test_parquet_requests_only_configured_columns(
    monkeypatch,
    tmp_path,
):
    config = make_custom_config(tmp_path)
    source = make_source_dataframe(config)
    parquet_path = tmp_path / "validation.parquet"
    parquet_path.touch()
    received = {}

    def fake_read_parquet(path, *, columns):
        received["path"] = Path(path)
        received["columns"] = tuple(columns)
        return source.loc[:, columns].copy()

    monkeypatch.setattr(
        pd,
        "read_parquet",
        fake_read_parquet,
    )

    loaded = load_split_dataframe(
        tmp_path,
        "validation",
        config=config,
    )

    assert received == {
        "path": parquet_path,
        "columns": (
            required_dataframe_columns(config)
        ),
    }
    assert_projected_dataframe(
        loaded,
        config,
    )
