from __future__ import annotations

from pathlib import Path
from typing import (
    List,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from .config import DataConfig

from .preprocessing import (
    TrajectoryPreprocessor,
)


def resolve_split_path(
    data_dir: str | Path,
    split: str,
) -> Path:
    """
    train / validation / test 파일을 찾는다.

    parquet을 우선 사용하고,
    없으면 CSV를 사용한다.
    """

    data_dir = Path(
        data_dir
    )

    split = split.lower()

    if split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            f"Unknown split: {split}"
        )

    parquet_path = (
        data_dir
        / f"{split}.parquet"
    )

    csv_path = (
        data_dir
        / f"{split}.csv"
    )

    if parquet_path.exists():
        return parquet_path

    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Could not find "
        f"{parquet_path.name} "
        f"or {csv_path.name} "
        f"inside {data_dir}"
    )


def load_split_dataframe(
    data_dir: str | Path,
    split: str,
) -> pd.DataFrame:

    path = resolve_split_path(
        data_dir,
        split,
    )

    if path.suffix == ".parquet":
        return pd.read_parquet(
            path
        )

    return pd.read_csv(
        path
    )


def validate_dataframe_columns(
    df: pd.DataFrame,
    config: DataConfig,
) -> None:

    if (
        not isinstance(
            config.time_column,
            str,
        )
        or not config.time_column.strip()
    ):
        raise ValueError(
            "DataConfig.time_column must be "
            "a non-empty column name."
        )

    required = {
        config.scenario_column,
        config.time_column,
        config.target_column,
        *config.input_columns,
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Dataset is missing "
            "required columns: "
            + ", ".join(missing)
        )


def sort_and_validate_scenario_timestamps(
    scenario_df: pd.DataFrame,
    config: DataConfig,
    scenario_id: object,
) -> pd.DataFrame:
    """
    Timestamp를 숫자형으로 검증하고 stable sort한다.

    정렬 후 timestamp는 반드시 유한하고 엄격히 증가해야 한다.
    """

    try:
        timestamps = pd.to_numeric(
            scenario_df[
                config.time_column
            ],
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scenario {scenario_id!r} "
            f"contains a non-numeric timestamp "
            f"in column {config.time_column!r}."
        ) from exc

    timestamp_values = timestamps.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(
        np.isfinite(timestamp_values)
    ):
        raise ValueError(
            f"Scenario {scenario_id!r} "
            "contains NaN/Inf timestamp "
            f"in column {config.time_column!r}."
        )

    # 숫자로 변환한 timestamp를 정렬 키로 사용한다.
    # kind="stable"은 동일 키의 기존 순서를 보존하지만,
    # 동일 timestamp 자체는 아래에서 명시적으로 거부한다.
    scenario_df = (
        scenario_df
        .assign(
            **{
                config.time_column:
                    timestamp_values
            }
        )
        .sort_values(
            config.time_column,
            kind="stable",
        )
    )

    sorted_timestamps = scenario_df[
        config.time_column
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    if sorted_timestamps.size < 2:
        return scenario_df

    deltas = np.diff(
        sorted_timestamps
    )

    duplicate_positions = np.flatnonzero(
        deltas == 0
    )

    if duplicate_positions.size:
        position = int(
            duplicate_positions[0]
        )
        timestamp = sorted_timestamps[
            position
        ]

        raise ValueError(
            f"Scenario {scenario_id!r} "
            "contains duplicate timestamp "
            f"{timestamp!r} in column "
            f"{config.time_column!r} "
            f"at sorted positions {position} "
            f"and {position + 1}."
        )

    non_increasing_positions = (
        np.flatnonzero(
            deltas < 0
        )
    )

    if non_increasing_positions.size:
        position = int(
            non_increasing_positions[0]
        )

        raise ValueError(
            f"Scenario {scenario_id!r} "
            "contains non-increasing "
            f"timestamps in column "
            f"{config.time_column!r}: "
            f"{sorted_timestamps[position]!r} "
            "followed by "
            f"{sorted_timestamps[position + 1]!r}."
        )

    return scenario_df


def fit_preprocessor_from_dataframe(
    df: pd.DataFrame,
    config: DataConfig,
) -> TrajectoryPreprocessor:
    """
    반드시 train dataframe으로만 호출한다.
    """

    validate_dataframe_columns(
        df,
        config,
    )

    inputs = df.loc[
        :,
        list(
            config.input_columns
        ),
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    targets = df.loc[
        :,
        config.target_column,
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    return (
        TrajectoryPreprocessor()
        .fit(
            inputs,
            targets,
        )
    )


class IMUForecastDataset(
    Dataset
):
    """
    Scenario-safe sliding-window dataset.

    output:

        x:
            (history_steps, 6)

        y:
            (prediction_steps,)

    한 window가 서로 다른 scenario를
    절대 가로지르지 않는다.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: DataConfig,
        preprocessor:
            TrajectoryPreprocessor,
    ) -> None:

        super().__init__()

        if not preprocessor.is_fitted:
            raise ValueError(
                "preprocessor must be "
                "fitted before building "
                "dataset"
            )

        validate_dataframe_columns(
            dataframe,
            config,
        )

        self.config = config
        self.preprocessor = preprocessor

        self._scenario_inputs: List[np.ndarray] = []

        self._scenario_targets: List[np.ndarray] = []

        self._scenario_ids: List[object] = []

        # (scenario_index, start_index)
        self._index: List[
                Tuple[int, int]
            ] = []

        grouped = dataframe.groupby(
            config.scenario_column,
            sort=False,
            observed=True,
        )

        for (
            scenario_id,
            scenario_df,
        ) in grouped:

            scenario_df = (
                sort_and_validate_scenario_timestamps(
                    scenario_df,
                    config,
                    scenario_id,
                )
            )

            raw_x = (
                scenario_df.loc[
                    :,
                    list(
                        config.input_columns
                    ),
                ]
                .to_numpy(
                    dtype=np.float32,
                    copy=True,
                )
            )

            raw_y = (
                scenario_df.loc[
                    :,
                    config.target_column,
                ]
                .to_numpy(
                    dtype=np.float32,
                    copy=True,
                )
            )

            if not np.all(
                np.isfinite(raw_x)
            ):
                raise ValueError(
                    f"Scenario "
                    f"{scenario_id!r} "
                    "contains NaN/Inf "
                    "in IMU inputs."
                )

            if not np.all(
                np.isfinite(raw_y)
            ):
                raise ValueError(
                    f"Scenario "
                    f"{scenario_id!r} "
                    "contains NaN/Inf "
                    "in target."
                )

            # history + future보다 짧은
            # scenario는 사용할 수 없음.
            if (
                len(raw_x)
                < config.total_window_steps
            ):
                continue

            # 전체 scenario를 한번에 normalize.
            x = (
                preprocessor
                .transform_inputs(
                    raw_x
                )
            )

            y = (
                preprocessor
                .transform_targets(
                    raw_y
                )
            )

            scenario_index = (
                len(
                    self._scenario_inputs
                )
            )

            self._scenario_inputs.append(
                x
            )

            self._scenario_targets.append(
                y
            )

            self._scenario_ids.append(
                scenario_id
            )

            max_start = (
                len(x)
                - config.total_window_steps
            )

            for start in range(
                0,
                max_start + 1,
                config.window_stride_steps,
            ):
                self._index.append(
                    (
                        scenario_index,
                        start,
                    )
                )

        if not self._index:
            raise ValueError(
                "No valid windows were "
                "created. Check history/"
                "prediction length and "
                "the dataset contents."
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self._index
        )

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        (
            scenario_index,
            start,
        ) = self._index[
            index
        ]

        input_end = (
            start
            + self.config.input_steps
        )

        target_end = (
            input_end
            + self.config.prediction_steps
        )

        x = (
            self._scenario_inputs[
                scenario_index
            ][
                start:
                input_end
            ]
        )

        y = (
            self._scenario_targets[
                scenario_index
            ][
                input_end:
                target_end
            ]
        )

        x = np.ascontiguousarray(
            x,
            dtype=np.float32,
        )

        y = np.ascontiguousarray(
            y,
            dtype=np.float32,
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )

    @property
    def num_scenarios(
        self,
    ) -> int:

        return len(
            self._scenario_inputs
        )

    @property
    def scenario_ids(
        self,
    ) -> Sequence[object]:

        return tuple(
            self._scenario_ids
        )
