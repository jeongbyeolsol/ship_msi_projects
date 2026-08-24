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

    required = {
        config.scenario_column,
        config.target_column,
        *config.input_columns,
    }

    if config.time_column is not None:
        required.add(
            config.time_column
        )

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

            # 데이터에 명시적인 time column을
            # 지정한 경우에만 정렬.
            #
            # 기본값에서는 원래 file row order 유지.
            if (
                config.time_column
                is not None
            ):
                scenario_df = (
                    scenario_df
                    .sort_values(
                        config.time_column,
                        kind="stable",
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