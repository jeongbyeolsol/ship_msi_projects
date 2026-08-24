from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class Standardizer:
    """
    NumPy 기반 standardization.

    x_norm = (x - mean) / std
    """

    eps: float = 1e-6

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
    ) -> "Standardizer":

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        if x.size == 0:
            raise ValueError(
                "Cannot fit Standardizer "
                "on an empty array."
            )

        if not np.all(
            np.isfinite(x)
        ):
            raise ValueError(
                "Training data contains "
                "NaN or Inf."
            )

        self.mean_ = np.asarray(
            np.mean(
                x,
                axis=0,
            ),
            dtype=np.float32,
        )

        self.std_ = np.asarray(
            np.std(
                x,
                axis=0,
            ),
            dtype=np.float32,
        )

        self.std_ = np.maximum(
            self.std_,
            self.eps,
        ).astype(
            np.float32
        )

        return self

    @property
    def is_fitted(self) -> bool:
        return (
            self.mean_ is not None
            and self.std_ is not None
        )

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        self._check_fitted()

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        return (
            (x - self.mean_)
            / self.std_
        ).astype(
            np.float32,
            copy=False,
        )

    def inverse_transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        self._check_fitted()

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        return (
            x * self.std_
            + self.mean_
        ).astype(
            np.float32,
            copy=False,
        )

    def state_dict(
        self,
    ) -> Dict[str, Any]:

        self._check_fitted()

        return {
            "eps": float(self.eps),

            "mean": np.asarray(
                self.mean_,
                dtype=np.float32,
            ),

            "std": np.asarray(
                self.std_,
                dtype=np.float32,
            ),
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
    ) -> None:

        self.eps = float(
            state.get(
                "eps",
                1e-6,
            )
        )

        self.mean_ = np.asarray(
            state["mean"],
            dtype=np.float32,
        )

        self.std_ = np.asarray(
            state["std"],
            dtype=np.float32,
        )

    def _check_fitted(
        self,
    ) -> None:

        if not self.is_fitted:
            raise RuntimeError(
                "Standardizer has not "
                "been fitted yet."
            )


class TrajectoryPreprocessor:
    """
    Training / validation / inference에서
    공통으로 사용하는 preprocessing.

    Input:
        6-axis IMU

        각 channel을 train statistics로
        개별 standardization.

    Target:
        vertical trajectory

        scalar mean/std로 standardization.
    """

    def __init__(
        self,
    ) -> None:

        self.input_scaler = (
            Standardizer()
        )

        self.target_scaler = (
            Standardizer()
        )

    @property
    def is_fitted(
        self,
    ) -> bool:

        return (
            self.input_scaler.is_fitted
            and self.target_scaler.is_fitted
        )

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> "TrajectoryPreprocessor":

        inputs = np.asarray(
            inputs,
            dtype=np.float32,
        )

        targets = np.asarray(
            targets,
            dtype=np.float32,
        )

        if inputs.ndim != 2:
            raise ValueError(
                "inputs must have shape "
                f"(N, C), got {inputs.shape}"
            )

        # IMU 6채널 각각 mean/std 계산
        self.input_scaler.fit(
            inputs
        )

        # target은 단일 물리량이므로
        # 모든 timestep에 동일한 scaler 적용
        self.target_scaler.fit(
            targets.reshape(-1)
        )

        return self

    def transform_inputs(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:

        return (
            self.input_scaler
            .transform(inputs)
        )

    def transform_targets(
        self,
        targets: np.ndarray,
    ) -> np.ndarray:

        return (
            self.target_scaler
            .transform(targets)
        )

    def inverse_targets(
        self,
        targets: np.ndarray,
    ) -> np.ndarray:

        return (
            self.target_scaler
            .inverse_transform(targets)
        )

    def state_dict(
        self,
    ) -> Dict[str, Any]:

        if not self.is_fitted:
            raise RuntimeError(
                "TrajectoryPreprocessor "
                "has not been fitted yet."
            )

        return {
            "input_scaler":
                self.input_scaler.state_dict(),

            "target_scaler":
                self.target_scaler.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, Any],
    ) -> "TrajectoryPreprocessor":

        obj = cls()

        obj.input_scaler.load_state_dict(
            state[
                "input_scaler"
            ]
        )

        obj.target_scaler.load_state_dict(
            state[
                "target_scaler"
            ]
        )

        return obj