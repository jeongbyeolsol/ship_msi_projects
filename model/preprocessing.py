from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .config import DEFAULT_ACCELEROMETER_CLIP_MPS2


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

        accelerometer x/y/z는 ±16 g로 clipping한 뒤,
        각 channel을 train statistics로
        개별 standardization.

    Target:
        vertical trajectory

        scalar mean/std로 standardization.
    """

    def __init__(
        self,
        accelerometer_clip_mps2: float = (
            DEFAULT_ACCELEROMETER_CLIP_MPS2
        ),
    ) -> None:

        if (
            not np.isscalar(accelerometer_clip_mps2)
            or not np.isfinite(accelerometer_clip_mps2)
            or accelerometer_clip_mps2 <= 0
        ):
            raise ValueError(
                "accelerometer_clip_mps2 must be "
                "a finite number > 0, got "
                f"{accelerometer_clip_mps2!r}."
            )

        self.accelerometer_clip_mps2 = float(
            accelerometer_clip_mps2
        )

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

        inputs = self._clip_inputs(inputs)

        # IMU 6채널 각각 mean/std 계산. Target은 clipping하지 않는다.
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

        inputs = self._clip_inputs(inputs)

        return (
            self.input_scaler
            .transform(inputs)
        )

    def _clip_inputs(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Clip only accelerometer x/y/z; leave gyro channels unchanged."""
        inputs = np.asarray(
            inputs,
            dtype=np.float32,
        )

        if inputs.ndim < 1 or inputs.shape[-1] < 3:
            raise ValueError(
                "inputs must contain at least three "
                "accelerometer channels on the last axis, "
                f"got shape {inputs.shape}."
            )

        if not np.all(np.isfinite(inputs)):
            raise ValueError(
                "IMU inputs contain NaN or Inf."
            )

        clipped = inputs.copy()
        clipped[..., :3] = np.clip(
            clipped[..., :3],
            -self.accelerometer_clip_mps2,
            self.accelerometer_clip_mps2,
        )
        return clipped

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
            "accelerometer_clip_mps2": (
                self.accelerometer_clip_mps2
            ),

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

        obj = cls(
            accelerometer_clip_mps2=state.get(
                "accelerometer_clip_mps2",
                DEFAULT_ACCELEROMETER_CLIP_MPS2,
            )
        )

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
