from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import DataConfig, ModelConfig
from .network import build_model
from .preprocessing import TrajectoryPreprocessor


def _load_checkpoint(
    path: Path,
    map_location: torch.device,
) -> Mapping[str, Any]:
    """
    Load a checkpoint created by model.train.

    The checkpoint contains NumPy arrays in the preprocessing state, so
    weights_only=False is required on recent PyTorch versions.

    Only load checkpoints produced by this project or another trusted source.
    """
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose
        # the weights_only argument.
        return torch.load(
            path,
            map_location=map_location,
        )


def _restore_dataclass(
    cls,
    state: Mapping[str, Any],
):
    """
    Restore a dataclass from a checkpoint while ignoring unknown future keys.

    Tuple-like fields are normalized back to tuples so checkpoints remain
    usable even if they have passed through JSON or another serialization
    layer that converted tuples to lists.
    """
    valid_names = {
        field.name
        for field in fields(cls)
    }

    kwargs = {
        key: value
        for key, value in state.items()
        if key in valid_names
    }

    if cls is DataConfig:
        if "input_columns" in kwargs:
            kwargs["input_columns"] = tuple(
                kwargs["input_columns"]
            )

    if cls is ModelConfig:
        for key in (
            "conv_channels",
            "conv_kernel_sizes",
            "conv_strides",
        ):
            if key in kwargs:
                kwargs[key] = tuple(
                    kwargs[key]
                )

    return cls(**kwargs)


class ModelInference:
    """
    Runtime inference backend for the IMU future-trajectory predictor.

    External contract
    -----------------
    Input:
        imu_window: np.ndarray
            shape = (T, 6)

            Channel order:
            [
                imu_acc_x_mps2,
                imu_acc_y_mps2,
                imu_acc_z_mps2,
                imu_gyro_x_rad_s,
                imu_gyro_y_rad_s,
                imu_gyro_z_rad_s,
            ]

            Values must be in the original physical units used by the
            training dataset. Do not normalize them before calling predict().

    Output:
        np.ndarray
            shape = (H,)

            The prediction is returned in the original target physical unit,
            not in normalized training space.

    Responsibilities
    ----------------
    - load checkpoint
    - restore DataConfig / ModelConfig
    - restore train-set normalization statistics
    - rebuild the network
    - normalize runtime IMU input
    - run the model
    - inverse-transform the target prediction
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.device = self._resolve_device(
            device
        )

        checkpoint = _load_checkpoint(
            self.checkpoint_path,
            map_location=self.device,
        )

        self._validate_checkpoint(
            checkpoint
        )

        self.data_config = _restore_dataclass(
            DataConfig,
            checkpoint["data_config"],
        )

        self.model_config = _restore_dataclass(
            ModelConfig,
            checkpoint["model_config"],
        )

        self.preprocessor = (
            TrajectoryPreprocessor
            .from_state_dict(
                checkpoint[
                    "preprocessor_state"
                ]
            )
        )

        self._validate_contract()

        self.model = build_model(
            self.model_config
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # Useful metadata for logging/debugging.
        self.epoch = checkpoint.get(
            "epoch"
        )

        self.val_loss = checkpoint.get(
            "val_loss"
        )

        self.val_mae_mps2 = checkpoint.get(
            "val_mae_mps2"
        )

    @staticmethod
    def _resolve_device(
        device: str | torch.device | None,
    ) -> torch.device:
        if device is None:
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if isinstance(
            device,
            torch.device,
        ):
            resolved = device
        else:
            device_str = str(
                device
            ).strip().lower()

            if device_str == "auto":
                return torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

            resolved = torch.device(
                device_str
            )

        if (
            resolved.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA device was requested, "
                "but CUDA is not available."
            )

        return resolved

    @staticmethod
    def _validate_checkpoint(
        checkpoint: Mapping[str, Any],
    ) -> None:
        required = {
            "model_state_dict",
            "data_config",
            "model_config",
            "preprocessor_state",
        }

        missing = sorted(
            required.difference(
                checkpoint.keys()
            )
        )

        if missing:
            raise ValueError(
                "Checkpoint is missing "
                "required entries: "
                + ", ".join(missing)
            )

    def _validate_contract(
        self,
    ) -> None:
        if not np.isclose(
            self.preprocessor.accelerometer_clip_mps2,
            self.data_config.accelerometer_clip_mps2,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Checkpoint contract mismatch: preprocessor "
                "accelerometer clipping limit is "
                f"{self.preprocessor.accelerometer_clip_mps2} m/s², "
                "but DataConfig defines "
                f"{self.data_config.accelerometer_clip_mps2} m/s²."
            )

        if (
            self.model_config.input_channels
            != self.data_config.num_input_channels
        ):
            raise ValueError(
                "Checkpoint contract mismatch: "
                f"model expects "
                f"{self.model_config.input_channels} "
                "input channels, but data config "
                f"defines "
                f"{self.data_config.num_input_channels}."
            )

        if (
            self.model_config.output_steps
            != self.data_config.prediction_steps
        ):
            raise ValueError(
                "Checkpoint contract mismatch: "
                f"model outputs "
                f"{self.model_config.output_steps} "
                "steps, but data config defines "
                f"{self.data_config.prediction_steps}."
            )

        input_mean = np.asarray(
            self.preprocessor
            .input_scaler
            .mean_
        )

        input_std = np.asarray(
            self.preprocessor
            .input_scaler
            .std_
        )

        expected_shape = (
            self.data_config
            .num_input_channels,
        )

        if input_mean.shape != expected_shape:
            raise ValueError(
                "Input scaler mean shape "
                f"{input_mean.shape} does not "
                f"match expected "
                f"{expected_shape}."
            )

        if input_std.shape != expected_shape:
            raise ValueError(
                "Input scaler std shape "
                f"{input_std.shape} does not "
                f"match expected "
                f"{expected_shape}."
            )

    def _validate_input(
        self,
        imu_window: np.ndarray,
    ) -> None:
        expected_shape = (
            self.data_config.input_steps,
            self.data_config.num_input_channels,
        )

        if imu_window.shape != expected_shape:
            raise ValueError(
                "imu_window must have shape "
                f"{expected_shape}, "
                f"got {imu_window.shape}."
            )

        if not np.all(
            np.isfinite(
                imu_window
            )
        ):
            raise ValueError(
                "imu_window contains "
                "NaN or Inf."
            )

    def predict(
        self,
        imu_window: np.ndarray,
    ) -> np.ndarray:
        """
        Predict a future target trajectory from one raw IMU history window.

        Parameters
        ----------
        imu_window:
            Raw, unnormalized IMU data with shape (T, 6).

        Returns
        -------
        np.ndarray
            Future trajectory with shape (H,), dtype float32,
            inverse-transformed to the target's physical unit.
        """
        imu_window = np.asarray(
            imu_window,
            dtype=np.float32,
        )

        self._validate_input(
            imu_window
        )

        normalized = (
            self.preprocessor
            .transform_inputs(
                imu_window
            )
        )

        normalized = (
            np.ascontiguousarray(
                normalized,
                dtype=np.float32,
            )
        )

        x = torch.from_numpy(
            normalized
        ).unsqueeze(0)

        x = x.to(
            self.device,
            non_blocking=(
                self.device.type
                == "cuda"
            ),
        )

        with torch.inference_mode():
            prediction_normalized = (
                self.model(
                    x
                )
            )

        if (
            prediction_normalized.ndim != 2
            or prediction_normalized.shape[0] != 1
            or prediction_normalized.shape[1]
            != self.data_config.prediction_steps
        ):
            raise RuntimeError(
                "Unexpected model output "
                f"shape: "
                f"{tuple(prediction_normalized.shape)}"
            )

        prediction_normalized = (
            prediction_normalized[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        prediction = (
            self.preprocessor
            .inverse_targets(
                prediction_normalized
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        ).reshape(-1)

        if (
            prediction.size
            != self.data_config.prediction_steps
        ):
            raise RuntimeError(
                "Unexpected inverse-transformed "
                "prediction length: "
                f"{prediction.size}"
            )

        if not np.all(
            np.isfinite(
                prediction
            )
        ):
            raise RuntimeError(
                "Model prediction contains "
                "NaN or Inf."
            )

        return prediction

    @property
    def input_shape(
        self,
    ) -> tuple[int, int]:
        return (
            self.data_config.input_steps,
            self.data_config.num_input_channels,
        )

    @property
    def output_shape(
        self,
    ) -> tuple[int]:
        return (
            self.data_config.prediction_steps,
        )

    @property
    def sample_rate_hz(
        self,
    ) -> float:
        return float(
            self.data_config.sample_rate_hz
        )

    @property
    def input_steps(
        self,
    ) -> int:
        return self.data_config.input_steps

    @property
    def num_input_channels(
        self,
    ) -> int:
        return (
            self.data_config.num_input_channels
        )

    @property
    def prediction_steps(
        self,
    ) -> int:
        return (
            self.data_config.prediction_steps
        )

    @property
    def prediction_seconds(
        self,
    ) -> float:
        return float(
            self.data_config.prediction_seconds
        )

    @property
    def model_type(
        self,
    ) -> str:
        return (
            self.model_config.model_type
        )
