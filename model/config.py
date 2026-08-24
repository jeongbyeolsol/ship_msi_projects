import math

from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Tuple


IMU_INPUT_COLUMNS: Tuple[str, ...] = (
    "imu_acc_x_mps2",
    "imu_acc_y_mps2",
    "imu_acc_z_mps2",
    "imu_gyro_x_rad_s",
    "imu_gyro_y_rad_s",
    "imu_gyro_z_rad_s",
)


def _require_positive_real(
    name: str,
    value: object,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a finite number > 0, "
            f"got {value!r}."
        )


def _require_non_negative_real(
    name: str,
    value: object,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a finite number >= 0, "
            f"got {value!r}."
        )


def _require_positive_int(
    name: str,
    value: object,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be an integer > 0, "
            f"got {value!r}."
        )


def _require_non_negative_int(
    name: str,
    value: object,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be an integer >= 0, "
            f"got {value!r}."
        )


def _require_dropout(
    name: str,
    value: object,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0 <= value < 1
    ):
        raise ValueError(
            f"{name} must be a finite number in [0, 1), "
            f"got {value!r}."
        )


def _require_non_empty_string(
    name: str,
    value: object,
) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string, "
            f"got {value!r}."
        )


@dataclass
class DataConfig:
    """Dataset and forecasting-window configuration."""

    data_dir: str = "V17_Synthetic_IMU_Dataset"

    scenario_column: str = "scenario_id"
    time_column: str = "time"

    input_columns: Tuple[str, ...] = field(
        default_factory=lambda: IMU_INPUT_COLUMNS
    )

    # 현재 설계에서의 prediction target
    target_column: str = "true_vertical_specific_force_mps2"

    sample_rate_hz: int = 100

    history_seconds: float = 30.0
    prediction_seconds: float = 15.0

    # Sliding-window stride.
    # 1초 간격으로 새로운 training window 생성.
    window_stride_seconds: float = 1.0

    def __post_init__(self) -> None:
        _require_positive_real(
            "sample_rate_hz",
            self.sample_rate_hz,
        )

        for name, value in (
            (
                "history_seconds",
                self.history_seconds,
            ),
            (
                "prediction_seconds",
                self.prediction_seconds,
            ),
            (
                "window_stride_seconds",
                self.window_stride_seconds,
            ),
        ):
            _require_positive_real(
                name,
                value,
            )

        for name, value in (
            ("data_dir", self.data_dir),
            (
                "scenario_column",
                self.scenario_column,
            ),
            ("time_column", self.time_column),
            (
                "target_column",
                self.target_column,
            ),
        ):
            _require_non_empty_string(
                name,
                value,
            )

        if not isinstance(
            self.input_columns,
            (tuple, list),
        ):
            raise ValueError(
                "input_columns must be a sequence of "
                f"6 channel names, got {self.input_columns!r}."
            )

        if len(self.input_columns) != 6:
            raise ValueError(
                "input_columns must contain exactly "
                "6 IMU channel names for the current "
                f"model structure, got {len(self.input_columns)}."
            )

        for index, column in enumerate(
            self.input_columns
        ):
            _require_non_empty_string(
                f"input_columns[{index}]",
                column,
            )

        if (
            len(set(self.input_columns))
            != len(self.input_columns)
        ):
            raise ValueError(
                "input_columns must not contain "
                "duplicate channel names."
            )

        if self.input_steps <= 0:
            raise ValueError(
                "sample_rate_hz * history_seconds "
                "must produce at least 1 input step."
            )

        if self.prediction_steps <= 0:
            raise ValueError(
                "sample_rate_hz * prediction_seconds "
                "must produce at least 1 prediction step."
            )

    @property
    def input_steps(self) -> int:
        return int(
            round(
                self.sample_rate_hz
                * self.history_seconds
            )
        )

    @property
    def prediction_steps(self) -> int:
        return int(
            round(
                self.sample_rate_hz
                * self.prediction_seconds
            )
        )

    @property
    def window_stride_steps(self) -> int:
        return max(
            1,
            int(
                round(
                    self.sample_rate_hz
                    * self.window_stride_seconds
                )
            ),
        )

    @property
    def total_window_steps(self) -> int:
        return (
            self.input_steps
            + self.prediction_steps
        )

    @property
    def num_input_channels(self) -> int:
        return len(self.input_columns)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    model_type:
        "lstm"
        "mamba"
    """

    model_type: str = "lstm"

    input_channels: int = 6
    output_steps: int = 1500

    # --------------------------------------------------------
    # Conv stem
    #
    # 3000 step
    #   ↓ stride 2
    # 1500
    #   ↓ stride 2
    # 750
    # --------------------------------------------------------

    conv_channels: Tuple[int, ...] = (
        32,
        64,
    )

    conv_kernel_sizes: Tuple[int, ...] = (
        7,
        5,
    )

    conv_strides: Tuple[int, ...] = (
        2,
        2,
    )

    conv_dropout: float = 0.05

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.10

    # --------------------------------------------------------
    # Mamba
    # --------------------------------------------------------

    mamba_num_layers: int = 3
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: float = 0.05

    # --------------------------------------------------------
    # Prediction head
    # --------------------------------------------------------

    head_hidden_size: int = 256
    head_dropout: float = 0.10

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_type, str)
            or self.model_type.strip().lower()
            not in {"lstm", "mamba"}
        ):
            raise ValueError(
                "model_type must be 'lstm' or 'mamba', "
                f"got {self.model_type!r}."
            )

        for name, value in (
            ("input_channels", self.input_channels),
            ("output_steps", self.output_steps),
            (
                "lstm_hidden_size",
                self.lstm_hidden_size,
            ),
            (
                "lstm_num_layers",
                self.lstm_num_layers,
            ),
            (
                "mamba_num_layers",
                self.mamba_num_layers,
            ),
            ("mamba_d_state", self.mamba_d_state),
            ("mamba_d_conv", self.mamba_d_conv),
            ("mamba_expand", self.mamba_expand),
            (
                "head_hidden_size",
                self.head_hidden_size,
            ),
        ):
            _require_positive_int(
                name,
                value,
            )

        for field_name, values in (
            ("conv_channels", self.conv_channels),
            (
                "conv_kernel_sizes",
                self.conv_kernel_sizes,
            ),
            ("conv_strides", self.conv_strides),
        ):
            if not isinstance(values, (tuple, list)):
                raise ValueError(
                    f"{field_name} must be a sequence of "
                    f"positive integers, got {values!r}."
                )

        conv_lengths = {
            len(self.conv_channels),
            len(self.conv_kernel_sizes),
            len(self.conv_strides),
        }

        if len(conv_lengths) != 1:
            raise ValueError(
                "conv_channels, conv_kernel_sizes, and "
                "conv_strides must have the same length; "
                f"got {len(self.conv_channels)}, "
                f"{len(self.conv_kernel_sizes)}, and "
                f"{len(self.conv_strides)}."
            )

        if not self.conv_channels:
            raise ValueError(
                "conv_channels, conv_kernel_sizes, and "
                "conv_strides must contain at least one layer."
            )

        for field_name, values in (
            ("conv_channels", self.conv_channels),
            (
                "conv_kernel_sizes",
                self.conv_kernel_sizes,
            ),
            ("conv_strides", self.conv_strides),
        ):
            for index, value in enumerate(values):
                _require_positive_int(
                    f"{field_name}[{index}]",
                    value,
                )

        for name, value in (
            ("conv_dropout", self.conv_dropout),
            ("lstm_dropout", self.lstm_dropout),
            ("mamba_dropout", self.mamba_dropout),
            ("head_dropout", self.head_dropout),
        ):
            _require_dropout(
                name,
                value,
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    seed: int = 42

    epochs: int = 80
    batch_size: int = 16
    num_workers: int = 4

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    grad_clip_norm: float = 1.0

    # "mse" or "smooth_l1"
    loss_type: str = "mse"

    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    early_stopping_patience: int = 12

    checkpoint_dir: str = "model/checkpoints"
    best_checkpoint_name: str = "best.pt"

    pin_memory: bool = True

    def __post_init__(self) -> None:
        _require_non_negative_int(
            "seed",
            self.seed,
        )

        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            (
                "early_stopping_patience",
                self.early_stopping_patience,
            ),
        ):
            _require_positive_int(
                name,
                value,
            )

        for name, value in (
            ("num_workers", self.num_workers),
            (
                "scheduler_patience",
                self.scheduler_patience,
            ),
        ):
            _require_non_negative_int(
                name,
                value,
            )

        _require_positive_real(
            "learning_rate",
            self.learning_rate,
        )

        for name, value in (
            ("weight_decay", self.weight_decay),
            ("grad_clip_norm", self.grad_clip_norm),
        ):
            _require_non_negative_real(
                name,
                value,
            )

        if (
            not isinstance(self.loss_type, str)
            or self.loss_type.strip().lower()
            not in {"mse", "smooth_l1"}
        ):
            raise ValueError(
                "loss_type must be 'mse' or 'smooth_l1', "
                f"got {self.loss_type!r}."
            )

        if (
            isinstance(self.scheduler_factor, bool)
            or not isinstance(self.scheduler_factor, Real)
            or not math.isfinite(
                float(self.scheduler_factor)
            )
            or not 0 < self.scheduler_factor < 1
        ):
            raise ValueError(
                "scheduler_factor must be a finite number "
                "in (0, 1), "
                f"got {self.scheduler_factor!r}."
            )

        _require_non_empty_string(
            "checkpoint_dir",
            self.checkpoint_dir,
        )
        _require_non_empty_string(
            "best_checkpoint_name",
            self.best_checkpoint_name,
        )

    def to_dict(self) -> dict:
        return asdict(self)
