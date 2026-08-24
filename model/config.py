from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


IMU_INPUT_COLUMNS: Tuple[str, ...] = (
    "imu_acc_x_mps2",
    "imu_acc_y_mps2",
    "imu_acc_z_mps2",
    "imu_gyro_x_rad_s",
    "imu_gyro_y_rad_s",
    "imu_gyro_z_rad_s",
)


@dataclass
class DataConfig:
    """Dataset and forecasting-window configuration."""

    data_dir: str = "V17_Synthetic_IMU_Dataset"

    scenario_column: str = "scenario_id"
    time_column: Optional[str] = None

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

    def to_dict(self) -> dict:
        return asdict(self)