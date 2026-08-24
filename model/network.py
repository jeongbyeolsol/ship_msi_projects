from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .config import ModelConfig


class ConvStem(
    nn.Module
):
    """
    Local temporal feature extraction
    + learnable downsampling.

    Input:
        (B, T, 6)

    Output:
        (B, T', C)
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:

        super().__init__()

        channels = (
            config.input_channels,
            *config.conv_channels,
        )

        kernels = (
            config.conv_kernel_sizes
        )

        strides = (
            config.conv_strides
        )

        if (
            len(
                config.conv_channels
            )
            != len(kernels)
        ):
            raise ValueError(
                "conv_channels and "
                "conv_kernel_sizes "
                "must match"
            )

        if (
            len(
                config.conv_channels
            )
            != len(strides)
        ):
            raise ValueError(
                "conv_channels and "
                "conv_strides "
                "must match"
            )

        layers: List[nn.Module] = []

        for (
            in_ch,
            out_ch,
            kernel,
            stride,
        ) in zip(
            channels[:-1],
            channels[1:],
            kernels,
            strides,
        ):

            padding = (
                kernel // 2
            )

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    ),

                    nn.BatchNorm1d(
                        out_ch
                    ),

                    nn.GELU(),

                    nn.Dropout(
                        config.conv_dropout
                    ),
                ]
            )

        self.net = nn.Sequential(
            *layers
        )

        self.output_dim = (
            config.conv_channels[-1]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected input "
                "(B, T, C), "
                f"got {tuple(x.shape)}"
            )

        # (B,T,C)
        # ↓
        # (B,C,T)
        x = x.transpose(
            1,
            2,
        )

        x = self.net(
            x
        )

        # 다시 sequence-first 형태
        #
        # (B,C,T')
        # ↓
        # (B,T',C)
        return x.transpose(
            1,
            2,
        )


class TrajectoryHead(
    nn.Module
):
    """
    Context vector
        ↓
    future trajectory
    """

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
    ) -> None:

        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),

            nn.Linear(
                input_dim,
                config.head_hidden_size,
            ),

            nn.GELU(),

            nn.Dropout(
                config.head_dropout
            ),

            nn.Linear(
                config.head_hidden_size,
                config.output_steps,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(
            x
        )


class ConvLSTMPredictor(
    nn.Module
):
    """
    Baseline model.

    IMU
     ↓
    Conv1D
     ↓
    LSTM
     ↓
    trajectory head
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:

        super().__init__()

        self.config = config

        self.stem = ConvStem(
            config
        )

        self.encoder = nn.LSTM(
            input_size=(
                self.stem.output_dim
            ),

            hidden_size=(
                config.lstm_hidden_size
            ),

            num_layers=(
                config.lstm_num_layers
            ),

            batch_first=True,

            dropout=(
                config.lstm_dropout
                if (
                    config.lstm_num_layers
                    > 1
                )
                else 0.0
            ),
        )

        self.head = (
            TrajectoryHead(
                input_dim=(
                    config
                    .lstm_hidden_size
                ),
                config=config,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # (B, 3000, 6)
        x = self.stem(
            x
        )

        # (B, ~750, 64)
        _, (
            hidden,
            _,
        ) = self.encoder(
            x
        )

        # 최상위 LSTM layer의
        # 마지막 hidden state
        context = hidden[-1]

        # (B, 1500)
        return self.head(
            context
        )


class MambaResidualBlock(
    nn.Module
):
    """
    PreNorm + Mamba + residual.
    """

    def __init__(
        self,
        d_model: int,
        config: ModelConfig,
    ) -> None:

        super().__init__()

        try:
            from mamba_ssm import (
                Mamba
            )

        except ImportError as exc:

            raise ImportError(
                "model_type='mamba' "
                "requires optional "
                "package 'mamba-ssm'. "
                "Install a version "
                "compatible with your "
                "PyTorch/CUDA setup."
            ) from exc

        self.norm = nn.LayerNorm(
            d_model
        )

        self.mamba = Mamba(
            d_model=d_model,

            d_state=(
                config.mamba_d_state
            ),

            d_conv=(
                config.mamba_d_conv
            ),

            expand=(
                config.mamba_expand
            ),
        )

        self.dropout = nn.Dropout(
            config.mamba_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return (
            x
            + self.dropout(
                self.mamba(
                    self.norm(x)
                )
            )
        )


class ConvMambaPredictor(
    nn.Module
):
    """
    Alternative model.

    IMU
     ↓
    Conv1D
     ↓
    Mamba × N
     ↓
    trajectory head
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:

        super().__init__()

        self.config = config

        self.stem = ConvStem(
            config
        )

        d_model = (
            self.stem.output_dim
        )

        self.blocks = (
            nn.ModuleList(
                [
                    MambaResidualBlock(
                        d_model,
                        config,
                    )

                    for _ in range(
                        config
                        .mamba_num_layers
                    )
                ]
            )
        )

        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )

        self.head = (
            TrajectoryHead(
                input_dim=d_model,
                config=config,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.stem(
            x
        )

        for block in self.blocks:
            x = block(
                x
            )

        x = self.final_norm(
            x
        )

        # causal sequence의
        # 마지막 state
        context = x[
            :,
            -1,
            :,
        ]

        return self.head(
            context
        )


def build_model(
    config: ModelConfig,
) -> nn.Module:

    model_type = (
        config.model_type
        .lower()
        .strip()
    )

    if model_type == "lstm":
        return (
            ConvLSTMPredictor(
                config
            )
        )

    if model_type == "mamba":
        return (
            ConvMambaPredictor(
                config
            )
        )

    raise ValueError(
        "Unsupported "
        f"model_type="
        f"{config.model_type!r}. "
        "Choose 'lstm' "
        "or 'mamba'."
    )