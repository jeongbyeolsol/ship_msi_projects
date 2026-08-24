import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from model.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
)
from model.preprocessing import TrajectoryPreprocessor
from model.train import (
    train_one_epoch,
    validate,
)


class NonFiniteOnSecondCall(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, prediction, target):
        self.calls += 1

        if self.calls == 2:
            return (
                prediction.sum() * 0
                + torch.tensor(
                    self.value,
                    device=prediction.device,
                )
            )

        return torch.mean(
            (prediction - target) ** 2
        )


def make_scalar_loader():
    return DataLoader(
        TensorDataset(
            torch.tensor(
                [[1.0], [2.0]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0], [0.0]],
                dtype=torch.float32,
            ),
        ),
        batch_size=1,
        shuffle=False,
    )


def make_fitted_preprocessor():
    return TrajectoryPreprocessor().fit(
        np.zeros(
            (2, 6),
            dtype=np.float32,
        ),
        np.zeros(
            2,
            dtype=np.float32,
        ),
    )


def test_training_loss_nan_fails_with_epoch_and_batch():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    with pytest.raises(
        FloatingPointError,
        match=(
            r"Non-finite training loss "
            r"at epoch 3, batch 2"
        ),
    ):
        train_one_epoch(
            model=model,
            loader=make_scalar_loader(),
            optimizer=optimizer,
            criterion=NonFiniteOnSecondCall(
                float("nan")
            ),
            device=torch.device("cpu"),
            grad_clip_norm=1.0,
            epoch=3,
        )


def test_validation_loss_inf_fails_with_epoch_and_batch():
    model = nn.Linear(1, 1)

    with pytest.raises(
        FloatingPointError,
        match=(
            r"Non-finite validation loss "
            r"at epoch 4, batch 2"
        ),
    ):
        validate(
            model=model,
            loader=make_scalar_loader(),
            criterion=NonFiniteOnSecondCall(
                float("inf")
            ),
            device=torch.device("cpu"),
            preprocessor=(
                make_fitted_preprocessor()
            ),
            epoch=4,
        )


def test_non_finite_gradient_fails_before_optimizer_step():
    model = nn.Linear(1, 1, bias=False)
    initial_weight = model.weight.detach().clone()
    model.weight.register_hook(
        lambda gradient: torch.full_like(
            gradient,
            float("nan"),
        )
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    with pytest.raises(
        FloatingPointError,
        match=(
            r"Non-finite gradient at epoch 5, "
            r"batch 1, parameter 'weight'"
        ),
    ):
        train_one_epoch(
            model=model,
            loader=make_scalar_loader(),
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
            grad_clip_norm=1.0,
            epoch=5,
        )

    torch.testing.assert_close(
        model.weight.detach(),
        initial_weight,
    )


def test_missing_checkpoint_is_not_reported_as_success(
    monkeypatch,
    tmp_path,
    capsys,
):
    train_module = importlib.import_module(
        "model.train"
    )
    data_config = DataConfig(
        sample_rate_hz=1,
        history_seconds=1,
        prediction_seconds=1,
        window_stride_seconds=1,
    )
    train_config = TrainConfig(
        epochs=1,
        batch_size=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path),
        best_checkpoint_name="missing.pt",
    )
    preprocessor = make_fitted_preprocessor()

    monkeypatch.setattr(
        train_module,
        "build_dataloaders",
        lambda *args, **kwargs: (
            object(),
            object(),
            preprocessor,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "build_model",
        lambda config: nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        train_module,
        "train_one_epoch",
        lambda **kwargs: 1.0,
    )
    monkeypatch.setattr(
        train_module,
        "validate",
        lambda **kwargs: (1.0, 0.5),
    )
    monkeypatch.setattr(
        train_module,
        "save_checkpoint",
        lambda **kwargs: None,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Checkpoint save did not create a file "
            r"at epoch 1"
        ),
    ):
        train_module.run_training(
            data_config=data_config,
            model_config=ModelConfig(),
            train_config=train_config,
        )

    captured = capsys.readouterr()
    assert "[Train] best checkpoint:" not in captured.out
