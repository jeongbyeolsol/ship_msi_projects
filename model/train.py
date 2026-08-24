from __future__ import annotations

import argparse
import random

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
)

from .config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
)

from .dataset import (
    IMUForecastDataset,
    fit_preprocessor_from_dataframe,
    load_split_dataframe,
)

from .network import (
    build_model,
)

from .preprocessing import (
    TrajectoryPreprocessor,
)


def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def make_loss(
    loss_type: str,
) -> nn.Module:

    loss_type = (
        loss_type
        .lower()
        .strip()
    )

    if loss_type == "mse":
        return nn.MSELoss()

    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()

    raise ValueError(
        f"Unsupported "
        f"loss_type="
        f"{loss_type!r}"
    )


def build_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> Tuple[
    DataLoader,
    DataLoader,
    TrajectoryPreprocessor,
]:

    # ========================================================
    # Train
    # ========================================================

    print(
        "[Data] Loading "
        "train split..."
    )

    train_df = (
        load_split_dataframe(
            data_config.data_dir,
            "train",
        )
    )

    # Normalizer는 train에서만 fit.
    print(
        "[Data] Fitting "
        "normalization from "
        "train split only..."
    )

    preprocessor = (
        fit_preprocessor_from_dataframe(
            train_df,
            data_config,
        )
    )

    train_dataset = (
        IMUForecastDataset(
            dataframe=train_df,
            config=data_config,
            preprocessor=preprocessor,
        )
    )

    del train_df

    # ========================================================
    # Validation
    # ========================================================

    print(
        "[Data] Loading "
        "validation split..."
    )

    validation_df = (
        load_split_dataframe(
            data_config.data_dir,
            "validation",
        )
    )

    validation_dataset = (
        IMUForecastDataset(
            dataframe=validation_df,
            config=data_config,
            preprocessor=preprocessor,
        )
    )

    del validation_df

    pin_memory = (
        train_config.pin_memory
        and torch.cuda.is_available()
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=(
            train_config.batch_size
        ),

        # row가 아니라
        # 완성된 window 단위 shuffle.
        shuffle=True,

        num_workers=(
            train_config.num_workers
        ),

        pin_memory=pin_memory,

        drop_last=False,

        persistent_workers=(
            train_config.num_workers
            > 0
        ),
    )

    validation_loader = (
        DataLoader(
            validation_dataset,

            batch_size=(
                train_config.batch_size
            ),

            shuffle=False,

            num_workers=(
                train_config.num_workers
            ),

            pin_memory=pin_memory,

            drop_last=False,

            persistent_workers=(
                train_config.num_workers
                > 0
            ),
        )
    )

    print(
        f"[Data] train: "
        f"{len(train_dataset):,} "
        f"windows / "
        f"{train_dataset.num_scenarios} "
        f"scenarios"
    )

    print(
        f"[Data] validation: "
        f"{len(validation_dataset):,} "
        f"windows / "
        f"{validation_dataset.num_scenarios} "
        f"scenarios"
    )

    return (
        train_loader,
        validation_loader,
        preprocessor,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer:
        torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
) -> float:

    model.train()

    total_loss = 0.0
    total_items = 0

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            x
        )

        loss = criterion(
            pred,
            y,
        )

        loss.backward()

        if grad_clip_norm > 0:

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    grad_clip_norm
                ),
            )

        optimizer.step()

        batch_size = (
            x.size(0)
        )

        total_loss += (
            float(
                loss.detach()
            )
            * batch_size
        )

        total_items += (
            batch_size
        )

    return (
        total_loss
        / max(
            1,
            total_items,
        )
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    preprocessor:
        TrajectoryPreprocessor,
) -> Tuple[
    float,
    float,
]:

    model.eval()

    total_loss = 0.0
    total_items = 0

    absolute_error_sum = 0.0
    absolute_error_count = 0

    # target scaler를 torch로 변환해
    # physical unit MAE 계산.
    target_mean = (
        torch.as_tensor(
            preprocessor
            .target_scaler
            .mean_,

            dtype=torch.float32,
            device=device,
        )
    )

    target_std = (
        torch.as_tensor(
            preprocessor
            .target_scaler
            .std_,

            dtype=torch.float32,
            device=device,
        )
    )

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        pred = model(
            x
        )

        loss = criterion(
            pred,
            y,
        )

        batch_size = (
            x.size(0)
        )

        total_loss += (
            float(loss)
            * batch_size
        )

        total_items += (
            batch_size
        )

        # ----------------------------------------------------
        # normalized value
        # ↓
        # physical unit [m/s^2]
        # ----------------------------------------------------

        pred_physical = (
            pred
            * target_std
            + target_mean
        )

        y_physical = (
            y
            * target_std
            + target_mean
        )

        absolute_error_sum += float(
            torch.sum(
                torch.abs(
                    pred_physical
                    - y_physical
                )
            )
        )

        absolute_error_count += (
            y.numel()
        )

    avg_loss = (
        total_loss
        / max(
            1,
            total_items,
        )
    )

    mae_mps2 = (
        absolute_error_sum
        / max(
            1,
            absolute_error_count,
        )
    )

    return (
        avg_loss,
        mae_mps2,
    )


def save_checkpoint(
    path: Path,

    model: nn.Module,

    optimizer:
        torch.optim.Optimizer,

    epoch: int,

    val_loss: float,
    val_mae_mps2: float,

    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,

    preprocessor:
        TrajectoryPreprocessor,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: Dict[
            str,
            object,
        ] = {

        "epoch":
            epoch,

        "val_loss":
            float(
                val_loss
            ),

        "val_mae_mps2":
            float(
                val_mae_mps2
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "data_config":
            data_config.to_dict(),

        "model_config":
            model_config.to_dict(),

        "train_config":
            train_config.to_dict(),

        "preprocessor_state":
            preprocessor.state_dict(),
    }

    torch.save(
        checkpoint,
        path,
    )


def run_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Path:

    # ========================================================
    # Dataset / network contract 동기화
    # ========================================================

    model_config = replace(
        model_config,

        input_channels=(
            data_config
            .num_input_channels
        ),

        output_steps=(
            data_config
            .prediction_steps
        ),
    )

    set_seed(
        train_config.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[Train] device: "
        f"{device}"
    )

    print(
        f"[Train] "
        f"model="
        f"{model_config.model_type}, "

        f"input=("
        f"{data_config.input_steps}, "
        f"{data_config.num_input_channels}), "

        f"target=("
        f"{data_config.prediction_steps},)"
    )

    # ========================================================
    # Data
    # ========================================================

    (
        train_loader,
        validation_loader,
        preprocessor,
    ) = build_dataloaders(
        data_config,
        train_config,
    )

    # ========================================================
    # Model
    # ========================================================

    model = (
        build_model(
            model_config
        )
        .to(device)
    )

    criterion = make_loss(
        train_config.loss_type
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),

            lr=(
                train_config
                .learning_rate
            ),

            weight_decay=(
                train_config
                .weight_decay
            ),
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,

            mode="min",

            factor=(
                train_config
                .scheduler_factor
            ),

            patience=(
                train_config
                .scheduler_patience
            ),
        )
    )

    checkpoint_path = (
        Path(
            train_config
            .checkpoint_dir
        )
        /
        train_config
        .best_checkpoint_name
    )

    best_val_loss = (
        float("inf")
    )

    epochs_without_improvement = 0

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        train_config.epochs + 1,
    ):

        train_loss = (
            train_one_epoch(
                model=model,

                loader=(
                    train_loader
                ),

                optimizer=optimizer,

                criterion=criterion,

                device=device,

                grad_clip_norm=(
                    train_config
                    .grad_clip_norm
                ),
            )
        )

        (
            val_loss,
            val_mae_mps2,
        ) = validate(
            model=model,

            loader=(
                validation_loader
            ),

            criterion=criterion,

            device=device,

            preprocessor=(
                preprocessor
            ),
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{train_config.epochs:03d} | "

            f"train_loss="
            f"{train_loss:.6f} | "

            f"val_loss="
            f"{val_loss:.6f} | "

            f"val_MAE="
            f"{val_mae_mps2:.6f} "
            f"m/s^2 | "

            f"lr="
            f"{current_lr:.3e}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=(
                    checkpoint_path
                ),

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                val_loss=(
                    val_loss
                ),

                val_mae_mps2=(
                    val_mae_mps2
                ),

                data_config=(
                    data_config
                ),

                model_config=(
                    model_config
                ),

                train_config=(
                    train_config
                ),

                preprocessor=(
                    preprocessor
                ),
            )

            print(
                "[Checkpoint] saved: "
                f"{checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >=
            train_config
            .early_stopping_patience
        ):

            print(
                "[Train] Early stopping: "
                "no validation improvement "
                "for "
                f"{epochs_without_improvement} "
                "epochs."
            )

            break

    print(
        "[Train] best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "[Train] best checkpoint: "
        f"{checkpoint_path}"
    )

    return checkpoint_path


def parse_args(
) -> argparse.Namespace:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Train the V17 IMU "
                "future-trajectory "
                "predictor."
            )
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,

        help=(
            "Directory containing "
            "train/validation/test "
            "parquet or CSV files."
        ),
    )

    parser.add_argument(
        "--model-type",

        choices=(
            "lstm",
            "mamba",
        ),

        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    return (
        parser.parse_args()
    )


def main(
) -> None:

    args = parse_args()

    data_config = (
        DataConfig()
    )

    model_config = (
        ModelConfig()
    )

    train_config = (
        TrainConfig()
    )

    # CLI override
    if args.data_dir is not None:

        data_config = replace(
            data_config,
            data_dir=args.data_dir,
        )

    if args.model_type is not None:

        model_config = replace(
            model_config,
            model_type=args.model_type,
        )

    if args.epochs is not None:

        train_config = replace(
            train_config,
            epochs=args.epochs,
        )

    if args.batch_size is not None:

        train_config = replace(
            train_config,
            batch_size=args.batch_size,
        )

    if args.num_workers is not None:

        train_config = replace(
            train_config,
            num_workers=args.num_workers,
        )

    if args.lr is not None:

        train_config = replace(
            train_config,
            learning_rate=args.lr,
        )

    run_training(
        data_config=(
            data_config
        ),

        model_config=(
            model_config
        ),

        train_config=(
            train_config
        ),
    )


if __name__ == "__main__":
    main()