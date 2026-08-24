from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig
from .dataset import IMUForecastDataset, load_split_dataframe
from .inference import _load_checkpoint, _restore_dataclass
from .network import build_model
from .preprocessing import TrajectoryPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checkpoints on common scenario-safe windows "
            "and report MSE by forecast horizon."
        )
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )
    return device


def validate_common_contract(
    configs: list[DataConfig],
    preprocessors: list[TrajectoryPreprocessor],
) -> None:
    reference = configs[0]
    contract_fields = (
        "data_dir",
        "scenario_column",
        "time_column",
        "input_columns",
        "target_column",
        "sample_rate_hz",
        "history_seconds",
        "window_stride_seconds",
        "accelerometer_clip_mps2",
    )

    for index, config in enumerate(configs[1:], start=1):
        mismatches = [
            field
            for field in contract_fields
            if getattr(config, field)
            != getattr(reference, field)
        ]
        if mismatches:
            raise ValueError(
                f"Checkpoint {index + 1} cannot be compared; "
                "DataConfig differs in: "
                + ", ".join(mismatches)
            )

    reference_state = preprocessors[0]
    for index, preprocessor in enumerate(
        preprocessors[1:],
        start=1,
    ):
        for name, left, right in (
            (
                "input mean",
                reference_state.input_scaler.mean_,
                preprocessor.input_scaler.mean_,
            ),
            (
                "input std",
                reference_state.input_scaler.std_,
                preprocessor.input_scaler.std_,
            ),
            (
                "target mean",
                reference_state.target_scaler.mean_,
                preprocessor.target_scaler.mean_,
            ),
            (
                "target std",
                reference_state.target_scaler.std_,
                preprocessor.target_scaler.std_,
            ),
        ):
            if not np.allclose(
                left,
                right,
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Checkpoint {index + 1} has different "
                    f"{name} normalization statistics."
                )


def horizon_bands(
    horizon_seconds: float,
) -> list[tuple[float, float]]:
    boundaries = (0.0, 1.0, 3.0, 5.0, 10.0, 15.0)
    bands = []
    for start, end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        clipped_end = min(end, horizon_seconds)
        if start < clipped_end:
            bands.append((start, clipped_end))
    return bands


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    device = resolve_device(args.device)
    checkpoints = []
    configs = []
    model_configs = []
    preprocessors = []

    for path in args.checkpoints:
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )
        checkpoint = _load_checkpoint(
            path,
            map_location=torch.device("cpu"),
        )
        checkpoints.append(checkpoint)
        configs.append(
            _restore_dataclass(
                DataConfig,
                checkpoint["data_config"],
            )
        )
        model_configs.append(
            _restore_dataclass(
                ModelConfig,
                checkpoint["model_config"],
            )
        )
        preprocessors.append(
            TrajectoryPreprocessor.from_state_dict(
                checkpoint["preprocessor_state"]
            )
        )

    validate_common_contract(
        configs,
        preprocessors,
    )

    reference_config = configs[0]
    max_steps = max(
        config.prediction_steps
        for config in configs
    )
    common_config = replace(
        reference_config,
        prediction_seconds=(
            max_steps
            / reference_config.sample_rate_hz
        ),
    )

    dataframe = load_split_dataframe(
        common_config.data_dir,
        args.split,
        config=common_config,
    )
    dataset = IMUForecastDataset(
        dataframe,
        common_config,
        preprocessors[0],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    target_std = float(
        np.asarray(
            preprocessors[0].target_scaler.std_
        )
    )

    print("=" * 78)
    print(
        f"split={args.split} | device={device} | "
        f"common_windows={len(dataset):,} | "
        f"max_horizon={max_steps / common_config.sample_rate_hz:g}s"
    )
    print(
        "MSE(normalized) uses the same train target scaler; "
        "physical MSE unit is (m/s²)²."
    )
    print("=" * 78)

    for path, checkpoint, config, model_config in zip(
        args.checkpoints,
        checkpoints,
        configs,
        model_configs,
    ):
        model = build_model(model_config)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.to(device)
        model.eval()

        steps = config.prediction_steps
        squared_error = np.zeros(
            steps,
            dtype=np.float64,
        )
        baseline_error = np.zeros(
            steps,
            dtype=np.float64,
        )
        sample_count = 0

        with torch.inference_mode():
            for inputs, common_targets in loader:
                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )
                targets = common_targets[
                    :,
                    :steps,
                ].to(
                    device,
                    non_blocking=True,
                )
                prediction = model(inputs)
                squared_error += (
                    (prediction - targets)
                    .square()
                    .sum(dim=0)
                    .cpu()
                    .numpy()
                )
                baseline_error += (
                    targets
                    .square()
                    .sum(dim=0)
                    .cpu()
                    .numpy()
                )
                sample_count += inputs.shape[0]

        print(
            f"\n[{path.name}] "
            f"epoch={checkpoint.get('epoch')} | "
            f"horizon={config.prediction_seconds:g}s"
        )

        for start, end in horizon_bands(
            config.prediction_seconds
        ):
            first = int(
                round(start * config.sample_rate_hz)
            )
            last = int(
                round(end * config.sample_rate_hz)
            )
            mse = float(
                squared_error[first:last].sum()
                / ((last - first) * sample_count)
            )
            baseline_mse = float(
                baseline_error[first:last].sum()
                / ((last - first) * sample_count)
            )
            skill = 100.0 * (
                1.0 - mse / baseline_mse
            )
            print(
                f"  {start:g}–{end:g}s | "
                f"MSE={mse:.6f} | "
                f"physical={mse * target_std**2:.6f} | "
                f"baseline={baseline_mse:.6f} | "
                f"skill={skill:+.2f}%"
            )

        overall_mse = float(
            squared_error.sum()
            / (steps * sample_count)
        )
        print(
            f"  0–{config.prediction_seconds:g}s overall | "
            f"MSE={overall_mse:.6f} | "
            f"physical={overall_mse * target_std**2:.6f}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
