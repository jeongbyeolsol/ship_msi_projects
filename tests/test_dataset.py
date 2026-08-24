import numpy as np
import torch

from model.dataset import IMUForecastDataset


def test_sliding_windows_do_not_cross_scenario_boundaries(
    synthetic_dataframe,
    tiny_data_config,
    fitted_preprocessor,
):
    dataset = IMUForecastDataset(
        synthetic_dataframe,
        tiny_data_config,
        fitted_preprocessor,
    )

    assert dataset.scenario_ids == (
        "scenario-a",
        "scenario-b",
    )
    assert len(dataset) == 6

    for index in range(len(dataset)):
        normalized_x, normalized_y = dataset[index]
        raw_x = (
            fitted_preprocessor
            .input_scaler
            .inverse_transform(
                normalized_x.numpy()
            )
        )
        raw_y = fitted_preprocessor.inverse_targets(
            normalized_y.numpy()
        )
        values = np.concatenate(
            [raw_x[:, 0], raw_y]
        )

        assert (
            np.all(values < 50.0)
            or np.all(values > 50.0)
        )


def test_dataset_input_output_shapes(
    synthetic_dataframe,
    tiny_data_config,
    fitted_preprocessor,
):
    dataset = IMUForecastDataset(
        synthetic_dataframe,
        tiny_data_config,
        fitted_preprocessor,
    )
    x, y = dataset[0]

    assert tuple(x.shape) == (
        tiny_data_config.input_steps,
        tiny_data_config.num_input_channels,
    )
    assert tuple(y.shape) == (
        tiny_data_config.prediction_steps,
    )
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
