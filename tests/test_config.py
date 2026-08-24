import pytest

from model.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
)


@pytest.mark.parametrize(
    ("config_class", "kwargs", "message"),
    [
        (DataConfig, {"sample_rate_hz": 0}, "sample_rate_hz"),
        (DataConfig, {"history_seconds": 0}, "history_seconds"),
        (DataConfig, {"prediction_seconds": -1}, "prediction_seconds"),
        (DataConfig, {"window_stride_seconds": 0}, "window_stride_seconds"),
        (
            DataConfig,
            {"accelerometer_clip_mps2": 0},
            "accelerometer_clip_mps2",
        ),
        (DataConfig, {"input_columns": ()}, "input_columns"),
        (
            ModelConfig,
            {
                "conv_channels": (4,),
                "conv_kernel_sizes": (3, 3),
            },
            "same length",
        ),
        (
            ModelConfig,
            {
                "conv_channels": (0,),
                "conv_kernel_sizes": (3,),
                "conv_strides": (1,),
            },
            "conv_channels",
        ),
        (ModelConfig, {"output_steps": 0}, "output_steps"),
        (ModelConfig, {"lstm_hidden_size": 0}, "lstm_hidden_size"),
        (ModelConfig, {"lstm_num_layers": 0}, "lstm_num_layers"),
        (TrainConfig, {"epochs": 0}, "epochs"),
        (TrainConfig, {"batch_size": 0}, "batch_size"),
        (TrainConfig, {"learning_rate": 0}, "learning_rate"),
        (TrainConfig, {"num_workers": -1}, "num_workers"),
    ],
)
def test_invalid_config_values_are_rejected(
    config_class,
    kwargs,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        config_class(**kwargs)
