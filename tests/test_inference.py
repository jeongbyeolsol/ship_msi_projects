from dataclasses import replace

import numpy as np
import pytest
import torch

from model.config import ModelConfig
from model.inference import ModelInference
from model.network import build_model


@pytest.fixture
def tiny_inference(
    tmp_path,
    synthetic_dataframe,
    tiny_data_config,
    fitted_preprocessor,
):
    config = tiny_data_config
    model_config = replace(
        ModelConfig(
            conv_channels=(4,),
            conv_kernel_sizes=(3,),
            conv_strides=(1,),
            conv_dropout=0.0,
            lstm_hidden_size=4,
            lstm_num_layers=1,
            lstm_dropout=0.0,
            head_hidden_size=4,
            head_dropout=0.0,
        ),
        input_channels=config.num_input_channels,
        output_steps=config.prediction_steps,
    )
    model = build_model(model_config)
    checkpoint_path = tmp_path / "tiny.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "data_config": config.to_dict(),
            "model_config": model_config.to_dict(),
            "preprocessor_state": (
                fitted_preprocessor.state_dict()
            ),
        },
        checkpoint_path,
    )

    inference = ModelInference(
        checkpoint_path,
        device="cpu",
    )
    scenario = (
        synthetic_dataframe[
            synthetic_dataframe[
                config.scenario_column
            ]
            == "scenario-a"
        ]
        .sort_values(
            config.time_column,
            kind="stable",
        )
    )
    raw_window = scenario[
        list(config.input_columns)
    ].iloc[
        :config.input_steps
    ].to_numpy(
        dtype=np.float32
    )

    model.eval()
    normalized_window = (
        fitted_preprocessor.transform_inputs(
            raw_window
        )
    )

    with torch.inference_mode():
        expected_normalized = model(
            torch.from_numpy(
                normalized_window
            ).unsqueeze(0)
        )[0].numpy()

    expected_prediction = (
        fitted_preprocessor.inverse_targets(
            expected_normalized
        )
    )

    return (
        inference,
        raw_window,
        expected_prediction,
    )


def test_checkpoint_model_inference_round_trip(
    tiny_inference,
):
    (
        inference,
        raw_window,
        expected_prediction,
    ) = tiny_inference
    prediction = inference.predict(raw_window)

    assert inference.input_shape == raw_window.shape
    assert prediction.shape == inference.output_shape
    assert prediction.dtype == np.float32
    assert np.all(np.isfinite(prediction))
    np.testing.assert_allclose(
        prediction,
        expected_prediction,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_inference_rejects_nan_and_inf_input(
    tiny_inference,
    invalid_value,
):
    inference, raw_window, _ = tiny_inference
    invalid_window = raw_window.copy()
    invalid_window[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        inference.predict(invalid_window)
