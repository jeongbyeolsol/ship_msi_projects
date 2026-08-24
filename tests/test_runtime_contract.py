import pytest

import main
from model.config import DataConfig
from predictor import Predictor


@pytest.fixture
def predictor_with_tiny_checkpoint(
    tiny_inference,
):
    inference, _, _ = tiny_inference
    predictor = object.__new__(Predictor)
    predictor.checkpoint_path = (
        inference.checkpoint_path
    )
    predictor.model = inference
    return predictor


def test_checkpoint_contract_properties_are_exposed(
    predictor_with_tiny_checkpoint,
    tiny_data_config,
):
    predictor = predictor_with_tiny_checkpoint

    assert predictor.sample_rate_hz == pytest.approx(
        tiny_data_config.sample_rate_hz
    )
    assert (
        predictor.input_steps
        == tiny_data_config.input_steps
    )
    assert (
        predictor.num_input_channels
        == tiny_data_config.num_input_channels
    )
    assert (
        predictor.prediction_steps
        == tiny_data_config.prediction_steps
    )
    assert predictor.prediction_seconds == pytest.approx(
        tiny_data_config.prediction_seconds
    )


def test_matching_runtime_contract_is_accepted(
    predictor_with_tiny_checkpoint,
    tiny_data_config,
):
    predictor_with_tiny_checkpoint.validate_runtime_contract(
        sample_rate_hz=(
            tiny_data_config.sample_rate_hz
        ),
        input_steps=tiny_data_config.input_steps,
        num_imu_channels=(
            tiny_data_config.num_input_channels
        ),
        prediction_steps=(
            tiny_data_config.prediction_steps
        ),
        prediction_seconds=(
            tiny_data_config.prediction_seconds
        ),
    )


@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"sample_rate_hz": 99.0}, "sample_rate_hz"),
        ({"input_steps": 999}, "input_steps"),
        ({"num_imu_channels": 5}, "IMU channel count"),
        ({"prediction_steps": 999}, "prediction_steps"),
        (
            {"prediction_seconds": 99.0},
            "prediction horizon seconds",
        ),
    ],
)
def test_runtime_contract_mismatch_fails_immediately(
    predictor_with_tiny_checkpoint,
    tiny_data_config,
    override,
    expected_message,
):
    runtime = {
        "sample_rate_hz": (
            tiny_data_config.sample_rate_hz
        ),
        "input_steps": tiny_data_config.input_steps,
        "num_imu_channels": (
            tiny_data_config.num_input_channels
        ),
        "prediction_steps": (
            tiny_data_config.prediction_steps
        ),
        "prediction_seconds": (
            tiny_data_config.prediction_seconds
        ),
    }
    runtime.update(override)

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        predictor_with_tiny_checkpoint.validate_runtime_contract(
            **runtime
        )


def test_main_passes_all_runtime_settings_to_predictor():
    class RecordingPredictor:
        def __init__(self):
            self.received = None

        def validate_runtime_contract(self, **runtime):
            self.received = runtime

    predictor = RecordingPredictor()
    main.validate_predictor_runtime_contract(
        predictor
    )

    assert predictor.received == {
        "sample_rate_hz": main.SENSOR_SAMPLE_RATE_HZ,
        "input_steps": main.INPUT_STEPS,
        "num_imu_channels": main.NUM_IMU_CHANNELS,
        "prediction_steps": main.PREDICTION_STEPS,
        "prediction_seconds": main.PREDICTION_SECONDS,
    }


def test_main_runtime_constants_match_default_data_config():
    config = DataConfig()

    assert main.SENSOR_SAMPLE_RATE_HZ == pytest.approx(
        config.sample_rate_hz
    )
    assert main.INPUT_STEPS == config.input_steps
    assert (
        main.NUM_IMU_CHANNELS
        == config.num_input_channels
    )
    assert (
        main.PREDICTION_STEPS
        == config.prediction_steps
    )
    assert main.PREDICTION_SECONDS == pytest.approx(
        config.prediction_seconds
    )
