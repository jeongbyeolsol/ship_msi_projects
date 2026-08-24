import numpy as np

from model.preprocessing import TrajectoryPreprocessor


def test_normalization_state_dict_round_trip(
    synthetic_dataframe,
    tiny_data_config,
    fitted_preprocessor,
):
    config = tiny_data_config
    raw_inputs = synthetic_dataframe[
        list(config.input_columns)
    ].to_numpy(
        dtype=np.float32
    )
    raw_targets = synthetic_dataframe[
        config.target_column
    ].to_numpy(
        dtype=np.float32
    )

    restored = TrajectoryPreprocessor.from_state_dict(
        fitted_preprocessor.state_dict()
    )

    np.testing.assert_allclose(
        restored.transform_inputs(raw_inputs),
        fitted_preprocessor.transform_inputs(
            raw_inputs
        ),
    )
    np.testing.assert_allclose(
        restored.transform_targets(raw_targets),
        fitted_preprocessor.transform_targets(
            raw_targets
        ),
    )
    np.testing.assert_allclose(
        restored.inverse_targets(
            restored.transform_targets(raw_targets)
        ),
        raw_targets,
        rtol=1e-6,
        atol=1e-5,
    )


def test_accelerometer_is_clipped_before_normalization_but_target_and_gyro_are_not():
    clip = 156.96
    inputs = np.array(
        [
            [200.0, -200.0, 10.0, 500.0, -500.0, 2.0],
            [100.0, -100.0, -10.0, -500.0, 500.0, -2.0],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [500.0, -500.0],
        dtype=np.float32,
    )

    preprocessor = TrajectoryPreprocessor(
        accelerometer_clip_mps2=clip,
    ).fit(inputs, targets)

    expected_inputs = inputs.copy()
    expected_inputs[:, :3] = np.clip(
        expected_inputs[:, :3],
        -clip,
        clip,
    )

    np.testing.assert_allclose(
        preprocessor.input_scaler.mean_,
        expected_inputs.mean(axis=0),
    )
    np.testing.assert_allclose(
        preprocessor.input_scaler.std_,
        expected_inputs.std(axis=0),
    )
    np.testing.assert_allclose(
        preprocessor.target_scaler.mean_,
        targets.mean(),
    )
    np.testing.assert_allclose(
        preprocessor.target_scaler.std_,
        targets.std(),
    )
    np.testing.assert_allclose(
        preprocessor.inverse_targets(
            preprocessor.transform_targets(targets)
        ),
        targets,
        rtol=1e-6,
        atol=1e-5,
    )

    restored = TrajectoryPreprocessor.from_state_dict(
        preprocessor.state_dict()
    )
    assert restored.accelerometer_clip_mps2 == clip
    np.testing.assert_allclose(
        restored.transform_inputs(inputs),
        preprocessor.transform_inputs(inputs),
    )
