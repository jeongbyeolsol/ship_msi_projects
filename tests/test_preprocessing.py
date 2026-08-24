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
