import pytest

from model.evaluate_horizons import (
    horizon_bands,
    validate_common_contract,
)


def test_horizon_bands_are_clipped_to_model_output():
    assert horizon_bands(1.0) == [
        (0.0, 1.0),
    ]
    assert horizon_bands(3.0) == [
        (0.0, 1.0),
        (1.0, 3.0),
    ]
    assert horizon_bands(15.0) == [
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
    ]


def test_comparison_rejects_different_input_contracts(
    tiny_data_config,
    fitted_preprocessor,
):
    changed = type(tiny_data_config)(
        sample_rate_hz=4,
        history_seconds=1.0,
        prediction_seconds=1.0,
        window_stride_seconds=0.5,
    )

    with pytest.raises(
        ValueError,
        match="sample_rate_hz",
    ):
        validate_common_contract(
            [tiny_data_config, changed],
            [fitted_preprocessor, fitted_preprocessor],
        )
