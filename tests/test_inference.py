import numpy as np
import pytest

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


def test_inference_applies_checkpoint_accelerometer_clipping(
    tiny_inference,
):
    inference, raw_window, _ = tiny_inference
    limit = inference.data_config.accelerometer_clip_mps2

    saturated_window = raw_window.copy()
    saturated_window[0, :3] = (
        [limit * 2.0, -limit * 2.0, limit * 3.0]
    )

    clipped_window = saturated_window.copy()
    clipped_window[:, :3] = np.clip(
        clipped_window[:, :3],
        -limit,
        limit,
    )

    np.testing.assert_allclose(
        inference.predict(saturated_window),
        inference.predict(clipped_window),
        rtol=1e-6,
        atol=1e-6,
    )
