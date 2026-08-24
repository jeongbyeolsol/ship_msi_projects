import numpy as np

from filter import RealTimeFilter


def test_future_msi_filtering_preserves_live_state():
    live_filter = RealTimeFilter(fs=100.0)

    for value in np.linspace(
        8.0,
        10.0,
        num=200,
        dtype=np.float32,
    ):
        live_filter.process(value)

    original_lp_state = live_filter.zi_lp.copy()
    original_iso_state = live_filter.zi_iso.copy()
    raw_prediction = np.full(
        150,
        9.81,
        dtype=np.float32,
    )

    filtered_prediction = live_filter.preview_array(
        raw_prediction
    )

    assert filtered_prediction.shape == raw_prediction.shape
    assert filtered_prediction.dtype == np.float32
    assert not np.allclose(
        filtered_prediction,
        raw_prediction,
    )
    np.testing.assert_array_equal(
        live_filter.zi_lp,
        original_lp_state,
    )
    np.testing.assert_array_equal(
        live_filter.zi_iso,
        original_iso_state,
    )
