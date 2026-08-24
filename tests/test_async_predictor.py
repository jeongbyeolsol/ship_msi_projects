import threading
import time

import numpy as np

from predictor import AsyncPredictorWorker, PredictionResult


def test_async_predictor_uses_snapshot_and_does_not_queue():
    started = threading.Event()
    release = threading.Event()

    class BlockingPredictor:
        def __init__(self):
            self.received = None

        def predict(self, window):
            self.received = window.copy()
            started.set()
            assert release.wait(timeout=2.0)
            return np.arange(5, dtype=np.float32)

    predictor = BlockingPredictor()
    worker = AsyncPredictorWorker(predictor)
    window = np.ones((2, 6), dtype=np.float32)

    try:
        assert worker.submit(
            window,
            input_timestamp=10.0,
        )
        assert started.wait(timeout=2.0)

        window[:] = 99.0
        assert not worker.submit(
            window,
            input_timestamp=11.0,
        )
        assert worker.poll() is None

        release.set()
        deadline = time.perf_counter() + 2.0
        result = None
        while result is None and time.perf_counter() < deadline:
            result = worker.poll()
            time.sleep(0.001)

        assert result is not None
        np.testing.assert_array_equal(
            predictor.received,
            np.ones((2, 6), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.trajectory,
            np.arange(5, dtype=np.float32),
        )
        assert result.input_timestamp == 10.0
    finally:
        release.set()
        worker.close()


def test_prediction_result_aligns_horizon_and_rejects_stale_data():
    result = PredictionResult(
        trajectory=np.arange(8, dtype=np.float32),
        input_timestamp=10.0,
        completed_timestamp=10.1,
    )

    np.testing.assert_array_equal(
        result.aligned_trajectory(
            now=10.25,
            sample_rate_hz=4.0,
            max_age_seconds=1.0,
        ),
        np.arange(1, 8, dtype=np.float32),
    )

    assert result.aligned_trajectory(
        now=11.01,
        sample_rate_hz=4.0,
        max_age_seconds=1.0,
    ).size == 0
