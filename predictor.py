from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

from model.inference import ModelInference


@dataclass(frozen=True)
class PredictionResult:
    """입력 snapshot 시각과 함께 전달되는 비동기 추론 결과."""

    trajectory: np.ndarray
    input_timestamp: float
    completed_timestamp: float

    def aligned_trajectory(
        self,
        *,
        now,
        sample_rate_hz,
        max_age_seconds,
    ):
        """
        추론 중 이미 지나간 horizon 앞부분을 제거한다.

        허용 age를 넘었거나 전체 horizon이 지난 결과는 빈 배열로
        반환해 actuator가 오래된 prediction을 사용하지 않게 한다.
        """
        age_seconds = max(
            0.0,
            float(now) - self.input_timestamp,
        )

        if age_seconds > max_age_seconds:
            return np.empty(
                0,
                dtype=np.float32,
            )

        elapsed_steps = int(
            age_seconds * sample_rate_hz
        )

        if elapsed_steps >= self.trajectory.size:
            return np.empty(
                0,
                dtype=np.float32,
            )

        return np.asarray(
            self.trajectory[elapsed_steps:],
            dtype=np.float32,
        ).copy()


class AsyncPredictorWorker:
    """
    하나의 최신 IMU snapshot만 비동기로 추론한다.

    실행 중인 작업이 있으면 새 요청을 거부해 오래된 요청이 queue에
    누적되지 않도록 한다. submit()과 poll()은 100 Hz loop를 막지 않는다.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="imu-inference",
        )
        self._future: Future | None = None
        self._closed = False

    def submit(
        self,
        imu_window,
        *,
        input_timestamp,
    ):
        if self._closed:
            raise RuntimeError(
                "AsyncPredictorWorker is closed."
            )

        if self._future is not None:
            return False

        # deque가 계속 갱신되어도 worker 입력은 변하지 않도록 복제한다.
        snapshot = np.array(
            imu_window,
            dtype=np.float32,
            copy=True,
            order="C",
        )

        self._future = self._executor.submit(
            self._predict,
            snapshot,
            float(input_timestamp),
        )
        return True

    def _predict(
        self,
        snapshot,
        input_timestamp,
    ):
        trajectory = self.predictor.predict(
            snapshot
        )
        return PredictionResult(
            trajectory=np.asarray(
                trajectory,
                dtype=np.float32,
            ).reshape(-1),
            input_timestamp=input_timestamp,
            completed_timestamp=time.perf_counter(),
        )

    def poll(self):
        """완료된 결과만 반환하며 실행 중이면 즉시 None을 반환한다."""
        future = self._future

        if future is None or not future.done():
            return None

        self._future = None
        return future.result()

    @property
    def busy(self):
        return self._future is not None

    def close(self):
        if self._closed:
            return

        self._closed = True
        self._executor.shutdown(
            wait=True,
            cancel_futures=True,
        )


class Predictor:
    """
    선박 운동 예측기의 외부 인터페이스.

    main.py는 model/ 내부 구현을 직접 알 필요 없이
    이 클래스만 사용한다.

    Input
    -----
    imu_window : np.ndarray
        shape = (T, 6)

        channel order:
        [
            imu_acc_x_mps2,
            imu_acc_y_mps2,
            imu_acc_z_mps2,
            imu_gyro_x_rad_s,
            imu_gyro_y_rad_s,
            imu_gyro_z_rad_s,
        ]

    Output
    ------
    future_trajectory : np.ndarray
        shape = (H,)

        미래 true_vertical_specific_force_mps2 trajectory.
        각 값의 단위는 m/s^2이다.
    """

    NUM_IMU_CHANNELS = 6

    def __init__(
        self,
        checkpoint_path,
    ):
        """
        Parameters
        ----------
        checkpoint_path : str
            학습된 모델 checkpoint 경로.

        Notes
        -----
        checkpoint 로딩 방식, device 선택, network 생성 등은
        predictor.py가 아니라 model/ 쪽에서 담당한다.
        """

        self.checkpoint_path = checkpoint_path

        self.model = ModelInference(
            checkpoint_path=checkpoint_path
        )

    @property
    def sample_rate_hz(self):
        return self.model.sample_rate_hz

    @property
    def input_steps(self):
        return self.model.input_steps

    @property
    def num_input_channels(self):
        return self.model.num_input_channels

    @property
    def prediction_steps(self):
        return self.model.prediction_steps

    @property
    def prediction_seconds(self):
        return self.model.prediction_seconds

    def validate_runtime_contract(
        self,
        *,
        sample_rate_hz,
        input_steps,
        num_imu_channels,
        prediction_steps,
        prediction_seconds,
    ):
        """
        Runtime 설정이 checkpoint의 DataConfig와 일치하는지 검사한다.

        모든 불일치를 한 번에 표시해 시스템 루프 진입 전에
        설정 문제를 쉽게 확인할 수 있도록 한다.
        """
        mismatches = []

        if not np.isclose(
            sample_rate_hz,
            self.sample_rate_hz,
            rtol=0.0,
            atol=1e-9,
        ):
            mismatches.append(
                "sample_rate_hz: "
                f"runtime={sample_rate_hz!r}, "
                f"checkpoint={self.sample_rate_hz!r}"
            )

        for name, runtime_value, checkpoint_value in (
            (
                "input_steps",
                input_steps,
                self.input_steps,
            ),
            (
                "IMU channel count",
                num_imu_channels,
                self.num_input_channels,
            ),
            (
                "prediction_steps",
                prediction_steps,
                self.prediction_steps,
            ),
        ):
            if runtime_value != checkpoint_value:
                mismatches.append(
                    f"{name}: "
                    f"runtime={runtime_value!r}, "
                    f"checkpoint={checkpoint_value!r}"
                )

        if not np.isclose(
            prediction_seconds,
            self.prediction_seconds,
            rtol=0.0,
            atol=1e-9,
        ):
            mismatches.append(
                "prediction horizon seconds: "
                f"runtime={prediction_seconds!r}, "
                f"checkpoint={self.prediction_seconds!r}"
            )

        if mismatches:
            details = "\n".join(
                f"- {mismatch}"
                for mismatch in mismatches
            )
            raise ValueError(
                "Runtime configuration does not match "
                "the checkpoint DataConfig:\n"
                f"{details}\n"
                "Use a checkpoint trained for the runtime "
                "settings or update main.py configuration."
            )

    def predict(self, imu_window):
        """
        과거 IMU 시계열을 이용해 미래 trajectory를 예측한다.

        Parameters
        ----------
        imu_window : array-like
            shape = (T, 6)

        Returns
        -------
        np.ndarray
            Future true_vertical_specific_force_mps2 trajectory,
            shape = (H,), unit = m/s^2.
        """

        # ----------------------------------------------------
        # 1. 입력 형식 통일
        # ----------------------------------------------------

        imu_window = np.asarray(
            imu_window,
            dtype=np.float32,
        )

        self._validate_input(imu_window)

        # ----------------------------------------------------
        # 2. Model inference
        #
        # normalization
        # tensor 변환
        # device 이동
        # network forward
        # denormalization
        #
        # 등의 모든 처리는 model/ 쪽에서 담당한다.
        # ----------------------------------------------------

        future_trajectory = self.model.predict(
            imu_window
        )

        # ----------------------------------------------------
        # 3. 외부 인터페이스 형식 통일
        # ----------------------------------------------------

        future_trajectory = np.asarray(
            future_trajectory,
            dtype=np.float32,
        ).reshape(-1)

        self._validate_output(
            future_trajectory
        )

        return future_trajectory

    def _validate_input(self, imu_window):
        """
        Predictor 외부 입력 계약 검사.
        """

        if imu_window.ndim != 2:
            raise ValueError(
                "imu_window must have shape (T, 6), "
                f"got {imu_window.shape}"
            )

        if imu_window.shape[1] != self.NUM_IMU_CHANNELS:
            raise ValueError(
                "imu_window must contain 6 IMU channels, "
                f"got {imu_window.shape[1]}"
            )

        if imu_window.shape[0] == 0:
            raise ValueError(
                "imu_window is empty."
            )

        if not np.all(np.isfinite(imu_window)):
            raise ValueError(
                "imu_window contains NaN or Inf."
            )

    def _validate_output(self, future_trajectory):
        """
        Predictor 외부 출력 계약 검사.
        """

        if future_trajectory.ndim != 1:
            raise ValueError(
                "future_trajectory must be 1-dimensional, "
                f"got {future_trajectory.shape}"
            )

        if future_trajectory.size == 0:
            raise ValueError(
                "future_trajectory is empty."
            )

        if not np.all(
            np.isfinite(future_trajectory)
        ):
            raise ValueError(
                "future_trajectory contains NaN or Inf."
            )
