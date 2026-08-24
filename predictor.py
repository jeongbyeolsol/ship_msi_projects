import numpy as np

# ------------------------------------------------------------
# NOTE:
# model/ 내부 구조가 확정되면 실제 inference 클래스를 연결한다.
#
# 예상 형태:
#
# from model.inference import ModelInference
#
# ------------------------------------------------------------

try:
    from model.inference import ModelInference
except ImportError:
    ModelInference = None


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

        미래 trajectory.

        정확히 어떤 물리량을 예측할지는 model 설계 단계에서
        확정한다.
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

        # ----------------------------------------------------
        # model/ 구현 전 임시 보호 코드
        # ----------------------------------------------------

        if ModelInference is None:
            raise ImportError(
                "model.inference.ModelInference가 아직 구현되지 않았습니다. "
                "model/ 구현 후 Predictor와 연결해야 합니다."
            )

        # ----------------------------------------------------
        # 실제 모델 inference backend
        # ----------------------------------------------------

        self.model = ModelInference(
            checkpoint_path=checkpoint_path
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
            shape = (H,)
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