import os
import time
from collections import deque

import numpy as np

from imu import MPU6050
from filter import RealTimeFilter
from predictor import AsyncPredictorWorker, Predictor
from msi import MSICalculator
from mpc import LightMPC
from controller import InterceptorController
from gps import GPSSpeedSensor


# ============================================================
# System configuration
# ============================================================

# V17 데이터셋의 IMU sampling rate
SENSOR_SAMPLE_RATE_HZ = 100.0
SENSOR_PERIOD_SEC = 1.0 / SENSOR_SAMPLE_RATE_HZ
NUM_IMU_CHANNELS = 6

# 모델 입력 history
HISTORY_SECONDS = 30.0
INPUT_STEPS = int(
    round(
        SENSOR_SAMPLE_RATE_HZ
        * HISTORY_SECONDS
    )
)

# 모델 출력 horizon
PREDICTION_SECONDS = 15.0
PREDICTION_STEPS = int(
    round(
        SENSOR_SAMPLE_RATE_HZ
        * PREDICTION_SECONDS
    )
)

# Predictor / MPC 제어 갱신 주기
CONTROL_RATE_HZ = 10.0
CONTROL_PERIOD_SEC = 1.0 / CONTROL_RATE_HZ

# 이보다 오래된 prediction은 제어에 사용하지 않는다.
MAX_PREDICTION_AGE_SEC = 0.5

# GPS는 IMU처럼 100 Hz로 읽을 필요가 없음
GPS_RATE_HZ = 5.0
GPS_PERIOD_SEC = 1.0 / GPS_RATE_HZ

# 인터셉터 작동 조건
ACTIVATION_SPEED_KNOTS = 30.0

# Predictor checkpoint
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/checkpoints/best.pt",
)


def validate_predictor_runtime_contract(
    predictor,
):
    """main.py runtime 상수와 checkpoint 계약을 비교한다."""
    predictor.validate_runtime_contract(
        sample_rate_hz=(
            SENSOR_SAMPLE_RATE_HZ
        ),
        input_steps=INPUT_STEPS,
        num_imu_channels=NUM_IMU_CHANNELS,
        prediction_steps=PREDICTION_STEPS,
        prediction_seconds=(
            PREDICTION_SECONDS
        ),
    )


class _RuntimeResources:
    """초기화 도중 실패해도 정리할 수 있도록 생성된 자원을 추적한다."""

    def __init__(self):
        self.imu_sensor = None
        self.gps_sensor = None
        self.actuator = None
        self.ai_predictor = None
        self.prediction_worker = None


def _run_system(resources):
    print("시스템 초기화 중...")

    # --------------------------------------------------------
    # 1. Hardware / module initialization
    # --------------------------------------------------------

    try:
        resources.imu_sensor = MPU6050()
        imu_sensor = resources.imu_sensor
        print("[IMU] MPU6050 초기화 완료")
    except Exception as e:
        print(f"[Error] IMU 초기화 실패: {e}")
        return

    # GPS 클래스 내부에서 연결 실패를 처리하고
    # 실패 시 speed=0.0 knot를 반환한다.
    resources.gps_sensor = GPSSpeedSensor(
        port="/dev/ttyUSB0",
        baudrate=9600,
    )
    gps_sensor = resources.gps_sensor

    # 현재 MSI 계산에 사용할 수직 방향 실시간 필터.
    #
    # Predictor의 입력에는 이 필터를 적용하지 않는다.
    # Predictor 입력은 raw 6-axis IMU이다.
    signal_filter = RealTimeFilter(
        fs=SENSOR_SAMPLE_RATE_HZ
    )

    # Predictor는 model/ 쪽 inference 코드를 호출하는
    # 얇은 wrapper로 구현할 예정.
    try:
        resources.ai_predictor = Predictor(
            checkpoint_path=MODEL_PATH
        )
        ai_predictor = resources.ai_predictor

        validate_predictor_runtime_contract(
            ai_predictor
        )

        print(
            "[Predictor] checkpoint/runtime "
            "계약 검증 완료"
        )

        resources.prediction_worker = (
            AsyncPredictorWorker(
                ai_predictor
            )
        )
        prediction_worker = (
            resources.prediction_worker
        )
    except Exception as e:
        print(
            "[Error] Predictor 초기화 또는 "
            f"runtime 계약 검증 실패: {e}"
        )
        return

    msi_calc = MSICalculator(
        fs=SENSOR_SAMPLE_RATE_HZ,
        window_minutes=20.0,
    )

    mpc_optimizer = LightMPC(
        control_weight=0.1
    )

    resources.actuator = InterceptorController(
        pin=18,
        min_stroke=0.0,
        max_stroke=50.0,
    )
    actuator = resources.actuator

    # --------------------------------------------------------
    # 2. Predictor input buffer
    #
    # 각 원소:
    # [ax, ay, az, gx, gy, gz]
    #
    # 전체 shape:
    # (INPUT_STEPS, NUM_IMU_CHANNELS)
    # = (3000, 6)  # 30 sec @ 100 Hz
    # --------------------------------------------------------

    recent_buffer = deque(
        maxlen=INPUT_STEPS
    )

    # 아직 Predictor가 실행되지 않았을 때 사용
    latest_future_trajectory = np.empty(
        0,
        dtype=np.float32,
    )
    latest_prediction_result = None

    # --------------------------------------------------------
    # 3. Runtime state
    # --------------------------------------------------------

    current_speed = 0.0

    last_gps_time = 0.0
    last_control_time = 0.0
    last_log_time = 0.0

    current_msi = 0.0
    current_msdv = 0.0
    applied_stroke = 0.0

    print(
        "실시간 제어 루프 진입 "
        f"(IMU: {SENSOR_SAMPLE_RATE_HZ:.0f} Hz, "
        f"Predictor/Control: {CONTROL_RATE_HZ:.0f} Hz)"
    )

    print(
        f"Predictor 입력: {HISTORY_SECONDS:.0f}초 "
        f"({INPUT_STEPS} samples × 6 channels)"
    )

    print(
        f"인터셉터 작동 기준: "
        f"{ACTIVATION_SPEED_KNOTS:.1f} Knot 이상"
    )

    # 센서 주기 누적 기준
    next_sensor_time = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()

            # =================================================
            # 1. GPS update
            # =================================================

            if now - last_gps_time >= GPS_PERIOD_SEC:
                current_speed = gps_sensor.read_speed_knots()
                last_gps_time = now

            # =================================================
            # 2. 6-axis IMU acquisition
            # =================================================

            try:
                imu_sample = imu_sensor.read_imu()
            except Exception as e:
                print(f"[Warning] IMU 읽기 실패: {e}")

                # 센서 데이터가 없으면 제어하지 않는다.
                actuator.set_stroke(0.0)

                # 다음 sensor tick으로 이동
                next_sensor_time += SENSOR_PERIOD_SEC
                sleep_time = (
                    next_sensor_time
                    - time.perf_counter()
                )

                if sleep_time > 0:
                    time.sleep(sleep_time)

                continue

            imu_sample = np.asarray(
                imu_sample,
                dtype=np.float32,
            )

            # 예상되지 않은 IMU 형식은 즉시 탐지
            if imu_sample.shape != (
                NUM_IMU_CHANNELS,
            ):
                raise ValueError(
                    "read_imu() must return shape "
                    f"({NUM_IMU_CHANNELS},), "
                    f"got {imu_sample.shape}"
                )

            # Predictor용 raw IMU buffer
            recent_buffer.append(imu_sample)

            # =================================================
            # 3. Current vertical signal for MSI
            # =================================================
            #
            # Predictor 입력과는 별도 경로이다.
            #
            # Predictor:
            #   raw 6-axis IMU
            #
            # MSI current measurement:
            #   vertical acceleration -> ISO filter
            #
            # -------------------------------------------------

            current_vertical_accel = float(
                imu_sample[2]
            )

            current_msi_signal = signal_filter.process(
                current_vertical_accel
            )

            # =================================================
            # 4. Check activation condition
            # =================================================

            buffer_ready = (
                len(recent_buffer) == INPUT_STEPS
            )

            speed_ready = (
                current_speed
                >= ACTIVATION_SPEED_KNOTS
            )

            system_active = (
                buffer_ready
                and speed_ready
            )

            # =================================================
            # 5. Predictor inference
            #
            # Sensor acquisition = 100 Hz
            # Predictor inference = 10 Hz
            # =================================================

            control_due = (
                now - last_control_time
                >= CONTROL_PERIOD_SEC
            )

            # 완료 확인은 non-blocking이다. 추론 중 이미 지난
            # prediction 앞부분은 제거하고, 너무 오래된 결과는 버린다.
            try:
                prediction_result = (
                    prediction_worker.poll()
                )

                if prediction_result is not None:
                    latest_prediction_result = (
                        prediction_result
                    )

            except Exception as e:
                print(
                    "[Warning] Predictor 추론 실패: "
                    f"{e}"
                )
                latest_future_trajectory = np.empty(
                    0,
                    dtype=np.float32,
                )
                latest_prediction_result = None

            # 최신 결과를 현재 시각에 다시 맞춘다. 동일 결과를
            # 재사용하는 동안에도 지난 horizon이 계속 제거된다.
            if latest_prediction_result is not None:
                latest_future_trajectory = (
                    latest_prediction_result
                    .aligned_trajectory(
                        now=now,
                        sample_rate_hz=(
                            SENSOR_SAMPLE_RATE_HZ
                        ),
                        max_age_seconds=(
                            MAX_PREDICTION_AGE_SEC
                        ),
                    )
                )

                if latest_future_trajectory.size == 0:
                    print(
                        "[Warning] 최신 prediction이 "
                        "stale하여 폐기되었습니다."
                    )
                    latest_prediction_result = None

            if system_active and control_due:

                # deque -> ndarray
                #
                # shape:
                # (3000, 6)
                imu_window = np.asarray(
                    recent_buffer,
                    dtype=np.float32,
                )

                # worker가 실행 중이면 새 요청을 queue에 쌓지 않는다.
                prediction_worker.submit(
                    imu_window,
                    input_timestamp=now,
                )

            elif not system_active:
                # 저속 또는 history 부족 시 미래 예측을
                # 제어에 사용하지 않는다.
                latest_future_trajectory = np.empty(
                    0,
                    dtype=np.float32,
                )
                latest_prediction_result = None

            # =================================================
            # 6. MSI update
            # =================================================
            #
            # Predictor가 준비되어 있으면
            # 현재 신호 + 필터링된 미래 trajectory 사용.
            #
            # 아니면 현재까지 측정된 데이터만 이용.
            #
            # Predictor의 raw 출력은 MPC에 그대로 사용하고,
            # MSI용 미래 신호만 현재 RealTimeFilter 상태를
            # 복제한 필터로 처리한다. 원본 필터 상태는
            # 미래 신호 처리에 의해 변경되지 않는다.
            # =================================================

            future_msi_trajectory = (
                signal_filter
                .preview_array(
                    latest_future_trajectory
                )
            )

            current_msi, current_msdv = (
                msi_calc.update_and_calculate_future_msi(
                    current_msi_signal,
                    future_msi_trajectory,
                )
            )

            # =================================================
            # 7. MPC + actuator control
            # =================================================

            if control_due:

                if (
                    system_active
                    and latest_future_trajectory.size > 0
                ):
                    target_stroke = (
                        mpc_optimizer
                        .calculate_target_stroke(
                            latest_future_trajectory
                        )
                    )

                else:
                    # 30 knot 미만
                    # 또는 Predictor history 부족
                    # 또는 prediction 실패
                    target_stroke = 0.0

                applied_stroke = actuator.set_stroke(
                    target_stroke
                )

                last_control_time = now

            # =================================================
            # 8. Status log
            # =================================================
            #
            # 콘솔 출력 때문에 100 Hz loop가 느려지는 것을
            # 막기 위해 1초에 한 번만 출력한다.
            # =================================================

            if now - last_log_time >= 1.0:

                history_seconds = (
                    len(recent_buffer)
                    / SENSOR_SAMPLE_RATE_HZ
                )

                state = (
                    "ACTIVE"
                    if system_active
                    else "STANDBY"
                )

                print(
                    f"[{state}] "
                    f"Speed: {current_speed:5.1f} kn | "
                    f"History: {history_seconds:4.1f}"
                    f"/{HISTORY_SECONDS:.0f}s | "
                    f"MSI: {current_msi:6.2f}% | "
                    f"Stroke: {applied_stroke:6.2f} mm"
                )

                last_log_time = now

            # =================================================
            # 9. Keep 100 Hz sampling period
            # =================================================

            next_sensor_time += SENSOR_PERIOD_SEC

            sleep_time = (
                next_sensor_time
                - time.perf_counter()
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

            else:
                # 처리 시간이 너무 길어 sampling schedule을
                # 놓친 경우 누적 지연을 방지한다.
                next_sensor_time = (
                    time.perf_counter()
                )

    except KeyboardInterrupt:
        print(
            "\n사용자에 의해 시스템을 종료합니다."
        )

    except Exception as e:
        print(
            f"\n[Error] 시스템 오류: {e}"
        )

def _cleanup_resources(resources):
    """생성에 성공한 자원만 역방향으로 안전하게 정리한다."""

    # 다른 자원을 닫기 전에 actuator를 안전 위치로 이동한다.
    actuator = resources.actuator
    if actuator is not None:
        try:
            actuator.emergency_stop()
        except Exception as e:
            print(
                "[Warning] actuator emergency_stop "
                f"실패: {e}"
            )
            # emergency_stop 도중 오류가 나도 pigpio 연결은
            # 가능한 범위에서 마지막으로 정리한다.
            pi = getattr(actuator, "pi", None)
            if pi is not None:
                try:
                    pi.stop()
                except Exception as stop_error:
                    print(
                        "[Warning] actuator pigpio "
                        f"정리 실패: {stop_error}"
                    )

    predictor = resources.ai_predictor

    prediction_worker = resources.prediction_worker
    if prediction_worker is not None:
        try:
            prediction_worker.close()
        except Exception as e:
            print(
                "[Warning] inference worker 정리 실패: "
                f"{e}"
            )

    if predictor is not None:
        close_predictor = getattr(
            predictor,
            "close",
            None,
        )
        if callable(close_predictor):
            try:
                close_predictor()
            except Exception as e:
                print(
                    "[Warning] Predictor 정리 실패: "
                    f"{e}"
                )

    gps_sensor = resources.gps_sensor
    if gps_sensor is not None:
        try:
            close_gps = getattr(
                gps_sensor,
                "close",
                None,
            )
            if callable(close_gps):
                close_gps()
            else:
                serial_port = getattr(
                    gps_sensor,
                    "ser",
                    None,
                )
                if serial_port is not None:
                    serial_port.close()
        except Exception as e:
            print(
                "[Warning] GPS 정리 실패: "
                f"{e}"
            )

    imu_sensor = resources.imu_sensor
    if imu_sensor is not None:
        try:
            imu_sensor.close()
        except Exception as e:
            print(
                "[Warning] IMU 정리 실패: "
                f"{e}"
            )


def main():
    resources = _RuntimeResources()

    try:
        _run_system(resources)
    except KeyboardInterrupt:
        print(
            "\n사용자에 의해 시스템을 종료합니다."
        )
    except Exception as e:
        print(
            f"\n[Error] 시스템 오류: {e}"
        )
    finally:
        _cleanup_resources(resources)
        print("시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    main()
