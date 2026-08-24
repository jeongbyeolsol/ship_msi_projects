import time
from collections import deque

import numpy as np

from imu import MPU6050
from filter import RealTimeFilter
from predictor import Predictor
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

# 모델 입력 history
HISTORY_SECONDS = 30.0
INPUT_STEPS = int(SENSOR_SAMPLE_RATE_HZ * HISTORY_SECONDS)

# Predictor / MPC 제어 갱신 주기
CONTROL_RATE_HZ = 10.0
CONTROL_PERIOD_SEC = 1.0 / CONTROL_RATE_HZ

# GPS는 IMU처럼 100 Hz로 읽을 필요가 없음
GPS_RATE_HZ = 5.0
GPS_PERIOD_SEC = 1.0 / GPS_RATE_HZ

# 인터셉터 작동 조건
ACTIVATION_SPEED_KNOTS = 30.0

# Predictor checkpoint
MODEL_PATH = "model/checkpoints/best.pt"


def main():
    print("시스템 초기화 중...")

    # --------------------------------------------------------
    # 1. Hardware / module initialization
    # --------------------------------------------------------

    try:
        imu_sensor = MPU6050()
        print("[IMU] MPU6050 초기화 완료")
    except Exception as e:
        print(f"[Error] IMU 초기화 실패: {e}")
        return

    # GPS 클래스 내부에서 연결 실패를 처리하고
    # 실패 시 speed=0.0 knot를 반환한다.
    gps_sensor = GPSSpeedSensor(
        port="/dev/ttyUSB0",
        baudrate=9600,
    )

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
        ai_predictor = Predictor(
            checkpoint_path=MODEL_PATH
        )
    except Exception as e:
        print(f"[Error] Predictor 초기화 실패: {e}")
        imu_sensor.close()
        return

    msi_calc = MSICalculator(
        fs=SENSOR_SAMPLE_RATE_HZ,
        window_minutes=20.0,
    )

    mpc_optimizer = LightMPC(
        control_weight=0.1
    )

    actuator = InterceptorController(
        pin=18,
        min_stroke=0.0,
        max_stroke=50.0,
    )

    # --------------------------------------------------------
    # 2. Predictor input buffer
    #
    # 각 원소:
    # [ax, ay, az, gx, gy, gz]
    #
    # 전체 shape:
    # (INPUT_STEPS, 6)
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
            if imu_sample.shape != (6,):
                raise ValueError(
                    "read_imu() must return shape (6,), "
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

            if system_active and control_due:

                # deque -> ndarray
                #
                # shape:
                # (3000, 6)
                imu_window = np.asarray(
                    recent_buffer,
                    dtype=np.float32,
                )

                try:
                    latest_future_trajectory = (
                        ai_predictor.predict(
                            imu_window
                        )
                    )

                    latest_future_trajectory = (
                        np.asarray(
                            latest_future_trajectory,
                            dtype=np.float32,
                        ).reshape(-1)
                    )

                    if not np.all(
                        np.isfinite(
                            latest_future_trajectory
                        )
                    ):
                        raise ValueError(
                            "Predictor output contains "
                            "NaN or Inf."
                        )

                except Exception as e:
                    print(
                        "[Warning] Predictor 추론 실패: "
                        f"{e}"
                    )

                    latest_future_trajectory = (
                        np.empty(
                            0,
                            dtype=np.float32,
                        )
                    )

            elif not system_active:
                # 저속 또는 history 부족 시 미래 예측을
                # 제어에 사용하지 않는다.
                latest_future_trajectory = np.empty(
                    0,
                    dtype=np.float32,
                )

            # =================================================
            # 6. MSI update
            # =================================================
            #
            # Predictor가 준비되어 있으면
            # 현재 신호 + 미래 trajectory 사용.
            #
            # 아니면 현재까지 측정된 데이터만 이용.
            # =================================================

            current_msi, current_msdv = (
                msi_calc.update_and_calculate_future_msi(
                    current_msi_signal,
                    latest_future_trajectory,
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

    finally:
        # -----------------------------------------------------
        # Fail-safe shutdown
        # -----------------------------------------------------

        try:
            actuator.emergency_stop()
        except Exception:
            pass

        try:
            imu_sensor.close()
        except Exception:
            pass

        # 현재 gps.py에는 close()가 없으므로
        # serial 객체가 존재하면 직접 닫는다.
        try:
            if (
                hasattr(gps_sensor, "ser")
                and gps_sensor.ser is not None
            ):
                gps_sensor.ser.close()
        except Exception:
            pass

        print("시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    main()