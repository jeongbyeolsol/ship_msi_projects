import math
import time

import numpy as np
import smbus2


# Predictor와 데이터셋에서 사용하는 채널 순서
IMU_CHANNELS = (
    "imu_acc_x_mps2",
    "imu_acc_y_mps2",
    "imu_acc_z_mps2",
    "imu_gyro_x_rad_s",
    "imu_gyro_y_rad_s",
    "imu_gyro_z_rad_s",
)


class MPU6050:
    """
    MPU6050 6축 IMU 인터페이스.

    read_imu() 반환 형식:
        np.ndarray, shape=(6,), dtype=np.float32

        [
            acc_x_mps2,
            acc_y_mps2,
            acc_z_mps2,
            gyro_x_rad_s,
            gyro_y_rad_s,
            gyro_z_rad_s,
        ]

    주의:
        센서의 실제 장착 방향이 V17 데이터셋의 선박 좌표계와
        동일해야 한다. 축 방향이 다를 경우 추후 axis mapping을
        추가해야 한다.
    """

    # MPU6050 register
    PWR_MGMT_1 = 0x6B
    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG = 0x1B
    ACCEL_XOUT_H = 0x3B

    # V17 전체 데이터에서 ±8 g를 넘는 수직 충격이
    # 관측되었으므로, MPU6050이 지원하는 최대 범위인
    # ±16 g를 사용해 실제 입력의 포화를 최소화한다.
    ACCEL_LSB_PER_G = 2048.0

    # ±250 deg/s
    GYRO_LSB_PER_DPS = 131.0

    # 프로젝트에서 사용하는 중력가속도
    GRAVITY_MPS2 = 9.81

    def __init__(
        self,
        bus_num=1,
        address=0x68,
        acc_bias_mps2=None,
        gyro_bias_rad_s=None,
    ):
        self.bus = smbus2.SMBus(bus_num)
        self.address = address

        # MPU6050 wake-up
        self.bus.write_byte_data(
            self.address,
            self.PWR_MGMT_1,
            0x00,
        )

        # Accelerometer: ±16 g (AFS_SEL=3, bits 4:3 = 11)
        self.bus.write_byte_data(
            self.address,
            self.ACCEL_CONFIG,
            0x18,
        )

        # Gyroscope: ±250 deg/s
        self.bus.write_byte_data(
            self.address,
            self.GYRO_CONFIG,
            0x00,
        )

        # 센서가 깨어날 시간을 조금 확보
        time.sleep(0.05)

        # Bias는 최종 Predictor 입력과 동일한 SI 단위로 관리한다.
        self.acc_bias_mps2 = self._make_bias(
            acc_bias_mps2,
            name="acc_bias_mps2",
        )

        self.gyro_bias_rad_s = self._make_bias(
            gyro_bias_rad_s,
            name="gyro_bias_rad_s",
        )

        # 기존 pure vertical acceleration 계산을
        # 임시로 유지하기 위한 자세 추정 상태
        self.pitch = 0.0
        self.roll = 0.0
        self.last_time = time.perf_counter()

    @staticmethod
    def _make_bias(value, name):
        """
        bias를 shape=(3,) float32 배열로 변환한다.
        """
        if value is None:
            return np.zeros(3, dtype=np.float32)

        bias = np.asarray(value, dtype=np.float32)

        if bias.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,), "
                f"got {bias.shape}"
            )

        return bias

    @staticmethod
    def _to_signed_16(high_byte, low_byte):
        """
        MPU6050의 두 바이트를 signed int16 값으로 변환한다.
        """
        value = (high_byte << 8) | low_byte

        if value & 0x8000:
            value -= 0x10000

        return value

    def _read_raw_imu(self):
        """
        Accel + temperature + gyro register 14 bytes를
        한 번의 I2C transaction으로 읽는다.

        Returns:
            ax_raw, ay_raw, az_raw,
            gx_raw, gy_raw, gz_raw
        """

        data = self.bus.read_i2c_block_data(
            self.address,
            self.ACCEL_XOUT_H,
            14,
        )

        ax_raw = self._to_signed_16(data[0], data[1])
        ay_raw = self._to_signed_16(data[2], data[3])
        az_raw = self._to_signed_16(data[4], data[5])

        # data[6], data[7]은 temperature이므로 사용하지 않음

        gx_raw = self._to_signed_16(data[8], data[9])
        gy_raw = self._to_signed_16(data[10], data[11])
        gz_raw = self._to_signed_16(data[12], data[13])

        return (
            ax_raw,
            ay_raw,
            az_raw,
            gx_raw,
            gy_raw,
            gz_raw,
        )

    def read_imu(self):
        """
        Predictor 입력용 6축 IMU 샘플을 반환한다.

        Returns:
            np.ndarray:
                shape = (6,)
                dtype = np.float32

                [
                    acc_x_mps2,
                    acc_y_mps2,
                    acc_z_mps2,
                    gyro_x_rad_s,
                    gyro_y_rad_s,
                    gyro_z_rad_s,
                ]
        """

        (
            ax_raw,
            ay_raw,
            az_raw,
            gx_raw,
            gy_raw,
            gz_raw,
        ) = self._read_raw_imu()

        # --------------------------------------------------
        # Accelerometer
        # raw -> g -> m/s^2
        # --------------------------------------------------

        accel = (
            np.array(
                [ax_raw, ay_raw, az_raw],
                dtype=np.float32,
            )
            / self.ACCEL_LSB_PER_G
            * self.GRAVITY_MPS2
        )

        accel -= self.acc_bias_mps2

        # --------------------------------------------------
        # Gyroscope
        # raw -> deg/s -> rad/s
        # --------------------------------------------------

        gyro_dps = (
            np.array(
                [gx_raw, gy_raw, gz_raw],
                dtype=np.float32,
            )
            / self.GYRO_LSB_PER_DPS
        )

        gyro = np.deg2rad(gyro_dps).astype(np.float32)

        gyro -= self.gyro_bias_rad_s

        # --------------------------------------------------
        # Predictor input
        # --------------------------------------------------

        return np.concatenate(
            [accel, gyro],
        ).astype(np.float32, copy=False)

    def read_pure_linear_accel(self):
        """
        기존 시스템과의 임시 호환용 함수.

        중력을 제거한 Z축 선형 가속도를 반환한다.

        새 Predictor에서는 이 값을 입력으로 사용하지 않고,
        read_imu()의 6축 데이터를 사용한다.

        Returns:
            float: vertical linear acceleration [m/s^2]
        """

        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time

        sample = self.read_imu()

        acc_x, acc_y, acc_z = sample[:3]
        gyro_x, gyro_y, _ = sample[3:]

        # rad/s -> deg/s
        gyro_x_dps = math.degrees(float(gyro_x))
        gyro_y_dps = math.degrees(float(gyro_y))

        # Accelerometer 기반 자세 추정
        acc_pitch = math.degrees(
            math.atan2(
                acc_y,
                math.sqrt(acc_x**2 + acc_z**2),
            )
        )

        acc_roll = math.degrees(
            math.atan2(
                -acc_x,
                acc_z,
            )
        )

        # Complementary filter
        alpha = 0.98

        self.pitch = (
            alpha * (self.pitch + gyro_x_dps * dt)
            + (1.0 - alpha) * acc_pitch
        )

        self.roll = (
            alpha * (self.roll + gyro_y_dps * dt)
            + (1.0 - alpha) * acc_roll
        )

        pitch_rad = math.radians(self.pitch)
        roll_rad = math.radians(self.roll)

        gravity_z = (
            math.cos(pitch_rad)
            * math.cos(roll_rad)
            * self.GRAVITY_MPS2
        )

        return float(acc_z - gravity_z)

    def close(self):
        """
        I2C bus를 닫는다.
        """
        if self.bus is not None:
            self.bus.close()
            self.bus = None
