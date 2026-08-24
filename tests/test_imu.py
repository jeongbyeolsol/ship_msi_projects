import numpy as np

import imu


class _FakeBus:
    def __init__(self, bus_num):
        self.bus_num = bus_num
        self.writes = []

    def write_byte_data(self, address, register, value):
        self.writes.append((address, register, value))


def test_mpu6050_uses_16g_accelerometer_and_keeps_250dps_gyro(
    monkeypatch,
):
    bus = _FakeBus(1)
    monkeypatch.setattr(imu.smbus2, "SMBus", lambda _: bus)
    monkeypatch.setattr(imu.time, "sleep", lambda _: None)

    sensor = imu.MPU6050()

    assert (
        sensor.address,
        sensor.ACCEL_CONFIG,
        0x18,
    ) in bus.writes
    assert (
        sensor.address,
        sensor.GYRO_CONFIG,
        0x00,
    ) in bus.writes
    assert sensor.ACCEL_LSB_PER_G == 2048.0
    assert sensor.GYRO_LSB_PER_DPS == 131.0

    monkeypatch.setattr(
        sensor,
        "_read_raw_imu",
        lambda: (2048, -2048, 0, 131, -131, 0),
    )
    sample = sensor.read_imu()

    assert sample.shape == (6,)
    assert sample.dtype == np.float32
    np.testing.assert_allclose(
        sample[:3],
        [9.81, -9.81, 0.0],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample[3:],
        np.deg2rad([1.0, -1.0, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )
